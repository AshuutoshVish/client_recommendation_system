import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine, URL
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer


class ClientAutoencoder(nn.Module):
    """Autoencoder model for client embedding generation."""

    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def to_tensor(X):
    """Convert input to PyTorch tensor."""
    return torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else X)


def get_db_engine():
    """Create a database engine using environment variables."""
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError("Missing one or more required environment variables.")

    db_url = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_password,
        host=db_host,
        port=int(db_port),
        database=db_name
    )
    engine = create_engine(db_url)
    # Test the connection
    engine.connect().close()
    print("Database connection is made successfully..!\n")
    return engine


def process_catcher_data(engine):
    """Process data from SQL database and return a clean dataframe."""
    print("Processing data from SQL database...")

    # Get catcher data
    catcher_query = """
    SELECT
        id as jobma_catcher_id,
        is_premium,
        subscription_status,
        company_size
    FROM jobma_catcher
    WHERE is_deleted = 0 AND is_active = 1
    """
    catcher_df = pd.read_sql(catcher_query, engine)

    # Get wallet data
    wallet_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        amount as wallet_amount,
        is_unlimited
    FROM wallet
    WHERE is_deleted = 0
    """
    wallet_df = pd.read_sql(wallet_query, engine)

    # Get subscription data
    subscription_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        SUM(amount) as subscription_sum,
        COUNT(*) as subscription_count
    FROM subscription_history
    WHERE is_deleted = 0
    GROUP BY catcher_id
    """
    subscription_df = pd.read_sql(subscription_query, engine)

    # Get interview data
    interview_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        COUNT(CASE WHEN interview_type = 'recorded' THEN 1 END) as recorded_interview_count,
        COUNT(CASE WHEN interview_type = 'live' THEN 1 END) as live_interview_count,
        COUNT(*) as invites_count
    FROM jobma_pitcher_invitations
    WHERE is_deleted = 0
    GROUP BY catcher_id
    """
    interview_df = pd.read_sql(interview_query, engine)

    # Get interview done data
    interview_done_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        COUNT(*) as interview_done
    FROM jobma_pitcher_invitations
    WHERE is_deleted = 0 AND status = 'done'
    GROUP BY catcher_id
    """
    interview_done_df = pd.read_sql(interview_done_query, engine)

    # Get kit data
    kit_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        COUNT(*) as pre_recorded_kit_counts
    FROM job_assessment_kit
    WHERE is_deleted = 0
    GROUP BY catcher_id
    """
    kit_df = pd.read_sql(kit_query, engine)

    # Get job posting data
    job_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        COUNT(*) as jobs_posted
    FROM jobma_employer_job_posting
    WHERE is_deleted = 0
    GROUP BY catcher_id
    """
    job_df = pd.read_sql(job_query, engine)

    # Get login data
    login_query = """
    SELECT
        catcher_id as jobma_catcher_id,
        DATEDIFF(CURRENT_DATE, MAX(login_time)) as since_last_login
    FROM jobma_login
    GROUP BY catcher_id
    """
    login_df = pd.read_sql(login_query, engine)

    # Get sub user data
    subuser_query = """
    SELECT
        parent_id as jobma_catcher_id,
        COUNT(*) as sub_user
    FROM jobma_catcher
    WHERE is_deleted = 0 AND parent_id IS NOT NULL
    GROUP BY parent_id
    """
    subuser_df = pd.read_sql(subuser_query, engine)

    # Merge all dataframes
    merged_df = catcher_df
    for df in [wallet_df, subscription_df, interview_df, interview_done_df,
              kit_df, job_df, login_df, subuser_df]:
        merged_df = pd.merge(merged_df, df, on='jobma_catcher_id', how='left')

    # Fill missing values
    merged_df['wallet_amount'] = merged_df['wallet_amount'].fillna(0)
    merged_df['is_unlimited'] = merged_df['is_unlimited'].fillna(0)
    merged_df['subscription_sum'] = merged_df['subscription_sum'].fillna(0)
    merged_df['subscription_count'] = merged_df['subscription_count'].fillna(0)
    merged_df['recorded_interview_count'] = merged_df['recorded_interview_count'].fillna(0)
    merged_df['live_interview_count'] = merged_df['live_interview_count'].fillna(0)
    merged_df['invites_count'] = merged_df['invites_count'].fillna(0)
    merged_df['interview_done'] = merged_df['interview_done'].fillna(0)
    merged_df['pre_recorded_kit_counts'] = merged_df['pre_recorded_kit_counts'].fillna(0)
    merged_df['jobs_posted'] = merged_df['jobs_posted'].fillna(0)
    merged_df['since_last_login'] = merged_df['since_last_login'].fillna(30)
    merged_df['sub_user'] = merged_df['sub_user'].fillna(0)

    print(f"Processed data: {len(merged_df)} records")
    return merged_df


def get_or_create_processed_data(path='new/new_processed_data.csv'):
    """Load data from CSV if exists, otherwise fetch from SQL and save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"Loading existing file from: {path}")
        df = pd.read_csv(path)
    else:
        print(f"No existing file found...!!")
        print(f"Creating new data at: {path}")
        engine = get_db_engine()
        df = process_catcher_data(engine)
        df.to_csv(path, index=False)
        print(f"Database is created at {path}...!")
    return df


def get_preprocessor():
    """Create a preprocessing pipeline for the data."""
    categorical = ['since_last_login']
    ordinal = ['company_size']
    binary = ['is_premium', 'is_unlimited', 'subscription_status']
    numeric = ['sub_user', 'subscription_count', 'live_interview_count', 'interview_done',
              'pre_recorded_kit_counts', 'invites_count', 'jobs_posted']
    log_scaled = ['subscription_sum', 'wallet_amount']

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical),
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=[["1-25", "26-100", "101-500", "500-1000", "More than 1000"]],
                                      handle_unknown='use_encoded_value', unknown_value=-1))
        ]), ordinal),
        ('bin', SimpleImputer(strategy='mean'), binary),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ]), numeric),
        ('log', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('log', FunctionTransformer(np.log1p, validate=False)),
            ('scale', StandardScaler())
        ]), log_scaled)
    ])
    return preprocessor


def get_or_create_preprocessor(df, pipeline_path='new_processor/new_preprocessor.joblib'):
    """Load preprocessor if exists, otherwise create and save."""
    if os.path.exists(pipeline_path):
        print(f"Loading preprocessor from {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
    else:
        print(f"Creating new preprocessor...")
        pipeline = get_preprocessor()
        pipeline.fit(df)
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(pipeline, pipeline_path)
        print(f"Preprocessor created and saved to {pipeline_path}")
    return pipeline


def train_autoencoder(X_all, model_path, input_dim):
    """Train an autoencoder model and save it."""
    print("Training started")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    epochs = 100
    batch_size = 32
    patience = 10
    X_train, X_val = train_test_split(X_all, test_size=0.2, random_state=42)
    train_tensor = to_tensor(X_train)
    val_tensor = to_tensor(X_val)

    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClientAutoencoder(input_dim).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, _ = batch
                inputs = inputs.to(device)
                _, outputs = model(inputs)
                val_loss += criterion(outputs, inputs).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_path)
    return model


def get_or_create_model(df_features, model_path='new_model/client_autoencoder.pth'):
    """Load model if exists, otherwise train and save."""
    input_dim = int(df_features.shape[1])
    model = ClientAutoencoder(input_dim)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Training new model...")
        model = train_autoencoder(df_features, model_path, input_dim)
    return model


def get_or_create_embeddings(model, X, embedding_path='new_embeddings/embeddings.npy'):
    """Load embeddings if exists, otherwise create and save."""
    if os.path.exists(embedding_path):
        # Load embeddings directly into memory for faster access
        embeddings = np.load(embedding_path, mmap_mode='r')  # Memory-mapped mode for faster loading
        return torch.tensor(embeddings, dtype=torch.float32)
    else:
        # Extract embeddings efficiently
        model.eval()
        with torch.no_grad():
            tensor = to_tensor(X)
            embeddings, _ = model(tensor)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, embeddings.cpu().numpy())
        return embeddings


def recommend_sample(user_inputs,
                     pipeline_path='new_processor/new_preprocessor.joblib',
                     model_path='new_model/client_autoencoder.pth',
                     embedding_path='new_embeddings/embeddings.npy',
                     top_n=5,
                     debug=False):
    """
    Generate recommendations based on user inputs.

    Args:
        user_inputs (dict): Dictionary of user features
        pipeline_path (str): Path to the preprocessor
        model_path (str): Path to the model
        embedding_path (str): Path to the embeddings
        top_n (int): Number of recommendations to return
        debug (bool): Whether to print debug information

    Returns:
        DataFrame: Recommendations with similarity scores
    """
    global df
    try:
        # Use references instead of copying for better performance
        if 'jobma_catcher_id' in df.columns:
            id_column = df['jobma_catcher_id']
            df_features_cols = df.drop('jobma_catcher_id', axis=1)
        else:
            id_column = None
            df_features_cols = df

        # Load preprocessor (should be already created)
        pipeline = joblib.load(pipeline_path)

        # Transform user input directly without creating a dataframe first
        user_features = pipeline.transform(pd.DataFrame([user_inputs]))

        # Load model (should be already created)
        input_dim = pipeline.transform(df_features_cols.iloc[:1]).shape[1]
        model = ClientAutoencoder(input_dim)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Load embeddings (should be already created)
        df_embeddings = torch.tensor(np.load(embedding_path, mmap_mode='r'), dtype=torch.float32)

        # Get user embedding efficiently
        model.eval()
        with torch.no_grad():
            user_tensor = to_tensor(user_features)
            user_embedding, _ = model(user_tensor)
            user_embedding = F.normalize(user_embedding, p=2, dim=1)

        # Calculate similarities using optimized matrix multiplication
        similarities = torch.matmul(user_embedding, df_embeddings.t())

        # Print debug information only if requested
        if debug:
            print("\n--- Debug Information ---")
            print(f"Shape of similarities: {similarities.shape}")
            print(f"Shape of all_embeddings: {df_embeddings.shape}")
            print(f"Shape of user_embedding: {user_embedding.shape}")
            print(f"Length of dataframe: {len(df_features_cols)}")
            print(f"Available N: {min(top_n, len(df_features_cols))}")

        # Get top indices efficiently
        top_indices = similarities[0].topk(min(top_n, len(df_features_cols))).indices.cpu().numpy()

        if debug:
            print(f"Top indices: {top_indices}")
            print("------------------------\n")

        # Create recommendations dataframe efficiently
        if id_column is not None:
            # Use vectorized operations for better performance
            result = pd.DataFrame({
                'jobma_catcher_id': id_column.iloc[top_indices].values,
                'similarity': similarities[0, top_indices].cpu().numpy()
            })
            # Use efficient merge with only necessary columns
            result = pd.merge(result, df, on='jobma_catcher_id', how='left')
        else:
            # Use iloc for better performance
            result = df.iloc[top_indices].copy()
            result['similarity'] = similarities[0, top_indices].cpu().numpy()

        return result

    except Exception as e:
        print(f"Error in recommend_sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def recommendations(debug=False):
    """
    Test the recommendation system with a sample user

    Args:
        debug (bool): Whether to print debug information
    """
    global df
    import time

    sample_user = {
        'company_size': '1-25',
        'is_premium': 0,
        'subscription_status': 1,
        'sub_user': 5,
        'wallet_amount': 60.0,
        'is_unlimited': 0,
        'subscription_sum': 0.6,
        'subscription_count': 1,
        'recorded_interview_count': 4,
        'live_interview_count': 2,
        'invites_count': 7,
        'interview_done': 5,
        'pre_recorded_kit_counts': 2,
        'jobs_posted': 10,
        'since_last_login': 3
    }

    if debug:
        print(sample_user)

    try:
        start_time = time.time()
        print("Generating recommendations for sample user...")

        recommendations = recommend_sample(
            user_inputs=sample_user,
            pipeline_path='new_processor/new_preprocessor.joblib',
            model_path='new_model/client_autoencoder.pth',
            embedding_path='new_embeddings/embeddings.npy',
            top_n=5,
            debug=debug
        )

        end_time = time.time()
        execution_time = end_time - start_time

        if not recommendations.empty:
            print("\nRecommendations from model:")
            pd.set_option('display.max_columns', None)  # Show all columns
            print(recommendations)
            print(f"\nRecommendation successful! (Execution time: {execution_time:.2f} seconds)")
        else:
            print("\nNo recommendations generated.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


# Initialize global variables
df = None

def initialize_system(data_path='new/new_processed_data.csv',
                     pipeline_path='new_processor/new_preprocessor.joblib',
                     model_path='new_model/client_autoencoder.pth',
                     embedding_path='new_embeddings/embeddings.npy'):
    """
    Initialize the recommendation system by loading or creating all necessary components.
    This should be called once at the start of the application.

    Args:
        data_path: Path to the data file
        pipeline_path: Path to the preprocessor file
        model_path: Path to the model file
        embedding_path: Path to the embeddings file
    """
    global df
    import time

    start_time = time.time()

    # Load data
    df = get_or_create_processed_data(data_path)
    print("Dataframe loaded successfully")

    # Prepare data for model
    df_copy = df.copy()
    if 'jobma_catcher_id' in df_copy.columns:
        df_copy = df_copy.drop('jobma_catcher_id', axis=1)

    # Load or create preprocessor
    pipeline = get_or_create_preprocessor(df_copy, pipeline_path)

    # Transform the dataframe
    df_features = pipeline.transform(df_copy)

    # Load or create model
    model = get_or_create_model(df_features, model_path)

    # Load or create embeddings
    _ = get_or_create_embeddings(model, df_features, embedding_path)

    end_time = time.time()
    print(f"System initialized in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Initialize the system (load all components)
    initialize_system()

    # Generate recommendations
    recommendations(debug=False)
