import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from dotenv import load_dotenv

def get_db_engine():
    from sqlalchemy import create_engine, URL
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
    with engine.connect() as conn:
        print("Database connection is made successfully..!\n")
    return engine

def create_dataframe(engine):
    def clean_catcher_data():
        catcher = pd.read_sql("SELECT jobma_catcher_id, is_premium, subscription_status, company_size, jobma_catcher_parent FROM jobma_catcher WHERE jobma_verified IN (1, '1')", engine).copy()
        catcher.replace("", np.nan, inplace=True)
        catcher['is_premium'] = pd.to_numeric(catcher['is_premium'], errors='coerce').fillna(0).astype(int)
        catcher['subscription_status'] = pd.to_numeric(catcher['subscription_status'], errors='coerce')
        catcher['subscription_status'] = catcher['subscription_status'].replace({2: 0}).fillna(0).astype(int)
        sub_accounts = catcher['jobma_catcher_parent'].value_counts()
        catcher['sub_user'] = catcher['jobma_catcher_id'].map(sub_accounts).fillna(0).astype(int)
        mode_val = catcher['company_size'].mode()[0]
        catcher['company_size'] = catcher['company_size'].fillna(mode_val)
        return catcher

    def clean_wallet_data():
        wallet = pd.read_sql("SELECT catcher_id AS jobma_catcher_id, wallet_amount, is_unlimited FROM wallet", engine).copy()
        wallet['wallet_amount'] = pd.to_numeric(wallet['wallet_amount'], errors='coerce').fillna(0)
        wallet['is_unlimited'] = pd.to_numeric(wallet['is_unlimited'], errors='coerce').fillna(0).astype(int)
        return wallet

    def clean_subscription_data():
        subscription = pd.read_sql("SELECT catcher_id AS jobma_catcher_id, subscription_amount, currency FROM subscription_history", engine).copy()
        subscription['subscription_amount'] = pd.to_numeric(subscription['subscription_amount'], errors='coerce').fillna(0)
        subscription['subscription_amount'] = subscription['subscription_amount'].where(subscription['subscription_amount'] > 0, 0)
        subscription['subscription_amount'] = np.where(subscription['currency'] == 0,subscription['subscription_amount'].round(1),(subscription['subscription_amount'] / 85).round(1))
        subscription = subscription.groupby('jobma_catcher_id').agg(subscription_sum=('subscription_amount', 'sum'),subscription_count=('jobma_catcher_id', 'count')).reset_index()
        return subscription

    def clean_login_data():
        login = pd.read_sql("SELECT jobma_user_id AS jobma_catcher_id, jobma_last_login FROM jobma_login", engine)
        login['jobma_last_login'] = pd.to_datetime(login['jobma_last_login'], errors='coerce')
        today = pd.to_datetime('2024-06-10 11:24:30')  #Need improvements
        login['since_last_login'] = (today - login['jobma_last_login']).dt.days
        login = login.groupby('jobma_catcher_id').agg(since_last_login=('since_last_login', 'min')).reset_index()
        since_login_filled = login['since_last_login'].fillna(float('inf'))
        bins = [0, 30, 90, 180, 365, float('inf')]
        labels = [0, 1, 2, 3, 4]
        login['since_last_login'] = pd.cut(since_login_filled, bins=bins, labels=labels, right=False)
        return login

    def clean_invitations_data():
        invitations = pd.read_sql("SELECT jobma_catcher_id, jobma_interview_mode, jobma_interview_status FROM jobma_pitcher_invitations", engine)
        recorded = invitations[invitations['jobma_interview_mode'].isin([1, '1'])].groupby('jobma_catcher_id').size().reset_index(name='recorded_interview_count')
        live = invitations[invitations['jobma_interview_mode'].isin([2, '2'])].groupby('jobma_catcher_id').size().reset_index(name='live_interview_count')
        invites = invitations[invitations['jobma_interview_status'].isin([0, '0'])].groupby('jobma_catcher_id').size().reset_index(name='invites_count')
        done = invitations[~invitations['jobma_interview_status'].isin([0, '0'])].groupby('jobma_catcher_id').size().reset_index(name='interview_done')
        summary = pd.merge(recorded, live, on='jobma_catcher_id', how='left')
        summary = pd.merge(summary, invites, on='jobma_catcher_id', how='left')
        summary = pd.merge(summary, done, on='jobma_catcher_id', how='left')
        summary.fillna(0, inplace=True)
        return summary

    def clean_pre_recorded_kit_data():
        kit = pd.read_sql("SELECT catcher_id AS jobma_catcher_id FROM job_assessment_kit", engine)
        return kit.groupby('jobma_catcher_id').size().reset_index(name='pre_recorded_kit_counts')

    def clean_job_posting_data():
        postings = pd.read_sql("SELECT jobma_catcher_id FROM jobma_employer_job_posting", engine).copy()
        return postings.groupby('jobma_catcher_id').size().reset_index(name='jobs_posted')

    def working_on_merged_df(work_df):
        columns_to_sum = ['recorded_interview_count', 'live_interview_count', 'invites_count', 'interview_done', 'pre_recorded_kit_counts', 'jobs_posted']
        work_df[columns_to_sum] = work_df[columns_to_sum].fillna(0).astype(int)
        child_sums = work_df[work_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')[columns_to_sum].sum()
        for col in columns_to_sum:
            work_df.loc[work_df['jobma_catcher_parent'] == 0, col] += work_df.loc[
                work_df['jobma_catcher_parent'] == 0, 'jobma_catcher_id'].map(child_sums[col]).fillna(0)

        sub_ac = work_df[work_df['jobma_catcher_parent'] != 0]
        min_login = sub_ac.groupby('jobma_catcher_parent')['since_last_login'].min().reset_index().rename(columns={'since_last_login': 'min_login'})
        work_df = work_df.merge(min_login, how='left', on='jobma_catcher_parent')
        work_df['since_last_login'] = np.where(work_df['jobma_catcher_parent'] != 0,work_df['min_login'],work_df['since_last_login'])
        work_df.drop(columns=['min_login'], inplace=True)

        work_df = work_df[work_df['jobma_catcher_parent'] == 0].drop('jobma_catcher_parent', axis=1)
        work_df['wallet_amount'] = work_df['wallet_amount'].fillna(0).round(1)
        work_df['subscription_sum'] = work_df['subscription_sum'].fillna(0).round(1)
        work_df['subscription_count'] = work_df['subscription_count'].fillna(0).astype(int)
        work_df['is_unlimited'] = work_df['is_unlimited'].replace('', 1).fillna(0).astype(int)
        work_df['since_last_login'] = work_df['since_last_login'].fillna(4).astype(int)
        return work_df

    catcher_df = clean_catcher_data()
    other_dfs = [clean_wallet_data(), clean_subscription_data(), clean_invitations_data(), clean_pre_recorded_kit_data(), clean_job_posting_data(), clean_login_data()]
    merged_df = catcher_df.copy()
    for df in other_dfs:
        merged_df = pd.merge(merged_df, df, how='left', on='jobma_catcher_id')

    final_df = working_on_merged_df(merged_df)
    return final_df



def get_or_create_dataframe(data_path='exp/data.csv'):
    dir_path = os.path.dirname(data_path)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory exists after makedirs: {os.path.exists(dir_path)}")

    if os.path.exists(data_path):
        print(f"Loading existing dataframe from storage  :-- {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"No existing now Creating new dataframe at: {data_path}")
        engine = get_db_engine()
        df = create_dataframe(engine)
        df.to_csv(data_path, index=False)
        print(f"Database is created at {data_path}...!")
    return df



def create_preprocessor():
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
    print("create_preprocessor function is runnning.......................................!!")
    categorical = ['since_last_login']
    ordinal = ['company_size']
    binary = ['is_premium', 'is_unlimited', 'subscription_status']
    numeric = ['sub_user', 'subscription_count', 'live_interview_count', 'interview_done', 'pre_recorded_kit_counts', 'invites_count', 'jobs_posted']
    log_scaled = ['subscription_sum', 'wallet_amount']

    preprocessor = ColumnTransformer([('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                                        ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical),
                                      ('ord', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                                        ('ordinal', OrdinalEncoder(categories=[["1-25", "26-100", "101-500", "500-1000", "More than 1000"]],
                                                                                   handle_unknown='use_encoded_value', unknown_value=-1))]), ordinal),
                                      ('bin', Pipeline([('imputer', SimpleImputer(strategy='mean'))]), binary),
                                      ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                                        ('scale', StandardScaler())]), numeric),
                                      ('log', Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                                        ('log', FunctionTransformer(np.log1p, validate=False)),
                                                        ('scale', StandardScaler())]), log_scaled)],
                                                        remainder='drop')
    return preprocessor

def get_or_create_preprocessor(preprocessor_path='exp/preprocessor.joblib', fit=True):
    dir_path = os.path.dirname(preprocessor_path)
    os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded from storage :--- {preprocessor_path}")
    elif fit:
        preprocessor = create_preprocessor()
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor trained and saved to: {preprocessor_path}")
    else:
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path} and fit=False.")
    return preprocessor

#model
class ClientAutoencoder(nn.Module):
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
    return torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else X)

def train_autoencoder(X_all, model_path, input_dim):
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    dir_path = os.path.dirname(model_path)
    os.makedirs(dir_path, exist_ok=True)
    #Can be changed or dynamic
    epochs=100
    batch_size=32
    patience= 5
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    X_train, X_val = train_test_split(X_all, test_size=0.2, random_state=42)
    train_tensor = to_tensor(X_train)
    val_tensor = to_tensor(X_val)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor),
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor),
                            batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClientAutoencoder(input_dim).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

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

        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {train_loss:.4f} | Validity Loss: {val_loss:.4f}")

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
        print(f"Model stored at : {model_path}")
    return model



def load_or_train_autoencoder(X, model_path='exp/Client_autoencoder.pth'):
    input_dim = X.shape[1]
    model = ClientAutoencoder(input_dim)
    dir_path = os.path.dirname(model_path)
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded from storage :----- {model_path}")
    else:
        print("No model found. Training new model...!!")
        model = train_autoencoder(X, model_path, input_dim)
    return model


def get_or_create_embeddings(X, model, embedding_path='exp/embeddings.npy'):
    import torch.nn.functional as F

    dir_path = os.path.dirname(embedding_path)
    os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(embedding_path):
        print(f"Loading embeddings from storage :---- {embedding_path}")
        embeddings = np.load(embedding_path)
        return torch.tensor(embeddings, dtype=torch.float32)
    else:
        print("Extracting and saving new embeddings...")
        model.eval()
        with torch.no_grad():
            tensor = to_tensor(X)
            embeddings, _ = model(tensor)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        np.save(embedding_path, embeddings.cpu().numpy())
        return embeddings


df = get_or_create_dataframe('exp/data.csv') #data
preprocessor = get_or_create_preprocessor(preprocessor_path='exp/preprocessor.joblib') #processor
X_transformed = preprocessor.fit_transform(df)
model = load_or_train_autoencoder(X_transformed) #model
embeddings = get_or_create_embeddings(X_transformed, model, embedding_path='exp/embeddings.npy') # embeddings


def recommend(jobma_catcher_id):
    import os
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import normalize

    df_data = get_or_create_dataframe(data_path='exp/data.csv')
    preprocessor = get_or_create_preprocessor(preprocessor_path='exp/preprocessor.joblib')
    X_transformed = preprocessor.fit_transform(df_data)
    model = load_or_train_autoencoder(X_transformed, model_path='exp/Client_autoencoder.pth')
    embeddings = get_or_create_embeddings(X_transformed, model, embedding_path='exp/embeddings.npy')

    client_df = df_data.copy()
    sample_df = client_df[client_df["jobma_catcher_id"] == jobma_catcher_id]
    if sample_df.empty:
        print(f"Client ID {jobma_catcher_id} not in dataset.")
        return pd.DataFrame()

    try:
        print(f"Processing sample for client ID {jobma_catcher_id}...")
        sample_processed = preprocessor.transform(sample_df)
        sample_tensor = torch.FloatTensor(sample_processed.toarray() if hasattr(sample_processed, 'toarray') else sample_processed)
        with torch.no_grad():
            sample_embedding, _ = model(sample_tensor)

        embeddings_tensor = torch.FloatTensor(embeddings)
        sample_embedding = sample_embedding[0]
        sample_norm = sample_embedding / torch.norm(sample_embedding)
        embeddings_norm = embeddings_tensor / embeddings_tensor.norm(dim=1, keepdim=True)
        similarities = torch.matmul(embeddings_norm, sample_norm).numpy()
        top_indices = np.argsort(similarities)[::-1]
        top_indices = [idx for idx in top_indices if client_df.iloc[idx]['jobma_catcher_id'] != jobma_catcher_id][:5]

        recommendations = []
        for idx in top_indices:
            client_data = client_df.iloc[idx].copy()
            client_data['similarity_score'] = round(float(similarities[idx]), 4)
            recommendations.append(client_data)

        result_df = pd.DataFrame(recommendations)
        print(result_df.to_string())
        return result_df

    except Exception as e:
        print(f"Error during recommendation process: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


import time
start_time = time.time()

recommend(jobma_catcher_id=4458)
print(df[df["jobma_catcher_id"]==4458].to_string())

recommend(jobma_catcher_id=9579)
print(df[df["jobma_catcher_id"]==9579].to_string())

end_time = time.time()
print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
