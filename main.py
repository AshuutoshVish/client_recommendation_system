import os
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_db_engine():
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
    try:
        with engine.connect() as conn:
            print("Database connection is made successfully..!")
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")
    return engine


def clean_catcher_data(engine):
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


def clean_wallet_data(engine):
    wallet = pd.read_sql("SELECT catcher_id AS jobma_catcher_id, wallet_amount, is_unlimited FROM wallet", engine).copy()
    wallet['wallet_amount'] = pd.to_numeric(wallet['wallet_amount'], errors='coerce').fillna(0)
    wallet['is_unlimited'] = pd.to_numeric(wallet['is_unlimited'], errors='coerce').fillna(0).astype(int)
    return wallet


def clean_subscription_data(engine):
    subscription = pd.read_sql("SELECT catcher_id AS jobma_catcher_id, subscription_amount, currency FROM subscription_history", engine).copy()
    subscription['subscription_amount'] = pd.to_numeric(subscription['subscription_amount'], errors='coerce').fillna(0)
    subscription['subscription_amount'] = subscription['subscription_amount'].where(subscription['subscription_amount'] > 0, 0)
    subscription['subscription_amount'] = np.where(subscription['currency'] == 0, subscription['subscription_amount'].round(1), (subscription['subscription_amount'] / 85).round(1))
    subscription = subscription.groupby('jobma_catcher_id').agg(subscription_sum=('subscription_amount', 'sum'), subscription_count=('jobma_catcher_id', 'count')).reset_index()
    return subscription


def clean_login_data(engine):
    login = pd.read_sql("SELECT jobma_user_id AS jobma_catcher_id, jobma_last_login FROM jobma_login", engine)
    login['jobma_last_login'] = pd.to_datetime(login['jobma_last_login'], errors='coerce')
    today = pd.Timestamp.today().normalize()
    login['since_last_login'] = (today - login['jobma_last_login']).dt.days
    login = login.groupby('jobma_catcher_id').agg(since_last_login=('since_last_login', 'min')).reset_index()
    since_login_filled = login['since_last_login'].fillna(float('inf'))
    bins = [0, 30, 90, 180, 365, float('inf')]
    labels = [0, 1, 2, 3, 4]
    login['since_last_login'] = pd.cut(since_login_filled, bins=bins, labels=labels, right=False)
    return login


def clean_invitations_data(engine):
    invitations = pd.read_sql("SELECT jobma_catcher_id FROM jobma_pitcher_invitations", con=engine)
    invitations = invitations.groupby('jobma_catcher_id').size().reset_index(name='invitations')
    return invitations


def clean_pre_recorded_kit_data(engine):
    kit = pd.read_sql("SELECT catcher_id AS jobma_catcher_id FROM job_assessment_kit", engine)
    kit = kit.groupby('jobma_catcher_id').size().reset_index(name='pre_recorded_kit_counts')
    return kit


def clean_job_posting_data(engine):
    postings = pd.read_sql("SELECT jobma_catcher_id FROM jobma_employer_job_posting", engine).copy()
    postings = postings.groupby('jobma_catcher_id').size().reset_index(name='jobs_posted')
    return postings


def clean_pre_rec_interviews_data(engine):
    interviews = pd.read_sql("SELECT jobma_catcher_id FROM jobma_interviews", con=engine).copy()
    interviews = interviews.groupby('jobma_catcher_id').size().reset_index(name='pre_rec_interview_taken')
    return interviews


def merge_catcher_related_data(engine):
    catcher_df = clean_catcher_data(engine)
    dfs = [
        clean_wallet_data(engine),
        clean_subscription_data(engine),
        clean_invitations_data(engine),
        clean_pre_recorded_kit_data(engine),
        clean_job_posting_data(engine),
        clean_pre_rec_interviews_data(engine),
        clean_login_data(engine)
    ]
    merged_df = catcher_df.copy()
    for df in dfs:
        merged_df = pd.merge(merged_df, df, how='left', on='jobma_catcher_id')
    return merged_df


def working_on_merged_df(work_df):
    columns_to_sum = ['invitations', 'pre_recorded_kit_counts', 'jobs_posted', 'pre_rec_interview_taken']
    work_df[columns_to_sum] = work_df[columns_to_sum].fillna(0).astype(int)
    child_sums = work_df[work_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')[columns_to_sum].sum()
    for col in columns_to_sum:
        work_df.loc[work_df['jobma_catcher_parent'] == 0, col] += (
            work_df.loc[work_df['jobma_catcher_parent'] == 0, 'jobma_catcher_id']
            .map(child_sums[col]).fillna(0)
        )

    sub_ac = work_df[work_df['jobma_catcher_parent'] != 0]
    min_login = sub_ac.groupby('jobma_catcher_parent')['since_last_login'].min().reset_index().rename(columns={'since_last_login': 'min_login'})
    work_df = work_df.merge(min_login, how='left', on='jobma_catcher_parent')
    work_df['since_last_login'] = np.where(work_df['jobma_catcher_parent'] != 0, work_df['min_login'], work_df['since_last_login'])
    work_df.drop(columns=['min_login'], inplace=True)
    work_df = work_df[work_df['jobma_catcher_parent'] == 0].drop('jobma_catcher_parent', axis=1)
    work_df['wallet_amount'] = work_df['wallet_amount'].fillna(0)
    work_df['subscription_sum'] = work_df['subscription_sum'].fillna(0)
    work_df['subscription_count'] = work_df['subscription_count'].fillna(0).astype(int)
    work_df['is_unlimited'] = work_df['is_unlimited'].replace('', 1).fillna(0).astype(int)
    work_df['since_last_login'] = work_df['since_last_login'].fillna(4).astype(int)
    return work_df

engine = get_db_engine()
merged = merge_catcher_related_data(engine)
final_df = working_on_merged_df(merged)
df = final_df.drop('jobma_catcher_id', axis=1)

def get_preprocessor():
    categorical = ['since_last_login']
    ordinal = ['company_size']
    binary = ['is_premium', 'is_unlimited', 'subscription_status']
    numeric = ['sub_user', 'subscription_count', 'invitations', 'pre_recorded_kit_counts', 'jobs_posted', 'pre_rec_interview_taken']
    log_scaled = ['subscription_sum', 'wallet_amount']

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical),
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=[['1-25', '26-100', '101-500', '500-1000', 'More than 1000']],
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

def preprocess_data(df, preprocessor=None, fit=True):
    if fit or preprocessor is None:
        preprocessor = get_preprocessor()
        features = preprocessor.fit_transform(df)
    else:
        features = preprocessor.transform(df)
    return features, preprocessor

    
#model
class ClientAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
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

def train_autoencoder(X_train, model_path, input_dim, epochs=50, batch_size=32):
    X_tensor = to_tensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClientAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), model_path)
    return model


# Loading model
def load_or_train_autoencoder(X, model_path='client_autoencoder.pth'):
    input_dim = X.shape[1]
    model = ClientAutoencoder(input_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Model loaded from disk.")
    else:
        print("No model found. Training new model...")
        model = train_autoencoder(X, model_path, input_dim)
    return model

# Embedding Extraction
def extract_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        tensor = to_tensor(X)
        embeddings, _ = model(tensor)
    return embeddings.cpu().numpy()


def generate_recommendations(embeddings, client_ids, top_k=5, min_similarity=0.5):
    similarity_matrix = cosine_similarity(embeddings)
    recommendations = {}
    for idx, client_id in enumerate(client_ids):
        sim_scores = similarity_matrix[idx]
        top_indices = np.argsort(sim_scores)[::-1]
        similar_clients = []
        for i in top_indices:
            if client_ids[i] == client_id:
                continue
            if sim_scores[i] >= min_similarity:
                similar_clients.append((client_ids[i], round(sim_scores[i], 4)))
            if len(similar_clients) >= top_k:
                break
        recommendations[client_id] = similar_clients
    return recommendations

# --- Entry Point ---
def get_similar_clients(input_client_id, df, top_k=5, min_similarity=0.2):
    client_ids = df['jobma_catcher_id'].values
    df_cleaned = df.drop('jobma_catcher_id', axis=1)

    features, preprocessor = preprocess_data(df_cleaned, fit=True)
    model = load_or_train_autoencoder(features)
    embeddings = extract_embeddings(model, features)
    recs = generate_recommendations(embeddings, client_ids, top_k, min_similarity)

    if input_client_id not in recs:
        print(f"Client ID {input_client_id} not in dataset.")
        return pd.DataFrame()

    similar_ids = [cid for cid, _ in recs[input_client_id]]
    scores = [score for _, score in recs[input_client_id]]
    result = df[df['jobma_catcher_id'].isin(similar_ids)].copy()
    result['similarity_score'] = scores
    return result.reset_index(drop=True)

if __name__ == "__main__":
    df = final_df
    input_id = 10521
    recommendations_df = get_similar_clients(input_id, df)
print(recommendations_df)


input_client_id = 10240
similar_clients_df = get_similar_clients(input_client_id, df)
print(similar_clients_df)

input_client_id = 6422
similar_clients_df = get_similar_clients(input_client_id, df)
print(similar_clients_df)