import os
import torch
import joblib
import pickle
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
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer


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
    with engine.connect() as conn:
        print("Database connection is made successfully..!\n")
    return engine

def process_catcher_data(engine):
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
    other_dfs = [clean_wallet_data(),
                 clean_subscription_data(),
                 clean_invitations_data(),
                 clean_pre_recorded_kit_data(),
                 clean_job_posting_data(),
                 clean_login_data()
                ]
    merged_df = catcher_df.copy()
    for df in other_dfs:
        merged_df = pd.merge(merged_df, df, how='left', on='jobma_catcher_id')

    final_df = working_on_merged_df(merged_df)
    return final_df

def get_or_create_processed_data(path='processed_data/processed_data.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"Loading existing file from : {path}")
        df = pd.read_csv(path)
    else:
        print(f"No existing file found...!!")
        print(f"Creating new data at: {path}")
        engine = get_db_engine()
        df = process_catcher_data(engine)
        df.to_csv(path, index=False)
        print(f"Database is created at {path}...!")
    return df
df = get_or_create_processed_data('processed_data/processed_data.csv')
print("Dataframe fetched successfully...!!")


def get_preprocessor():
    categorical = ['since_last_login']
    ordinal = ['company_size']
    binary = ['is_premium', 'is_unlimited', 'subscription_status']
    numeric = ['sub_user', 'subscription_count', 'live_interview_count', 'interview_done', 'pre_recorded_kit_counts', 'invites_count', 'jobs_posted']
    log_scaled = ['subscription_sum', 'wallet_amount']

    preprocessor = ColumnTransformer([('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical),
                                      ('ord', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('ordinal', OrdinalEncoder(categories=[["1-25", "26-100", "101-500", "500-1000", "More than 1000"]],handle_unknown='use_encoded_value', unknown_value=-1))]), ordinal),
                                      ('bin', SimpleImputer(strategy='mean'), binary),
                                      ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')),('scale', StandardScaler())]), numeric),
                                      ('log', Pipeline([('imputer', SimpleImputer(strategy='mean')),('log', FunctionTransformer(np.log1p, validate=False)),('scale', StandardScaler())]), log_scaled)])
    return preprocessor


def preprocess_data(df, preprocessor_path='processor/preprocessor.joblib', fit=True):
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        features = preprocessor.transform(df)
        print("Preprocessor loaded from disk.")
    elif fit:
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        preprocessor = get_preprocessor()
        features = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, preprocessor_path)
        print("Preprocessor trained and saved.")
    else:
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path} and fit=False.")
    return features, preprocessor


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
    print("Training started")
    epochs=100
    batch_size=32
    patience=10
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

def load_or_train_autoencoder(X, model_path='model/client_autoencoder.pth'):
    input_dim = X.shape[1]
    model = ClientAutoencoder(input_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Model loaded from disk.")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("No model found. Training new model...")
        model = train_autoencoder(X, model_path, input_dim)
    return model


def extract_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        tensor = to_tensor(X)
        embeddings, _ = model(tensor)
    return embeddings.cpu().numpy()


def recommend_sample(user_inputs):
    model=ClientAutoencoder()
    model.load_state_dict(torch.load('model/client_autoencoder.pth'))
    # input=()
    embeddings=embeddings
    def prepare_recommend(user_inputs, pipeline):
        user_df=pd.DataFrame([user_inputs])
        user_input_tensor = pipeline.transform(user_df)
        print(user_input_tensor)

        model.eval()
        with torch.no_grad():
            user_embedding, _ = model(user_input_tensor)
            user_embedding = F.normalise(user_embedding, p=2, dim=1)
            latent_embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            latent_embeddings_tensor = F.normalize(latent_embeddings_tensor, p=2, dim=1)
            similarities = F.cosine_similarity(user_embedding, latent_embeddings_tensor, dim=1)
            top_indices = similarities.topk(5).indices.cpu().numpy()
            recommended = df.iloc[top_indices].copy()
            recommended['similarity'] = similarities[top_indices].cpu().numpy()
        return prepare_recommend
    return recommend_sample(user_inputs, Pipeline)