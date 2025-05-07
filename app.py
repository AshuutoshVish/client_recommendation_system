import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Working on env
load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

if not all([db_user, db_password, db_host, db_port, db_name]):
    raise ValueError("Missing one or more required environment variables.")

db_url = URL.create(drivername="mysql+pymysql", username=db_user, password=db_password, host=db_host, port=int(db_port), database=db_name)
engine = create_engine(db_url)

try:
    with engine.connect() as conn:
        print("Database connection is made successfully..!")
except Exception as e:
    print("Database connection failed:", e)


def clean_catcher_data(engine):
    catcher = pd.read_sql("SELECT jobma_catcher_id, is_premium, subscription_status, company_size, jobma_catcher_parent FROM jobma_catcher WHERE jobma_verified IN (1, '1') ", engine).copy()
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


def clean_subscription_data(engine):#new
    subscription = pd.read_sql("SELECT catcher_id AS jobma_catcher_id, subscription_amount, currency FROM subscription_history", engine).copy()
    subscription['subscription_amount'] = pd.to_numeric(subscription['subscription_amount'], errors='coerce').fillna(0)
    subscription['subscription_amount'] = subscription['subscription_amount'].where(subscription['subscription_amount'] > 0, 0)
    subscription['subscription_amount'] = np.where(subscription['currency'] == 0, subscription['subscription_amount'].round(1), (subscription['subscription_amount'] / 85).round(1))
    subscription = subscription.groupby('jobma_catcher_id').agg(subscription_sum=('subscription_amount', 'sum'),subscription_count=('jobma_catcher_id', 'count')).reset_index()
    return subscription


def clean_login_data(engine):
    login = pd.read_sql("SELECT jobma_user_id AS jobma_catcher_id, jobma_last_login FROM jobma_login", engine)
    login['jobma_last_login'] = pd.to_datetime(login['jobma_last_login'], errors='coerce')
    today = pd.Timestamp.today().normalize()
    login['since_last_login'] = (today - login['jobma_last_login']).dt.days
    login = (login.groupby('jobma_catcher_id').agg(since_last_login=('since_last_login', 'min')).reset_index())
    since_login_filled = login['since_last_login'].fillna(float('inf'))
    bins = [0, 30, 90, 180, 365, float('inf')]
    labels = [0, 1, 2, 3, 4]
    login['since_last_login'] = pd.cut(since_login_filled, bins=bins, labels=labels, right=False)
    return login


def clean_invitations_data(engine):
    invitations= pd.read_sql("SELECT jobma_catcher_id FROM jobma_pitcher_invitations", con= engine)
    invitations= invitations.groupby('jobma_catcher_id').size().reset_index(name='invitations')
    return invitations


def clean_pre_recorded_kit_data(engine):
    pre_recorded_kit = pd.read_sql("SELECT catcher_id AS jobma_catcher_id  FROM job_assessment_kit", engine)
    pre_recorded_kit = pre_recorded_kit.groupby('jobma_catcher_id').size().reset_index(name='pre_recorded_kit_counts')
    return pre_recorded_kit


def clean_job_posting_data(engine):
    job_posted = pd.read_sql("SELECT jobma_catcher_id FROM jobma_employer_job_posting", engine).copy()
    job_posted = job_posted.groupby('jobma_catcher_id').size().reset_index(name='jobs_posted')
    return job_posted


def clean_pre_rec_interviews_data(engine):
    pre_rec_interviews = pd.read_sql("SELECT jobma_catcher_id FROM jobma_interviews", con = engine).copy()
    pre_rec_interviews = pre_rec_interviews.groupby('jobma_catcher_id').size().reset_index(name='pre_rec_interview_taken')
    return pre_rec_interviews

def merge_catcher_related_data(engine):
    catcher_df = clean_catcher_data(engine)
    wallet_df = clean_wallet_data(engine)
    subscription_df = clean_subscription_data(engine)
    login_df = clean_login_data(engine)
    invitations_df = clean_invitations_data(engine)
    pre_recorded_kit_df = clean_pre_recorded_kit_data(engine)
    job_posted_df = clean_job_posting_data(engine)
    pre_rec_interviews_df = clean_pre_rec_interviews_data(engine)

    merged_df = catcher_df.copy()
    dfs_to_merge = [wallet_df, subscription_df, invitations_df, pre_recorded_kit_df, job_posted_df, pre_rec_interviews_df, login_df]

    for i in dfs_to_merge:
        merged_df = pd.merge(merged_df, i, how='left', on='jobma_catcher_id')
    return merged_df

work_df = merge_catcher_related_data(engine)


def working_on_merged_df(work_df):
    #Adding the value of child to parent
    columns_to_sum = ['invitations', 'pre_recorded_kit_counts', 'jobs_posted', 'pre_rec_interview_taken']
    work_df[columns_to_sum] = work_df[columns_to_sum].fillna(0).astype(int)

    child_sums = (work_df[work_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')[columns_to_sum].sum())
    for col in columns_to_sum:
        work_df.loc[work_df['jobma_catcher_parent'] == 0, col] += (work_df.loc[work_df['jobma_catcher_parent'] == 0,'jobma_catcher_id'].map(child_sums[col]).fillna(0))


    sub_ac = work_df[work_df['jobma_catcher_parent'] != 0]
    min_login_per_parent = (sub_ac.groupby('jobma_catcher_parent')['since_last_login'].min().reset_index().rename(columns={'since_last_login': 'min_login'}))
    work_df = work_df.merge(min_login_per_parent,how='left',left_on='jobma_catcher_parent',right_on='jobma_catcher_parent')
    work_df['since_last_login'] = np.where(work_df['jobma_catcher_parent'] != 0,work_df['min_login'],work_df['since_last_login'])
    work_df.drop(columns=['min_login'], inplace=True)

    #Taking only the parent accounts
    work_df=work_df[work_df['jobma_catcher_parent'] == 0].drop('jobma_catcher_parent', axis=1)

    #Filling the null values
    work_df['wallet_amount'] = work_df['wallet_amount'].fillna(0)
    work_df['subscription_sum'] = work_df['subscription_sum'].fillna(0)
    work_df['subscription_count'] = work_df['subscription_count'].fillna(0).astype(int)
    work_df['is_unlimited'] = work_df['is_unlimited'].replace('', 1).fillna(0).astype(int)
    work_df['since_last_login'] = work_df['since_last_login'].fillna(4).astype(int)
    return work_df

final_df = working_on_merged_df(work_df)
df = work_df.drop('jobma_catcher_id', axis=1)
print(df.info())



binary_columns = ['is_premium', 'subscription_status', 'is_unlimited']
ordinal_columns = ['company_size', 'since_last_login']
normal_columns = ['wallet_amount', 'subscription_count', 'invitations', 'pre_recorded_kit_counts', 'jobs_posted', 'pre_rec_interview_taken']
log_columns = ['wallet_amount', 'subscription_sum']


# def log_transform(X):
#     X = X.copy()
#     X[X <= 0] = 1e-10
#     return np.log(X)

# def clean_binary(X):
#     return X.apply(lambda col: col.map(lambda val: 1 if pd.to_numeric(val, errors='coerce') > 0 else 0).astype(int))

# def clean_categorical(X):
#     return X.apply(lambda col: col.map(lambda val: str(val).strip().lower() if pd.notnull(val) else 'others'))

# def clean_numerical(X):
#     return X.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# def clean_ordinal(X):
#     return X.apply(lambda col: pd.to_numeric(col, errors='coerce')).clip(lower=0)

# # Binary pipeline
# binary_pipeline = Pipeline(steps=[('cleaner', FunctionTransformer(clean_binary, validate=False)),('imputer', SimpleImputer(strategy='most_frequent')),])

# categorical_pipeline = Pipeline(steps=[('cleaner', FunctionTransformer(clean_categorical, validate=False)),('imputer', SimpleImputer(strategy='most_frequent')),
#                                        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
#                                       ])

# ordinal_pipeline = Pipeline(steps=[('cleaner', FunctionTransformer(clean_ordinal, validate=False)),
#                                    ('imputer', SimpleImputer(strategy='median')),
#                                    ('log_transform', FunctionTransformer(lambda x: np.log1p(x), validate=False)),
#                                   ])

# numerical_pipeline = Pipeline(steps=[('cleaner', FunctionTransformer(clean_numerical, validate=False)),
#                                      ('imputer', SimpleImputer(strategy='mean')),
#                                     ])

# preprocessor = ColumnTransformer(transformers=[('bin', binary_pipeline, binary_columns),
#                                                ('cat', categorical_pipeline, categorical_columns), 
#                                                ('ord', ordinal_pipeline, ordinal_columns),
#                                                ('num', numerical_pipeline, numerical_columns),
#                                               ])

# # Final pipeline
# pipeline = Pipeline([('preprocessing', preprocessor),
#                     ('scaler', StandardScaler())])