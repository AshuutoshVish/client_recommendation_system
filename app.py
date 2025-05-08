import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


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
        subscription['subscription_amount'] = np.where(subscription['currency'] == 0, subscription['subscription_amount'].round(1), (subscription['subscription_amount'] / 85).round(1))
        subscription = subscription.groupby('jobma_catcher_id').agg(subscription_sum=('subscription_amount', 'sum'), subscription_count=('jobma_catcher_id', 'count')).reset_index()
        return subscription

    def clean_login_data():
        login = pd.read_sql("SELECT jobma_user_id AS jobma_catcher_id, jobma_last_login FROM jobma_login", engine)
        login['jobma_last_login'] = pd.to_datetime(login['jobma_last_login'], errors='coerce')
        today = pd.to_datetime('2024-06-10 11:24:30')
        login['since_last_login'] = (today - login['jobma_last_login']).dt.days
        login = login.groupby('jobma_catcher_id').agg(since_last_login=('since_last_login', 'min')).reset_index()
        since_login_filled = login['since_last_login'].fillna(float('inf'))
        bins = [0, 30, 90, 180, 365, float('inf')]
        labels = [0, 1, 2, 3, 4]
        login['since_last_login'] = pd.cut(since_login_filled, bins=bins, labels=labels, right=False)
        return login

    def clean_invitations_data():
        invitations = pd.read_sql("SELECT jobma_catcher_id, jobma_interview_mode,jobma_interview_status FROM jobma_pitcher_invitations", con=engine)
        recorded = invitations[invitations['jobma_interview_mode'].isin([1, '1'])].groupby('jobma_catcher_id').size().reset_index(name='recorded_interview_count')
        live = invitations[invitations['jobma_interview_mode'].isin([2, '2'])].groupby('jobma_catcher_id').size().reset_index(name='live_interview_count')
        invites = invitations[invitations['jobma_interview_status'].isin([0, '0'])].groupby('jobma_catcher_id').size().reset_index(name='invites_count')
        done = invitations[~invitations['jobma_interview_status'].isin([0, '0'])].groupby('jobma_catcher_id').size().reset_index(name='interview_done')
        summary = pd.merge(recorded, live, on='jobma_catcher_id', how='left')#may be need improvemnt
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
            work_df.loc[work_df['jobma_catcher_parent'] == 0, col] += work_df.loc[work_df['jobma_catcher_parent'] == 0, 'jobma_catcher_id'].map(child_sums[col]).fillna(0)

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

    # Merging everything
    catcher_df = clean_catcher_data()
    other_dfs = [clean_wallet_data(),clean_subscription_data(),clean_invitations_data(),clean_pre_recorded_kit_data(),clean_job_posting_data(),clean_login_data()]
    merged_df = catcher_df.copy()
    for df in other_dfs:
        merged_df = pd.merge(merged_df, df, how='left', on='jobma_catcher_id')

    final_df = working_on_merged_df(merged_df)
    return final_df

df = process_catcher_data(get_db_engine())
print(df)
print(df.info())