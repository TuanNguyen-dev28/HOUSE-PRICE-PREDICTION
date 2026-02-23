import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df['house_age'] = 2025 - df['year_built']
    df = df.drop(columns=['year_built'])
    return df

def split_features_target(df):
    X = df.drop(columns=['price'])
    y = df['price']
    return X, y