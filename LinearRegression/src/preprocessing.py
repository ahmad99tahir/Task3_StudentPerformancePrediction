# preprocessing.py
import pandas as pd
import numpy as np

def load_dataset(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def feature_scaling(X):
    """
    Perform feature scaling on the input data using mean normalization.

    Parameters:
    - X (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: Scaled data.
    """
    return (X - X.mean()) / X.std()

def add_bias_term(X):
    """
    Add a bias term (intercept) to the input data.

    Parameters:
    - X (pd.DataFrame): Input data.

    Returns:
    - np.ndarray: Data with an added bias term.
    """
    return np.c_[np.ones((X.shape[0], 1)), X]

