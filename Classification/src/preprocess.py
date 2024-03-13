# preprocess.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def preprocess_data(dataset):
    """
    Preprocesses the input dataset for machine learning.

    Parameters:
    - dataset (pd.DataFrame): The input dataset containing features and target variable.

    Returns:
    - X_train (numpy.ndarray): Training features.
    - X_test (numpy.ndarray): Testing features.
    - y_train (pd.Series): Training target variable.
    - y_test (pd.Series): Testing target variable.
    - X_train_smote (numpy.ndarray): Training features after applying SMOTE (Synthetic Minority Over-sampling Technique).
    - y_train_smote (pd.Series): Training target variable after applying SMOTE.
    - X_train_rus (numpy.ndarray): Training features after applying Random Undersampling.
    - y_train_rus (pd.Series): Training target variable after applying Random Undersampling.
    """
    
    dataset = dataset

    # Create a binary target variable (pass/fail)
    #dataset['Pass/Fail'] = np.where(dataset['G3'] >= 10, 1, 0)

    grades_pass_fail = []
    for index, row in dataset.iterrows():
        if row['G3'] >= 10:
            grades_pass_fail.append(1) #pass
        else:
            grades_pass_fail.append(0) #fail
            
    grades_pass_fail_series = pd.Series(grades_pass_fail)
    dataset["Pass/Fail"] = grades_pass_fail_series

    # Split the data into features (X) and target variable (y)
    X = dataset.iloc[:, :-3]  # Excluding 'G3' and 'Pass/Fail' columns
    y = dataset['Pass/Fail']

    # One-hot encode categorical variables
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22])],
                           remainder='passthrough')
    X_encoded = np.array(ct.fit_transform(X))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    # Apply random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


    return X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_rus, y_train_rus

