# train_models.py
from src.preprocessing import load_dataset, feature_scaling, add_bias_term
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

def compute_cost(X, y, theta):
    """
    Compute the mean squared error cost for linear regression.

    Parameters:
    - X (np.ndarray): Input features (with bias term).
    - y (np.ndarray): Target values.
    - theta (np.ndarray): Coefficients for linear regression.

    Returns:
    - float: Mean squared error cost.
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    """
    Perform gradient descent optimization for linear regression.

    Parameters:
    - X (np.ndarray): Input features (with bias term).
    - y (np.ndarray): Target values.
    - theta (np.ndarray): Initial coefficients for linear regression.
    - alpha (float): Learning rate.
    - iterations (int): Number of iterations for optimization.

    Returns:
    - np.ndarray: Optimized coefficients for linear regression.
    - np.ndarray: History of cost values during optimization.
    """
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / m
        theta = theta - alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

def train_simple_linear_model(file_path, feature_column, target_column, alpha, iterations):
    """
    Train a simple linear regression model.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - feature_column (str): Name of the feature column.
    - target_column (str): Name of the target column.
    - alpha (float): Learning rate for gradient descent.
    - iterations (int): Number of iterations for gradient descent.

    Returns:
    - np.ndarray: Optimized coefficients for the linear regression model.
    - np.ndarray: History of cost values during optimization.
    - np.ndarray: Scaled and bias-added feature matrix.
    - np.ndarray: Target values.
    - np.ndarray: Feature matrix with added bias term.
    """
    dataset = load_dataset(file_path)
    X = feature_scaling(dataset[feature_column].values.reshape(-1, 1))
    y = dataset[target_column].values
    X_bias = add_bias_term(X)
    
    theta = np.zeros(X_bias.shape[1])
    theta_optimized, cost_history = gradient_descent(X_bias, y, theta, alpha, iterations)

    return theta_optimized, cost_history, X, y,X_bias

def train_multiple_linear_model(file_path, feature_columns, target_column, alpha, iterations):
    """
    Train a multiple linear regression model.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - feature_columns (list): List of feature column names.
    - target_column (str): Name of the target column.
    - alpha (float): Learning rate for gradient descent.
    - iterations (int): Number of iterations for gradient descent.

    Returns:
    - np.ndarray: Optimized coefficients for the linear regression model.
    - np.ndarray: History of cost values during optimization.
    - np.ndarray: Scaled and bias-added feature matrix.
    - np.ndarray: Target values.
    - np.ndarray: Feature matrix with added bias term.
    """
    dataset = load_dataset(file_path)
    X = feature_scaling(dataset[feature_columns].values)
    y = dataset[target_column].values
    X_bias = add_bias_term(X)

    theta = np.zeros(X_bias.shape[1])
    theta_optimized, cost_history = gradient_descent(X_bias, y, theta, alpha, iterations)

    return theta_optimized, cost_history, X, y,X_bias

def train_all_features_model(file_path, target_column, alpha, iterations):
    """
    Train a multiple linear regression model using all features in the dataset.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - target_column (str): Name of the target column.
    - alpha (float): Learning rate for gradient descent.
    - iterations (int): Number of iterations for gradient descent.

    Returns:
    - np.ndarray: Optimized coefficients for the linear regression model.
    - np.ndarray: History of cost values during optimization.
    - np.ndarray: Scaled and bias-added feature matrix.
    - np.ndarray: Target values.
    """
    dataset = load_dataset(file_path)

    # Identify non-numeric columns
    non_numeric_columns = dataset.select_dtypes(exclude=np.number).columns

    # Exclude non-numeric columns from X_all
    X_all_numeric = dataset.drop(columns=[target_column] + list(non_numeric_columns)).values

    # One-hot encode categorical columns
    X_all_encoded = pd.get_dummies(dataset[non_numeric_columns], drop_first=True)
    X_all_encoded = np.concatenate([X_all_numeric, X_all_encoded], axis=1)

    # Feature Scaling for numeric columns
    X_all_scaled = (X_all_encoded - X_all_encoded.mean(axis=0)) / X_all_encoded.std(axis=0)

    # Add a bias term to X
    X_all_bias = np.c_[np.ones((X_all_scaled.shape[0], 1)), X_all_scaled]

    y_all = dataset[target_column].values

    # Run gradient descent
    theta_all_optimized, cost_history_all = gradient_descent(X_all_bias, y_all, np.zeros(X_all_bias.shape[1]), alpha, iterations)

    return theta_all_optimized, cost_history_all, X_all_bias, y_all


def k_fold_cross_validation(file_path, target_column, alpha_values, iterations):
    """
    Perform k-fold cross-validation for multiple linear regression.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - target_column (str): Name of the target column.
    - alpha_values (list): List of learning rates for cross-validation.
    - iterations (int): Number of iterations for gradient descent.

    Prints:
    - Learning rate and average R2 score for each iteration of cross-validation.
    """
    dataset = load_dataset(file_path)

    # Identify non-numeric columns
    non_numeric_columns = dataset.select_dtypes(exclude=np.number).columns

    # Exclude non-numeric columns from X_all
    X_all_numeric = dataset.drop(columns=[target_column] + list(non_numeric_columns)).values

    # One-hot encode categorical columns
    X_all_encoded = pd.get_dummies(dataset[non_numeric_columns], drop_first=True)
    X_all_encoded = np.concatenate([X_all_numeric, X_all_encoded], axis=1)

    # Feature Scaling for numeric columns
    X_all_scaled = (X_all_encoded - X_all_encoded.mean(axis=0)) / X_all_encoded.std(axis=0)

    # Add a bias term to X
    X_all_bias = np.c_[np.ones((X_all_scaled.shape[0], 1)), X_all_scaled]

    y_all = dataset[target_column].values

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for alpha in alpha_values:
        r2_scores = []
        for train_index, val_index in kf.split(X_all_bias):
            X_train_fold, X_val_fold = X_all_bias[train_index], X_all_bias[val_index]
            y_train_fold, y_val_fold = y_all[train_index], y_all[val_index]

            # Initialize parameters
            theta_fold = np.zeros(X_train_fold.shape[1])

            # Run gradient descent on the training fold
            theta_optimized_fold, _ = gradient_descent(X_train_fold, y_train_fold, theta_fold, alpha, iterations)

            # Predictions for the validation fold
            y_pred_val_fold = X_val_fold.dot(theta_optimized_fold)

            # Calculate R2 score for the validation fold
            r2_val_fold = r2_score(y_val_fold, y_pred_val_fold)
            r2_scores.append(r2_val_fold)

        # Calculate average R2 score for the learning rate
        avg_r2_score = np.mean(r2_scores)
        print(f'Learning Rate: {alpha}, Average R2 Score: {avg_r2_score:.2f}')