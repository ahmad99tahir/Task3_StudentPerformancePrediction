from src.preprocess import preprocess_data
from src.train_models import sigmoid,train_model, calculate_cost, gradient_descent, train_decision_tree, train_sklearn_logistic_regression, find_learning_rate
from src.evaluate import evaluate_model, plot_evaluation_metrics, k_fold_cross_validation, evaluate_model_sklearn, plot_evaluation_metrics_sklearn
import pandas as pd
import numpy as np
import joblib

def main():
    # Load and preprocess data
    data = pd.read_csv('dataset/student.csv')
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_rus, y_train_rus = preprocess_data(data)

    alpha = 0.1747
    num_iterations = 1000
    lambda_reg = 1

    #This code is to perform Logistic Regression with Gradient Descent
    # Train the logistic regression gradient descent model using SMOTE data
    theta_trained_smote, cost_history_smote = train_model(X_train_smote, y_train_smote, alpha, num_iterations, 1)
    # Evaluate the logistic regression gradient descent model
    evaluate_model(X_test, y_test, theta_trained_smote, lambda_reg)
    #Plot the evaluations
    plot_evaluation_metrics(X_test, y_test, theta_trained_smote)

    #This code is to perform K-Fold CV for Logistic Regression Gradient Descent and save the best model
    """
    #let's apply K-Fold cross-validation to select the best logistic regression gradient descent model with original data
    best_model_simple = k_fold_cross_validation(X_train, y_train, alpha, num_iterations, lambda_reg)
    joblib.dump(best_model_simple, 'models/logreg_gd_simple')
    # Evaluate the logistic regression gradient descent model
    evaluate_model(X_test, y_test, best_model_simple, lambda_reg)
    #Plot the evaluations
    plot_evaluation_metrics(X_test, y_test, best_model_simple)
    """

    #This code is to train, save and evaluate SKlearn's Logistic Regression
    """
    #Train and Evaluate using original Data
    logreg = train_sklearn_logistic_regression(X_train,y_train)
    joblib.dump(logreg,'models/logreg_simple')
    evaluate_model_sklearn(X_test,y_test,logreg)
    plot_evaluation_metrics_sklearn(X_test,y_test,logreg)
    """

    #This code is to train,save and evaluate Decision Trees
    """
    #Train using smote data
    dt = train_decision_tree(X_train_smote,y_train_smote)
    joblib.dump(dt,'models/tree_smote')
    evaluate_model_sklearn(X_test,y_test,dt)
    plot_evaluation_metrics_sklearn(X_test,y_test,dt)
    """

    #This code loads and evaluates a presaved gradient descent model.
    """
    model = joblib.load('models/logreg_gd_smote')
    # Evaluate the logistic regression gradient descent model
    evaluate_model(X_test, y_test, model, lambda_reg)
    #Plot the evaluations
    plot_evaluation_metrics(X_test, y_test, model)
    """

    #This code loads and evaluates a Logistic Regression or Decision Tree Model.
    """
    model = joblib.load('models/tree_simple')
    evaluate_model_sklearn(X_test,y_test,model)
    plot_evaluation_metrics_sklearn(X_test,y_test,model)
    """

if __name__ == "__main__":
    main()