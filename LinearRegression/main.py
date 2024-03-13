from src.preprocessing import load_dataset, add_bias_term, feature_scaling
from src.train_models import compute_cost, gradient_descent, train_simple_linear_model, train_multiple_linear_model, train_all_features_model, k_fold_cross_validation
from src.evaluate import visualize_model,visualize_cost, evaluate_model
from sklearn.model_selection import train_test_split

def main():
    filepath = 'dataset/student.csv'
    df = load_dataset(filepath)
    df.head()

    #This code runs the simple linear regression model using a single feature
    """
    target_column = 'G3'
    alpha = 0.01
    iterations = 1000
    # Train simple linear regression model
    theta_optimized, cost_history, X, y, X_bias = train_simple_linear_model(filepath, 'studytime', target_column, alpha, iterations)
    # Visualize the model
    visualize_model(X, X_bias, y, theta_optimized, 'Simple Linear Regression', 'red')
    # Visualize the convergence of the cost function
    visualize_cost(iterations, cost_history)
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)
    # Evaluate the model on the test set
    evaluate_model(X_test, y_test, theta_optimized)
    """

    #This code runs the multiple linear regression model using multiple features
    """
    target_column = 'G3'
    feature_columns = ['studytime', 'absences', 'Medu', 'Fedu', 'age', 'health', 'goout', 'traveltime', 'failures']
    alpha = 0.01
    iterations = 1000
    # Train multiple linear regression model
    theta_multi_optimized, cost_history_multi, X_multi, y_multi, X_bias_multi = train_multiple_linear_model(filepath, feature_columns, target_column, alpha, iterations)
    # Visualize the model
    visualize_cost(iterations, cost_history_multi)
    # Evaluate the multiple linear regression model on the test set
    evaluate_model(X_bias_multi, y_multi, theta_multi_optimized)
    """

    #This code runs the multiple linear regression model using all the features
    target_column = 'G3'
    alpha = 0.01
    iterations = 1000
    theta_all, cost_history_all, X_all, y_all = train_all_features_model(filepath, target_column, alpha, iterations)
    visualize_cost(iterations,cost_history_all)
    evaluate_model(X_all,y_all,theta_all)
    

    # K-fold Cross-Validation
    """
    target_column = 'G3'
    iterations = 1000
    alpha_values = [0.001, 0.01, 0.025, 0.05 ,0.1, 0.25, 0.5]
    my = k_fold_cross_validation(filepath, target_column, alpha_values, iterations)
    """

if __name__ == "__main__":
    main()