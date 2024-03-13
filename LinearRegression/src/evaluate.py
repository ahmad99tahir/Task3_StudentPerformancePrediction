# evaluate.py
from src.train_models import train_simple_linear_model, train_multiple_linear_model,train_all_features_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt

def visualize_model(X,X_bias, y, theta_optimized, title, color):
    """
    Visualize the linear regression model predictions alongside the actual data.

    Parameters:
    - X (np.ndarray): Scaled feature matrix.
    - X_bias (np.ndarray): Feature matrix with added bias term.
    - y (np.ndarray): Target values.
    - theta_optimized (np.ndarray): Optimized coefficients for the linear regression model.
    - title (str): Title for the plot.
    - color (str): Color for the regression line.

    Displays a scatter plot of the actual data and the linear regression model predictions.
    """
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, X_bias.dot(theta_optimized), color=color, label='Linear Regression')
    plt.xlabel('Feature (Scaled)')
    plt.ylabel('Target Variable')
    plt.title(title)
    plt.legend()
    plt.show()

def visualize_cost(iterations, cost_history):
    """
    Visualize the convergence of the cost function during gradient descent.

    Parameters:
    - iterations (int): Number of iterations during optimization.
    - cost_history (np.ndarray): History of cost values.

    Displays a plot of the cost function against the number of iterations.
    """
    # Visualize the cost function
    plt.plot(range(1, iterations+1), cost_history, color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Cost Function')
    plt.show()

def evaluate_model(X_test, y_test, theta_optimized):
    """
    Evaluate the performance of the linear regression model on a test set.

    Parameters:
    - X_test (np.ndarray): Scaled feature matrix for the test set.
    - y_test (np.ndarray): Target values for the test set.
    - theta_optimized (np.ndarray): Optimized coefficients for the linear regression model.

    Prints:
    - R2 Score and Mean Absolute Error for the model's predictions on the test set.
    """
    y_pred = X_test.dot(theta_optimized)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'R2 Score: {r2:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
