# train_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import numpy as np



def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    - z (numpy.ndarray): Input values.

    Returns:
    - numpy.ndarray: Output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def calculate_cost(X, y, theta, lambda_reg):
    """
    Calculate the logistic regression cost function with regularization.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Model parameters.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - float: Logistic regression cost with regularization.
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    regularization_term = (lambda_reg / (2 * m)) * np.sum(theta[1:]**2)
    cost = (-1 / m) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h))) + regularization_term
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations, lambda_reg):
    """
    Perform gradient descent for logistic regression with regularization.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Initial model parameters.
    - alpha (float): Learning rate.
    - num_iterations (int): Number of iterations for gradient descent.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - numpy.ndarray: Trained model parameters.
    - list: History of cost values during training.
    """
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * (X.T.dot(h - y))
        regularization_term = (lambda_reg / m) * np.concatenate(([0], theta[1:]))
        gradient += regularization_term
        theta -= alpha * gradient
        cost = calculate_cost(X, y, theta, lambda_reg)
        cost_history.append(cost)

    return theta, cost_history

def find_learning_rate(X, y, theta, num_iterations, lambda_reg, lr_range=(1e-5, 10), num_lr_steps=100):
    """
    Find the optimal learning rate using a learning rate range.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Initial model parameters.
    - num_iterations (int): Number of iterations for each learning rate test.
    - lambda_reg (float): Regularization parameter.
    - lr_range (tuple): Range of learning rates to test.
    - num_lr_steps (int): Number of learning rate steps to test.

    Returns:
    - float: Optimal learning rate.
    """
    losses = []
    learning_rates = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_lr_steps)

    for lr in learning_rates:
        _, cost_history = gradient_descent(X, y, theta, lr, num_iterations, lambda_reg)
        losses.append(cost_history[-1])

    # Plot the learning rate vs. loss
    plt.plot(learning_rates, losses, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

    # Find the learning rate at which the loss decreases most rapidly
    best_lr_index = np.argmin(losses)
    best_lr = learning_rates[best_lr_index]

    return best_lr

def train_model(X_train, y_train, alpha, num_iterations, lambda_reg):
    """
    Train a logistic regression model.

    Parameters:
    - X_train (numpy.ndarray): Training feature matrix.
    - y_train (numpy.ndarray): Training target variable.
    - alpha (float): Learning rate.
    - num_iterations (int): Number of iterations for gradient descent.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - numpy.ndarray: Trained model parameters.
    - list: History of cost values during training.
    """
    theta_initial = np.zeros(X_train.shape[1])
    theta_trained, cost_history = gradient_descent(X_train, y_train, theta_initial, alpha, num_iterations, lambda_reg)
    return theta_trained, cost_history

def train_sklearn_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model using scikit-learn with hyperparameter tuning.

    Parameters:
    - X_train (numpy.ndarray): Training feature matrix.
    - y_train (numpy.ndarray): Training target variable.

    Returns:
    - sklearn.linear_model.LogisticRegression: Trained logistic regression model.
    """
    # Initialize Logistic Regression model
    model = LogisticRegression()

    # Specify hyperparameters for GridSearchCV
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')

    # Fit the model with hyperparameter tuning
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def train_decision_tree(X_train, y_train):
    """
    Train a decision tree model using scikit-learn with hyperparameter tuning.

    Parameters:
    - X_train (numpy.ndarray): Training feature matrix.
    - y_train (numpy.ndarray): Training target variable.

    Returns:
    - sklearn.tree.DecisionTreeClassifier: Trained decision tree model.
    """
    # Initialize Decision Tree model
    model = DecisionTreeClassifier(random_state=0)

    # Specify hyperparameters for GridSearchCV
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None,1,2,3,4,5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')

    # Fit the model with hyperparameter tuning
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_