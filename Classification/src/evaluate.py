import numpy as np
from src.train_models import sigmoid,train_model
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def evaluate_model(X_test, y_test, best_model, lambda_reg):
    """
    Evaluate the logistic regression model on the test set and print various metrics.

    Parameters:
    - X_test (numpy.ndarray): Testing feature matrix.
    - y_test (numpy.ndarray): Testing target variable.
    - best_model (numpy.ndarray): Trained logistic regression model parameters.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - float: Accuracy of the model on the test set.
    """
    # Predict on the test set
    y_pred_prob = sigmoid(X_test.dot(best_model))
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return accuracy


# Function to plot ROC curve, precision-recall curve, and confusion matrix
def plot_evaluation_metrics(X, y, theta):
    """
    Plot ROC curve, precision-recall curve, and confusion matrix for a logistic regression model.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Model parameters.

    Returns:
    - None
    """
    # Predict probabilities on the validation set
    y_prob = sigmoid(X.dot(theta))
    y_pred = np.where(y_prob >= 0.5, 1, 0)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_prob)

    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    plt.subplot(1, 3, 3)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Fail', 'Pass'])
    plt.yticks([0, 1], ['Fail', 'Pass'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    plt.tight_layout()
    plt.show()

def k_fold_cross_validation(X_train, y_train, alpha, num_iterations, lambda_reg):
    """
    Perform k-fold cross-validation to find the best logistic regression model.

    Parameters:
    - X_train (numpy.ndarray): Training feature matrix.
    - y_train (pd.Series): Training target variable.
    - alpha (float): Learning rate.
    - num_iterations (int): Number of iterations for gradient descent.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - numpy.ndarray: Best-trained logistic regression model parameters.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

    best_model = None

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val = X_train[train_index], X_train[val_index]
        y_train_fold, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train the model
        theta_initial = np.zeros(X_train_fold.shape[1])
        theta_trained, _ = train_model(X_train_fold, y_train_fold, alpha, num_iterations, lambda_reg)

        # Save the model if it performs better than the previous best model
        if best_model is None or evaluate_model(X_val, y_val, theta_trained, lambda_reg) > evaluate_model(X_val, y_val, best_model, lambda_reg):
            best_model = theta_trained

    return best_model

def evaluate_model_sklearn(X_test, y_test, model):
    """
    Evaluate a logistic regression model using scikit-learn's Logistic Regression model.

    Parameters:
    - X_test (numpy.ndarray): Testing feature matrix.
    - y_test (pd.Series): Testing target variable.
    - model (sklearn.linear_model.LogisticRegression): Trained scikit-learn Logistic Regression model.

    Returns:
    - float: Accuracy of the model on the test set.
    """
    # Predict on the test set using sklearn's Logistic Regression model
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'ROC_AUC: {roc_auc:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return accuracy

def plot_evaluation_metrics_sklearn(X, y, model):
    """
    Plot ROC curve, precision-recall curve, and confusion matrix for a model using scikit-learn.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (pd.Series): Target variable.
    - model : Trained scikit-learn model.

    Returns:
    - None
    """
    # Predict probabilities on the validation set
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_prob)

    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    plt.subplot(1, 3, 3)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Fail', 'Pass'])
    plt.yticks([0, 1], ['Fail', 'Pass'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    plt.tight_layout()
    plt.show()