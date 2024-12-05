from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    param_grid = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['saga'],
        'max_iter': [10000]
    }

    model = LogisticRegression(random_state=42)

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the test data
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    # Get classification report as a dictionary
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Add classification report and best hyperparameters to metrics
    metrics['classification_report'] = classification_report_dict
    metrics['best_hyperparameters'] = grid_search.best_params_

    # Convert metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame([metrics])

    return metrics_df