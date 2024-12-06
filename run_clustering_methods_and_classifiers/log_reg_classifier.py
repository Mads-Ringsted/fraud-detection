python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    param_dist = {
        'C': np.logspace(-2, 2, 20),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    model = LogisticRegression(random_state=42, max_iter=10000)

    # Use RandomizedSearchCV for faster hyperparameter tuning
    rand_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_log_loss',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    rand_search.fit(X_train, y_train)

    # Get the best model
    best_model = rand_search.best_estimator_

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
    metrics['best_hyperparameters'] = rand_search.best_params_

    # Convert metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame([metrics])

    return metrics_df