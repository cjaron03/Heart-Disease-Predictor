"""
Model Utilities for Heart Disease Prediction

This module contains helper functions for loading data, preprocessing,
training models, and evaluating performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Column names for the Cleveland Heart Disease dataset
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Features used for prediction in the web app (8 key features)
APP_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak']


def load_and_preprocess_data(filepath='processed.cleveland.data'):
    """
    Load the Cleveland Heart Disease dataset and preprocess it.

    Parameters:
        filepath (str): Path to the data file

    Returns:
        tuple: (X, y, df) - features, target, and full dataframe
    """
    # Load data with column names
    df = pd.read_csv(filepath, names=COLUMN_NAMES, na_values='?')

    # Handle missing values with mode imputation (ca and thal are categorical)
    for col in ['ca', 'thal']:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Binarize target: 0 = no disease, 1-4 -> 1 = disease
    df['target'] = (df['target'] > 0).astype(int)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y, df


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets with stratification.

    Parameters:
        X: Features
        y: Target
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_decision_tree(X_train, y_train, **kwargs):
    """
    Train a Decision Tree classifier.

    Parameters:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional hyperparameters for DecisionTreeClassifier

    Returns:
        DecisionTreeClassifier: Trained model
    """
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def tune_decision_tree(X_train, y_train, cv=5):
    """
    Tune Decision Tree hyperparameters using GridSearchCV.

    Parameters:
        X_train: Training features
        y_train: Training target
        cv (int): Number of cross-validation folds

    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    param_grid = {
        'max_depth': [3, 5, 7, 10, None]
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='f1',
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest classifier.

    Parameters:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional hyperparameters for RandomForestClassifier

    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, cv=5):
    """
    Tune Random Forest hyperparameters using GridSearchCV.

    Parameters:
        X_train: Training features
        y_train: Training target
        cv (int): Number of cross-validation folds

    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='f1',
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return performance metrics.

    Parameters:
        model: Trained classifier
        X_test: Test features
        y_test: Test target

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score
    """
    y_pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }


def get_feature_importance(model, feature_names):
    """
    Get feature importances from a tree-based model.

    Parameters:
        model: Trained tree-based classifier
        feature_names: List of feature names

    Returns:
        pd.DataFrame: DataFrame with features and their importances, sorted by importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df


def save_model(model, filepath='best_model.pkl'):
    """
    Save a trained model to disk.

    Parameters:
        model: Trained model to save
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)


def load_model(filepath='best_model.pkl'):
    """
    Load a trained model from disk.

    Parameters:
        filepath (str): Path to the saved model

    Returns:
        Trained model
    """
    return joblib.load(filepath)


def prepare_input_for_prediction(user_input, all_features=COLUMN_NAMES[:-1]):
    """
    Prepare user input for model prediction.

    The web app only collects 8 features. This function fills in default values
    for the remaining features to match the model's expected input.

    Parameters:
        user_input (dict): Dictionary with user-provided feature values
        all_features (list): List of all feature names expected by the model

    Returns:
        pd.DataFrame: DataFrame with one row containing all features
    """
    # Default values for features not collected from user
    # These are median/mode values from the dataset
    defaults = {
        'age': 55,
        'sex': 1,
        'cp': 3,
        'trestbps': 130,
        'chol': 246,
        'fbs': 0,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 0.8,
        'slope': 2,
        'ca': 0,
        'thal': 3
    }

    # Merge user input with defaults
    full_input = {**defaults, **user_input}

    # Create DataFrame with correct column order
    df = pd.DataFrame([full_input])[all_features]

    return df
