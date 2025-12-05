"""
Heart Disease Risk Advisor - Flask Web Application

A user-friendly web interface for predicting heart disease risk
using a trained machine learning model with SHAP explainability.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import pickle
import os
import shap

app = Flask(__name__)

# Load the trained model, feature names, and SHAP explainer at startup
MODEL_PATH = 'best_model.pkl'
FEATURES_PATH = 'feature_names.pkl'
SHAP_PATH = 'shap_explainer.pkl'

model = None
feature_names = None
shap_explainer = None

# Base features from user input (in order)
BASE_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Default values for features not collected from user (median/mode from dataset)
FEATURE_DEFAULTS = {
    'fbs': 0,
    'restecg': 0,
    'slope': 2
}

# Human-readable feature names for display
FEATURE_LABELS = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Blood Pressure',
    'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Angina',
    'oldpeak': 'ST Depression',
    'slope': 'ST Slope',
    'ca': 'Major Vessels',
    'thal': 'Thalassemia',
    'cardiac_efficiency': 'Cardiac Efficiency',
    'heart_rate_reserve': 'Heart Rate Reserve',
    'age_adjusted_hr': 'Age-Adjusted HR',
    'age_chol_risk': 'Age-Cholesterol Risk',
    'age_bp_risk': 'Age-BP Risk',
    'metabolic_risk': 'Metabolic Risk',
    'st_severity': 'ST Severity',
    'exercise_stress_index': 'Exercise Stress Index',
    'vessel_severity': 'Vessel Severity',
    'high_chol': 'High Cholesterol Flag',
    'high_bp': 'High BP Flag',
    'low_hr_reserve': 'Low HR Reserve',
    'asymptomatic_cp': 'Asymptomatic Pain'
}


def engineer_features(df):
    """
    Create advanced domain-specific features for heart disease prediction.
    Must match the feature engineering in the notebook exactly.
    """
    df = df.copy()

    # Cardiac Efficiency Metrics
    df['cardiac_efficiency'] = df['thalach'] / df['trestbps']
    df['heart_rate_reserve'] = (220 - df['age']) - df['thalach']
    df['age_adjusted_hr'] = df['thalach'] / (220 - df['age'])

    # Risk Interaction Features
    df['age_chol_risk'] = df['age'] * df['chol'] / 1000
    df['age_bp_risk'] = df['age'] * df['trestbps'] / 1000
    df['metabolic_risk'] = (df['chol'] / 200) + (df['trestbps'] / 120) + df['fbs']

    # ST Depression Analysis
    df['st_severity'] = df['oldpeak'] * (df['slope'] + 1)
    df['exercise_stress_index'] = df['oldpeak'] * df['exang'] + (df['oldpeak'] ** 2)

    # Vessel Disease Indicators
    df['vessel_severity'] = df['ca'] * df['thal'] / 3

    # Binary Risk Flags
    df['high_chol'] = (df['chol'] > 240).astype(int)
    df['high_bp'] = (df['trestbps'] > 140).astype(int)
    df['low_hr_reserve'] = (df['heart_rate_reserve'] > 50).astype(int)

    # Chest Pain Risk
    df['asymptomatic_cp'] = (df['cp'] == 4).astype(int)

    return df


def validate_input(data):
    """Validate user inputs are within reasonable ranges."""
    validations = {
        'age': (20, 100, "Age must be between 20 and 100"),
        'trestbps': (80, 220, "Blood pressure must be between 80 and 220"),
        'chol': (100, 600, "Cholesterol must be between 100 and 600"),
        'thalach': (60, 220, "Max heart rate must be between 60 and 220"),
        'oldpeak': (0, 7, "ST depression must be between 0 and 7"),
    }

    for field, (min_val, max_val, msg) in validations.items():
        try:
            val = float(data.get(field, 0))
            if not (min_val <= val <= max_val):
                return False, msg
        except (ValueError, TypeError):
            return False, f"Invalid value for {field}"

    return True, None


def load_model():
    """Load the trained model, feature names, and SHAP background data."""
    global model, feature_names, shap_explainer

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please run the Jupyter notebook first to train and save the model.")
        model = None

    if os.path.exists(FEATURES_PATH):
        feature_names = joblib.load(FEATURES_PATH)
        print(f"Feature names loaded: {len(feature_names)} features")
    else:
        feature_names = BASE_FEATURES

    if os.path.exists(SHAP_PATH):
        with open(SHAP_PATH, 'rb') as f:
            shap_data = pickle.load(f)
        # Store background data for creating explainer on demand
        if isinstance(shap_data, dict) and 'background' in shap_data:
            shap_explainer = shap_data['background']
            print("SHAP background data loaded successfully")
        else:
            # Old format - direct explainer
            shap_explainer = shap_data
            print("SHAP explainer loaded successfully")
    else:
        print("Warning: SHAP explainer not found. SHAP explanations will be unavailable.")
        shap_explainer = None


def get_shap_explanations(input_df):
    """Get SHAP explanations for a prediction using KernelExplainer."""
    if shap_explainer is None or model is None:
        return []

    try:
        # Create prediction function for SHAP
        def model_predict(X):
            return model.predict_proba(X)[:, 1]

        # Check if shap_explainer is background data (DataFrame) or actual explainer
        if isinstance(shap_explainer, pd.DataFrame):
            # Create KernelExplainer with background data
            explainer = shap.KernelExplainer(model_predict, shap_explainer)
            shap_values = explainer.shap_values(input_df, nsamples=100)
        else:
            # Try using as callable explainer
            shap_values = shap_explainer(input_df)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values[0]
            else:
                shap_values = shap_values[0]

        explanations = []
        for i, feat in enumerate(input_df.columns):
            if isinstance(shap_values, np.ndarray):
                shap_val = float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i])
            else:
                shap_val = float(shap_values[i])

            explanations.append({
                'feature': feat,
                'label': FEATURE_LABELS.get(feat, feat),
                'shap_value': shap_val,
                'direction': 'increases risk' if shap_val > 0 else 'decreases risk',
                'impact': abs(shap_val)
            })

        # Sort by absolute impact
        explanations.sort(key=lambda x: x['impact'], reverse=True)
        return explanations[:5]  # Return top 5
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_feature_importance_for_input(input_df):
    """Get feature importance from the model."""
    if model is None or not hasattr(model, 'feature_importances_'):
        return {}

    importances = model.feature_importances_
    feature_contributions = {}

    for i, feat in enumerate(input_df.columns):
        feature_contributions[feat] = float(importances[i])

    # Sort by importance
    sorted_contributions = dict(sorted(feature_contributions.items(),
                                       key=lambda x: x[1], reverse=True))
    return sorted_contributions


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with SHAP explanations."""
    if model is None:
        return jsonify({
            'error': True,
            'message': 'Model not loaded. Please run the Jupyter notebook first to train and save the model.'
        }), 500

    try:
        # Get form data
        data = request.get_json()

        # Validate inputs
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': True,
                'message': error_msg
            }), 400

        # Extract user inputs
        user_input = {
            'age': float(data.get('age', 50)),
            'sex': int(data.get('sex', 1)),
            'cp': int(data.get('cp', 3)),
            'trestbps': float(data.get('trestbps', 130)),
            'chol': float(data.get('chol', 250)),
            'thalach': float(data.get('thalach', 150)),
            'exang': int(data.get('exang', 0)),
            'oldpeak': float(data.get('oldpeak', 1.0)),
            'ca': int(data.get('ca', 0)),
            'thal': int(data.get('thal', 3))
        }

        # Merge with defaults for features not collected
        full_input = {**FEATURE_DEFAULTS, **user_input}

        # Create DataFrame with base features
        input_df = pd.DataFrame([full_input])[BASE_FEATURES]

        # Apply feature engineering
        input_df_eng = engineer_features(input_df)

        # Ensure columns match feature_names order
        if feature_names is not None and len(feature_names) > len(BASE_FEATURES):
            input_df_eng = input_df_eng[feature_names]

        # Make prediction
        prediction = model.predict(input_df_eng)[0]

        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df_eng)[0]
            probability = float(proba[1])  # Probability of disease

        # Get SHAP explanations
        shap_explanations = get_shap_explanations(input_df_eng)

        # Get feature importance (fallback if SHAP unavailable)
        feature_importance = get_feature_importance_for_input(input_df_eng)

        # Prepare response
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability,
            'feature_importance': feature_importance,
            'shap_explanations': shap_explanations,
            'message': get_result_message(prediction, probability)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error processing request: {str(e)}'
        }), 400


def get_result_message(prediction, probability):
    """Generate a human-readable result message."""
    if prediction == 1:
        msg = "Based on the provided information, the model indicates an ELEVATED RISK of heart disease."
        if probability:
            msg += f" (Confidence: {probability*100:.1f}%)"
    else:
        msg = "Based on the provided information, the model indicates a LOWER RISK of heart disease."
        if probability:
            msg += f" (Confidence: {(1-probability)*100:.1f}%)"

    return msg


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'shap_available': shap_explainer is not None,
        'features': len(feature_names) if feature_names else 0
    })


# Load model when app starts
load_model()


if __name__ == '__main__':
    app.run(debug=True, port=5001)
