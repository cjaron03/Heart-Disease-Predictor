"""
Heart Disease Risk Advisor - Flask Web Application

A user-friendly web interface for predicting heart disease risk
using a trained machine learning model.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and feature names at startup
MODEL_PATH = 'best_model.pkl'
FEATURES_PATH = 'feature_names.pkl'

model = None
feature_names = None

# All features expected by the model (in order)
ALL_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Default values for features not collected from user (median/mode from dataset)
FEATURE_DEFAULTS = {
    'fbs': 0,
    'restecg': 0,
    'slope': 2,
    'ca': 0,
    'thal': 3
}


def load_model():
    """Load the trained model and feature names."""
    global model, feature_names

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please run the Jupyter notebook first to train and save the model.")
        model = None

    if os.path.exists(FEATURES_PATH):
        feature_names = joblib.load(FEATURES_PATH)
    else:
        feature_names = ALL_FEATURES


def get_feature_importance_for_input(input_data):
    """
    Get feature importance contribution for a specific input.
    Returns the feature importances weighted by the input values (normalized).
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        return {}

    importances = model.feature_importances_
    feature_contributions = {}

    for i, (feat, val) in enumerate(zip(ALL_FEATURES, input_data)):
        # Normalize contribution by importance
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
    """Handle prediction requests."""
    if model is None:
        return jsonify({
            'error': True,
            'message': 'Model not loaded. Please run the Jupyter notebook first to train and save the model.'
        }), 500

    try:
        # Get form data
        data = request.get_json()

        # Extract user inputs
        user_input = {
            'age': float(data.get('age', 50)),
            'sex': int(data.get('sex', 1)),
            'cp': int(data.get('cp', 3)),
            'trestbps': float(data.get('trestbps', 130)),
            'chol': float(data.get('chol', 250)),
            'thalach': float(data.get('thalach', 150)),
            'exang': int(data.get('exang', 0)),
            'oldpeak': float(data.get('oldpeak', 1.0))
        }

        # Merge with defaults for features not collected
        full_input = {**FEATURE_DEFAULTS, **user_input}

        # Create input array in correct order
        input_array = np.array([[full_input[feat] for feat in ALL_FEATURES]])

        # Make prediction
        prediction = model.predict(input_array)[0]

        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_array)[0]
            probability = float(proba[1])  # Probability of disease

        # Get feature importance
        feature_importance = get_feature_importance_for_input(input_array[0])

        # Prepare response
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability,
            'feature_importance': feature_importance,
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
        'model_loaded': model is not None
    })


# Load model when app starts
load_model()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
