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
shap_explainer_cache = None
TORNADO_FEATURE_ORDER = [
    'age',
    'age_adjusted_hr',
    'age_bp_risk',
    'age_chol_risk',
    'metabolic_risk',
    'st_severity',
    'exercise_stress_index',
    'vessel_severity',
    'high_bp',
    'high_chol',
    'asymptomatic_cp',
    'exang',
    'chol',
    'trestbps',
    'thalach'
]

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

# Detailed feature explanations for user-friendly display
FEATURE_EXPLANATIONS = {
    'age': {
        'description': 'Your age affects baseline heart disease risk',
        'detail': 'Heart disease risk naturally increases with age as arteries stiffen over time.',
        'category': 'demographic',
        'is_modifiable': False
    },
    'sex': {
        'description': 'Biological sex influences heart disease patterns',
        'detail': 'Men typically face higher risk at younger ages; risk equalizes after menopause.',
        'category': 'demographic',
        'is_modifiable': False
    },
    'cp': {
        'description': 'Your chest pain type indicates potential heart involvement',
        'detail': 'Different types of chest pain suggest different levels of cardiac concern.',
        'category': 'symptoms',
        'is_modifiable': False
    },
    'trestbps': {
        'description': 'Your resting blood pressure level',
        'detail': 'High blood pressure forces your heart to work harder, increasing strain.',
        'category': 'cardiovascular',
        'is_modifiable': True
    },
    'chol': {
        'description': 'Your total cholesterol level',
        'detail': 'High cholesterol can lead to plaque buildup in arteries, restricting blood flow.',
        'category': 'metabolic',
        'is_modifiable': True
    },
    'thalach': {
        'description': 'Your maximum heart rate during exercise',
        'detail': 'A lower max heart rate may indicate reduced cardiovascular fitness.',
        'category': 'cardiovascular',
        'is_modifiable': True
    },
    'exang': {
        'description': 'Whether you experience chest pain during exercise',
        'detail': 'Exercise-induced chest pain may indicate insufficient blood flow to the heart.',
        'category': 'symptoms',
        'is_modifiable': False
    },
    'oldpeak': {
        'description': 'ST depression observed on ECG during exercise',
        'detail': 'Higher values suggest the heart may not be getting enough oxygen during stress.',
        'category': 'test_results',
        'is_modifiable': False
    },
    'ca': {
        'description': 'Number of major vessels with blockage',
        'detail': 'More blocked vessels indicate more severe coronary artery disease.',
        'category': 'test_results',
        'is_modifiable': False
    },
    'thal': {
        'description': 'Results from thallium stress test',
        'detail': 'Shows how well blood flows to your heart during rest and exercise.',
        'category': 'test_results',
        'is_modifiable': False
    },
    'age_adjusted_hr': {
        'description': 'Your heart rate relative to your age-predicted maximum',
        'detail': 'Shows how close you got to your expected maximum heart rate during exercise.',
        'category': 'cardiovascular',
        'is_modifiable': True
    },
    'age_bp_risk': {
        'description': 'Combined effect of age and blood pressure',
        'detail': 'Older age combined with higher blood pressure compounds risk.',
        'category': 'cardiovascular',
        'is_modifiable': True
    },
    'age_chol_risk': {
        'description': 'Combined effect of age and cholesterol',
        'detail': 'Higher cholesterol becomes more dangerous as we age.',
        'category': 'metabolic',
        'is_modifiable': True
    },
    'metabolic_risk': {
        'description': 'Overall metabolic health indicator',
        'detail': 'Combines cholesterol, blood pressure, and blood sugar for overall metabolic assessment.',
        'category': 'metabolic',
        'is_modifiable': True
    },
    'st_severity': {
        'description': 'Severity of ECG abnormalities',
        'detail': 'Combines ST depression amount with slope pattern for a complete picture.',
        'category': 'test_results',
        'is_modifiable': False
    },
    'exercise_stress_index': {
        'description': 'How your heart responds to exercise stress',
        'detail': 'Higher values indicate more concerning responses during physical exertion.',
        'category': 'cardiovascular',
        'is_modifiable': True
    },
    'vessel_severity': {
        'description': 'Overall severity of blood vessel involvement',
        'detail': 'Combines vessel blockage count with stress test results.',
        'category': 'test_results',
        'is_modifiable': False
    },
    'high_bp': {
        'description': 'Whether blood pressure is elevated (>140 mmHg)',
        'detail': 'High blood pressure is a major modifiable risk factor for heart disease.',
        'category': 'cardiovascular',
        'is_modifiable': True
    },
    'high_chol': {
        'description': 'Whether cholesterol is elevated (>240 mg/dL)',
        'detail': 'High cholesterol is a key modifiable risk factor for heart disease.',
        'category': 'metabolic',
        'is_modifiable': True
    },
    'asymptomatic_cp': {
        'description': 'Presence of silent (asymptomatic) chest pain',
        'detail': 'Silent ischemia can indicate heart problems without typical warning symptoms.',
        'category': 'symptoms',
        'is_modifiable': False
    }
}

# Actionable recommendations for modifiable factors
FEATURE_RECOMMENDATIONS = {
    'trestbps': 'Reduce sodium intake, exercise regularly, manage stress, and maintain healthy weight.',
    'chol': 'Reduce saturated fats, eat more fiber, include omega-3 fatty acids, and exercise regularly.',
    'high_bp': 'Monitor blood pressure regularly, reduce salt, limit alcohol, and consider medication if advised.',
    'high_chol': 'Adopt a heart-healthy diet, increase physical activity, and discuss statins with your doctor.',
    'metabolic_risk': 'Focus on overall lifestyle: balanced diet, regular exercise, weight management.',
    'age_bp_risk': 'Blood pressure management becomes more important with age. Monitor closely.',
    'age_chol_risk': 'Cholesterol management is crucial as you age. Regular screening recommended.',
    'thalach': 'Improve cardiovascular fitness through regular aerobic exercise.',
    'age_adjusted_hr': 'Build cardiovascular endurance with progressive exercise training.',
    'exercise_stress_index': 'Work with your doctor to develop a safe exercise program.'
}

# Category display names and icons
CATEGORY_INFO = {
    'cardiovascular': {'name': 'Heart & Blood Vessels', 'icon': 'favorite'},
    'metabolic': {'name': 'Metabolic Health', 'icon': 'science'},
    'symptoms': {'name': 'Symptoms', 'icon': 'healing'},
    'test_results': {'name': 'Medical Test Results', 'icon': 'assignment'},
    'demographic': {'name': 'Personal Factors', 'icon': 'person'}
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
    global model, feature_names, shap_explainer, shap_explainer_cache

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
    shap_explainer_cache = None  # reset cache on reload


def ensure_shap_explainer():
    """
    Build or reuse a SHAP explainer to avoid recreating expensive KernelExplainers.
    Prefers TreeExplainer for tree-based models (fast), falls back to stored explainer.
    """
    global shap_explainer_cache

    if shap_explainer_cache is not None:
        return shap_explainer_cache

    if model is None or shap_explainer is None:
        return None

    try:
        # If the pickle already contains an explainer/callable, reuse it.
        if hasattr(shap_explainer, 'shap_values') or callable(shap_explainer):
            shap_explainer_cache = shap_explainer
            return shap_explainer_cache

        # Use fast TreeExplainer for tree-based models (DecisionTree/RandomForest).
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            background = shap_explainer if isinstance(shap_explainer, (pd.DataFrame, np.ndarray)) else None
            shap_explainer_cache = shap.TreeExplainer(model, data=background, model_output="probability")
            return shap_explainer_cache

        # Fallback: build KernelExplainer once if we have background data.
        if isinstance(shap_explainer, (pd.DataFrame, np.ndarray)):
            shap_explainer_cache = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], shap_explainer)
            return shap_explainer_cache

    except Exception as e:
        print(f"SHAP explainer setup error: {e}")

    shap_explainer_cache = None
    return None


def get_shap_explanations(input_df):
    """Get SHAP explanations for a prediction using KernelExplainer."""
    if model is None:
        return []

    explainer = ensure_shap_explainer()
    if explainer is None:
        return []

    try:
        # Compute SHAP values from cached explainer
        if hasattr(explainer, 'shap_values'):
            shap_values = explainer.shap_values(input_df)
        else:
            shap_values = explainer(input_df)

        # Handle different SHAP return formats
        if isinstance(shap_values, list):
            shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        values_array = np.array(shap_values)
        # Squeeze to 1D: some SHAP returns are (1, n, 2) or (1, n)
        values_array = np.squeeze(values_array)
        if values_array.ndim > 1:
            values_array = values_array[0]
        # Guard: if returned length doesn't match features, bail out to fallback
        if len(values_array) != input_df.shape[1]:
            return []

        explanations = []
        for i, feat in enumerate(input_df.columns):
            shap_val = float(values_array[i])

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


def build_tornado_factors(input_df_eng, shap_explanations, feature_importance):
    """
    Build a list of factors for the tornado chart. Guarantees non-empty output by
    falling back to feature importance with direction inferred from deltas to median.
    Now includes explanations, recommendations, and categories for better UX.
    """
    factors = []
    row = input_df_eng.iloc[0].to_dict()

    # Median background for direction hint if we have a DataFrame
    medians = {}
    if isinstance(shap_explainer, pd.DataFrame):
        medians = shap_explainer.median()

    def add_factor(feature, impact, direction_is_risk):
        # Get explanation data if available
        explanation_data = FEATURE_EXPLANATIONS.get(feature, {})
        recommendation = FEATURE_RECOMMENDATIONS.get(feature, None)
        category = explanation_data.get('category', 'other')
        category_info = CATEGORY_INFO.get(category, {'name': 'Other', 'icon': 'info'})

        factors.append({
            'feature': feature,
            'label': FEATURE_LABELS.get(feature, feature),
            'impact': float(abs(impact)),
            'direction': bool(direction_is_risk),
            'value': float(row.get(feature, 0)) if feature in row else None,
            # New UX fields
            'description': explanation_data.get('description', ''),
            'detail': explanation_data.get('detail', ''),
            'category': category,
            'category_name': category_info['name'],
            'category_icon': category_info['icon'],
            'is_modifiable': explanation_data.get('is_modifiable', False),
            'recommendation': recommendation
        })

    # Preferred: SHAP with sign
    if shap_explanations:
        for item in shap_explanations:
            add_factor(item['feature'], item['shap_value'], item['shap_value'] > 0)
    else:
        # Fallback: feature importance with direction inferred from delta to median
        if not feature_importance:
            # If no importance, create a minimal one from existing columns
            feature_importance = {k: float(abs(v)) for k, v in row.items()}
        for feature, importance in feature_importance.items():
            baseline = medians.get(feature, 0)
            current_val = row.get(feature, baseline)
            delta = 0 if current_val is None else (current_val - baseline)
            direction = delta >= 0
            add_factor(feature, importance if importance != 0 else abs(delta), direction)

    # Deduplicate by feature, keep highest impact
    dedup = {}
    for f in sorted(factors, key=lambda x: x['impact'], reverse=True):
        if f['feature'] not in dedup:
            dedup[f['feature']] = f

    # Order by priority list first, then remaining by impact
    ordered = []
    for feat in TORNADO_FEATURE_ORDER:
        if feat in dedup:
            ordered.append(dedup.pop(feat))
    # Append any leftovers by impact
    ordered.extend(sorted(dedup.values(), key=lambda x: x['impact'], reverse=True))

    # Limit to top 8 for readability
    return ordered[:8]


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

        # Capture the engineered input values for UI display
        input_values = {}
        for k, v in input_df_eng.iloc[0].to_dict().items():
            if isinstance(v, (np.floating, float, int, np.integer)):
                input_values[k] = float(v)
            else:
                input_values[k] = v

        # Build tornado factors (never empty)
        tornado_factors = build_tornado_factors(input_df_eng, shap_explanations, feature_importance)

        # Prepare response
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability,
            'feature_importance': feature_importance,
            'shap_explanations': shap_explanations,
            'tornado_factors': tornado_factors,
            'inputs': input_values,
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
