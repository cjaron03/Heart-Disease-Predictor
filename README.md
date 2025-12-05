# Heart Disease Risk Advisor

A machine learning project for predicting cardiovascular disease risk using the UCI Cleveland Heart Disease dataset.

**Course:** CS445 - Artificial Intelligence

## Project Overview

This project implements a heart disease prediction system following the CRISP-DM methodology:

1. **Data Understanding** - Exploratory analysis with interactive Plotly visualizations
2. **Data Preparation** - Missing value handling, target binarization, train/test split
3. **Modeling** - Decision Tree and Random Forest classifiers with hyperparameter tuning
4. **Evaluation** - Model comparison and insights on key risk factors
5. **Deployment** - Interactive web-based Heart Risk Advisor

## Dataset

- **Source:** UCI Machine Learning Repository - Cleveland Heart Disease Dataset
- **Records:** 303 patients
- **Features:** 13 clinical attributes (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Target:** Heart disease presence (binarized: 0 = no disease, 1 = disease)

## Project Structure

```
Heart-Disease-Predictor/
├── processed.cleveland.data    # Dataset
├── heart_disease_analysis.ipynb # Main analysis notebook (CRISP-DM)
├── model_utils.py              # ML helper functions
├── app.py                      # Flask web application
├── templates/
│   └── index.html              # Web frontend
├── static/
│   └── style.css               # Styling
├── best_model.pkl              # Trained model (generated)
├── feature_names.pkl           # Feature list (generated)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Jupyter notebook to train the model:**
   ```bash
   jupyter notebook heart_disease_analysis.ipynb
   ```
   Run all cells to:
   - Perform data analysis and visualizations
   - Train and evaluate models
   - Save the best model to `best_model.pkl`

3. **Launch the web application:**
   ```bash
   python app.py
   ```
   Open http://localhost:5000 in your browser.

## Features

### Jupyter Notebook
- Interactive Plotly visualizations (correlation heatmap, box plots, distributions)
- Decision Tree and Random Forest models with GridSearchCV tuning
- Model comparison table with accuracy, precision, recall, and F1 scores
- Feature importance analysis
- Ethical considerations and dataset limitations discussion

### Web Application
- User-friendly form for entering health metrics
- Real-time risk prediction using the trained model
- Visual display of key risk factors
- Color-coded results (green = lower risk, red = higher risk)
- Educational disclaimer

## Models

| Model | Description |
|-------|-------------|
| Decision Tree (Default) | Baseline tree classifier |
| Decision Tree (Tuned) | Optimized `max_depth` via cross-validation |
| Random Forest (Tuned) | Optimized `n_estimators` and `max_depth` |

## Disclaimer

This tool is for **educational purposes only** and is NOT a substitute for professional medical advice. Always consult healthcare providers for actual medical decisions.
