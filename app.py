# -----------------------------------------------------------------------------------
# AMBULANCE HANDOVER TIME MODEL (V4 ENHANCED)
# -----------------------------------------------------------------------------------
# This version includes all features from V3 except model variable selection.
# Variables, coefficients, and plan data are hardcoded.
# -----------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Set the Streamlit layout to use the full browser width
st.set_page_config(layout="wide")

# -------------------------
# Hardcoded model parameters from trained regression
# -------------------------
# Updated model parameters from the new OLS regression results
model_intercept = -280.1827
model_coefficients = {
    '% Patients Not Meeting Criteria to Reside - Adult': 6.8585,
    'Unvalidated 4hr % Performance (Since Midnight) - ED All-Type': -0.8295,
    '% Open beds that are escalation beds': 22.8535,
    'CFT - Virtual Ward % Occupancy': -0.2681,
    'SWAST - Ambulance Conveyances (Rolling 60 mins) - Royal Cornwall Hospital Treliske (RCH)': -17.1076,
    'Acute OPEL Score 24/26': 2.9249,
    'Total Patients delayed in CHAOS (previous day)': 4.9386
}

# -------------------------
# Hardcoded average/default values
# -------------------------
default_values = {
    '% Patients Not Meeting Criteria to Reside - Adult': 15.52,
    'Unvalidated 4hr % Performance (Since Midnight) - ED All-Type': 43.39,
    '% Open beds that are escalation beds': 1.52,
    'CFT - Virtual Ward % Occupancy': 46.36,
    'SWAST - Ambulance Conveyances (Rolling 60 mins) - Royal Cornwall Hospital Treliske (RCH)': 4.84,
    'Acute OPEL Score 24/26': 68.36,  # Fixed key
    'Total Patients delayed in CHAOS (previous day)': 31.02
}

# -------------------------
# Hardcoded monthly plan data
# -------------------------
plan_data = {
    'Month': [
        'Apr 2025', 'May 2025', 'Jun 2025', 'Jul 2025', 'Aug 2025', 'Sep 2025',
        'Oct 2025', 'Nov 2025', 'Dec 2025', 'Jan 2026', 'Feb 2026', 'Mar 2026'
    ],
    'Planned': [
        73.33, 80.67, 80.08, 47.40, 48.45, 64.72,
        78.20, 54.33, 66.48, 66.93, 49.90, 67.83
    ]
}
plan_df = pd.DataFrame(plan_data)

# -------------------------
# Streamlit UI
# -------------------------
st.title("Ambulance Handover Time Simulator")
st.header("Interactive What-If Analysis")
st.markdown("Adjust the values below to simulate the impact on handover time predictions:")

positive_features = {k: v for k, v in model_coefficients.items() if v > 0}
negative_features = {k: v for k, v in model_coefficients.items() if v < 0}

st.markdown("### 📈 Positively Correlated Inputs")
input_values = {}
pos_cols = st.columns(2)
for idx, (var, coef) in enumerate(positive_features.items()):
    initial_val = default_values[var]
    with pos_cols[idx % 2]:
        slider_val = st.slider(
            label=f"{var} (β={coef:+.2f})",
            min_value=initial_val * 0.5,
            max_value=initial_val * 1.5,
            value=initial_val,
            key=f"pos_{var}"
        )
        input_values[var] = slider_val

st.markdown("### 🖋️ Negatively Correlated Inputs")
neg_cols = st.columns(2)
for idx, (var, coef) in enumerate(negative_features.items()):
    initial_val = default_values[var]
    with neg_cols[idx % 2]:
        slider_val = st.slider(
            label=f"{var} (β={coef:+.2f})",
            min_value=initial_val * 0.5,
            max_value=initial_val * 1.5,
            value=initial_val,
            key=f"neg_{var}"
        )
        input_values[var] = slider_val

# -------------------------
# Prediction Calculation
# -------------------------
prediction = model_intercept + sum(model_coefficients[var] * input_values[var] for var in input_values)
default_prediction = model_intercept + sum(model_coefficients[var] * default_values[var] for var in default_values)
percent_change = ((prediction - default_prediction) / default_prediction) * 100

# -------------------------
# Plan Adjustment
# -------------------------
plan_df['Revised'] = plan_df['Planned'] * (1 + percent_change / 100)
actual_annual = plan_df['Planned'].mean()
revised_annual = plan_df['Revised'].mean()
diff = revised_annual - actual_annual
percent_diff = (diff / actual_annual) * 100
color = 'red' if revised_annual > 45 else 'green'

# Define selected_features to resolve the undefined variable error
selected_features = list(model_coefficients.keys())  # Default to all features if not explicitly defined

# -------------------------
# Annual Results Display
# -------------------------
st.header("\U0001F4C5 Annual Plan Adjustment")
changes = []
for col in selected_features:
    original = default_values[col]
    new = input_values[col]
    if not np.isclose(original, new):
        delta_pct = ((new - original) / original) * 100 if original != 0 else 0
        direction = "increased" if delta_pct > 0 else "decreased"
        changes.append(f"**{col}** is {direction} by {abs(delta_pct):.1f}%")

if changes:
    sentence = " and ".join(changes)
    st.markdown(f"📝 If {sentence}, then the historic data shows that the plan for ambulance handover delays could change to:")
plan_df['Revised'] = plan_df['Planned'] * (1 + percent_change / 100)
actual_annual = plan_df['Planned'].mean()
revised_annual = plan_df['Revised'].mean()
diff = revised_annual - actual_annual
percent_diff = (diff / actual_annual) * 100
color = 'red' if revised_annual > 45 else 'green'

st.markdown(f"<h1 style='color:{color}; font-size: 40px'>Modelled Annual Average: {revised_annual:.2f} mins</h1>", unsafe_allow_html=True)
st.markdown(f"<h4>Original Annual Average: {actual_annual:.2f} mins</h4>", unsafe_allow_html=True)
st.markdown(f"<h4>Difference: {diff:+.2f} mins ({percent_diff:+.2f}%)</h4>", unsafe_allow_html=True)


# -------------------------
# Monthly Chart
# -------------------------
st.header("Monthly Plan Comparison")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(plan_df['Month'], plan_df['Planned'], marker='o', label='Original Plan')
ax.plot(plan_df['Month'], plan_df['Revised'], marker='o', label='Revised Plan')
ax.set_title("Monthly Handover Time Plan vs Revised")
ax.set_xlabel("Month")
ax.set_ylabel("Handover Time (minutes)")
ax.legend()
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# -------------------------
# Table Output
# -------------------------
st.subheader("Monthly Comparison Table")
st.dataframe(plan_df.rename(columns={
    'Month': 'Month',
    'Planned': 'Original Plan (mins)',
    'Revised': 'Revised Plan (mins)'
}))

# -------------------------
# Model Summary and Explanation
# -------------------------
st.header("Model Training Summary")
st.markdown("""
This model is based on a multivariate linear regression using system inputs and operational indicators. It simulates expected ambulance handover times when those inputs change.
The coefficients were trained on historic data up to March 2025.

This tool is not AI, but a transparent and interpretable proof of concept. It shows the **direction** and **relative strength** of how each input has historically impacted handover time.

Further improvements could include better training data, inclusion of additional predictors, or more advanced modelling.
""")

st.header("Model Summary Stats")
st.text("""R-squared: 0.7102 - This tells us that about 71% of the variance in handover time is explained by the model.
This represents a reasonably strong model fit.

Adjusted R-squared: 0.6335 - Adjusts for the number of predictors and helps prevent overfitting.

F-statistic: 9.26 (p = 6.28e-07) - Indicates the model is statistically significant overall.
A high F-statistic and a low p-value suggest the model performs better than one with no predictors.
""")

# -------------------------
# Actual vs Predicted Chart (real data)
# -------------------------
st.header("Actual vs Predicted Handover Time (Training Fit)")
actual_values = [25.385, 50.71375, 69.944583, 95.596667, 26.795, 93.3275, 73.643333, 47.859167, 72.92375, 44.2575, 70.52125, 76.6225, 43.678333, 32.06375, 57.524167, 43.84125, 58.7725, 104.182083, 53.187917, 35.180417, 103.805417, 47.912917, 47.6375, 101.281667, 157.709167, 146.6175, 60.980417, 146.986667, 162.4525, 175.065833, 166.785417, 134.488333, 132.245417, 93.98, 22.925, 31.80125, 49.36625, 107.099583, 172.871667, 122.657083, 104.5175, 119.026667, 37.666957, 29.457083]
predicted_values = [30.74048, 49.116674, 65.586539, 70.617874, 20.289197, 94.32614, 94.235091, 52.396262, 79.511123, 66.532588, 105.670629, 73.353536, 55.073084, 50.249839, 49.83029, 74.20871, 36.230563, 38.282879, 24.963231, 48.278169, 129.874598, 94.745774, 68.575051, 90.080644, 110.687879, 102.083319, 104.692678, 138.224234, 120.743483, 133.759486, 159.995805, 132.006272, 158.123697, 116.178954, 55.479407, 41.171656, 70.349553, 96.396319, 141.1797, 134.532264, 86.845305, 83.390355, 72.246323, 30.502556]
training_df = pd.DataFrame({"Index": range(len(actual_values)), "Actual": actual_values, "Predicted": predicted_values})
fig_fit, ax_fit = plt.subplots(figsize=(8, 3))
ax_fit.plot(training_df['Index'], training_df['Actual'], label='Actual')
ax_fit.plot(training_df['Index'], training_df['Predicted'], label='Predicted')
ax_fit.set_title("Actual vs Predicted Handover Time")
ax_fit.set_xlabel("Index")
ax_fit.set_ylabel("Handover Time (minutes)")
ax_fit.legend()
st.pyplot(fig_fit)
