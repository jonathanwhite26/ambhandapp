# -----------------------------------------------------------------------------------
# AMBULANCE HANDOVER TIME MODEL
# -----------------------------------------------------------------------------------
# Overview for Data Scientists:
#
# This Streamlit web app allows interactive what-if analysis on ambulance handover times.
# It is built on a basic multivariate linear regression model (OLS) using the `statsmodels` library.
# The model is trained on historic hourly data extracted from SHREWD (last 3 months to March 2025),
# and makes predictions about handover times given changes in operational or system variables.
#
# Users select the features to include in the regression model and interactively adjust values
# using sliders. The model output updates in real-time and compares against an operational plan.
#
# Features:
# - Uses cached CSV loading functions for speed.
# - Computes model coefficients and prediction.
# - Calculates difference between predicted and default scenario.
# - Adjusts monthly and annual plans based on scenario.
# - Offers annotated model performance summary.
# - Outputs dynamic plots and data tables.
#
# Target Users:
# - NHS managers, analysts, and operational leads.
# - Designed as a transparent and interpretable proof of concept.
# -----------------------------------------------------------------------------------

import pandas as pd  # For data handling
import numpy as np  # For numerical operations
import statsmodels.api as sm  # For OLS regression model
import matplotlib.pyplot as plt  # For plotting
import streamlit as st  # For building interactive web apps

# Set the Streamlit layout to use the full browser width
st.set_page_config(layout="wide")

# -------------------------
# Load data from Excel workbook (SharePoint-hosted)
# -------------------------
@st.cache_data
def load_excel_data():
    file_path = 'https://nhs-my.sharepoint.com/personal/jonathan_white26_nhs_net/_layouts/15/download.aspx?SourceUrl=https://nhs-my.sharepoint.com/:x:/g/personal/jonathan_white26_nhs_net/EX07mIrBOk1AhamM_x4K5xQBVVwr-K3W4LgHagTqqHgeHg?e=g72qEc'
    xls = pd.ExcelFile(file_path)

    df = xls.parse('DailyInput')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    plan_df = xls.parse('PlanData')
    plan_df['Planned'] = pd.to_timedelta(plan_df['Planned'], errors='coerce').dt.total_seconds() / 60
    plan_df['Month'] = pd.to_datetime('01-' + plan_df['Month'], format='%d-%b-%y', errors='coerce')
    initial_rows = len(plan_df)
    plan_df.dropna(subset=['Month'], inplace=True)
    if len(plan_df) < initial_rows:
        st.warning(f"Dropped {initial_rows - len(plan_df)} rows with invalid month format.")
    plan_df.sort_values('Month', inplace=True)
    plan_df['MonthStr'] = plan_df['Month'].dt.strftime('%b %Y')

    return df, plan_df

# -------------------------
# Load monthly plan data and preprocess
# -------------------------
@st.cache_data
def load_plan_data():
    import os
    file_path = 'https://nhs-my.sharepoint.com/personal/jonathan_white26_nhs_net/_layouts/15/download.aspx?SourceUrl=https://nhs-my.sharepoint.com/personal/jonathan_white26_nhs_net/Documents/AmbHandPlan.csv'
    df = pd.read_csv(file_path)
    df['Planned'] = pd.to_timedelta(df['Planned'], errors='coerce').dt.total_seconds() / 60  # Convert time to minutes
    df['Month'] = pd.to_datetime('01-' + df['Month'], format='%d-%b-%y', errors='coerce')  # Force first of month
    initial_rows = len(df)
    df.dropna(subset=['Month'], inplace=True)  # Drop bad dates
    if len(df) < initial_rows:
        st.warning(f"Dropped {initial_rows - len(df)} rows with invalid month format.")
    df.sort_values('Month', inplace=True)
    df['MonthStr'] = df['Month'].dt.strftime('%b %Y')  # Add a string version of month for plotting
    return df

# -------------------------
# Load both datasets and define target/feature columns
# -------------------------
df, plan_df = load_excel_data()
y = df.iloc[:, 1]  # Target variable (handover time)
all_features = df.columns[2:16]  # Independent variables to allow selection

# -------------------------
# UI: Title, instructions, and sidebar controls
# -------------------------
st.title("Ambulance Handover Time Model")
st.header("Interactive What-If Analysis")
st.markdown("Adjust the values below to simulate impact on handover time predictions:")

st.sidebar.header("Feature Selection")
selected_features = st.sidebar.multiselect("Select independent variables to include in the model, everything is included by default. Removing variables that are weakly correlated can improve the model and reduce overfitting:", all_features, default=list(all_features))

# -------------------------
# Model training and prediction
# -------------------------
if selected_features:
    X = df[selected_features]
    X = sm.add_constant(X)  # Add intercept to the model
    model = sm.OLS(y, X).fit()  # Fit OLS regression
    predicted = model.predict(X).clip(lower=0)  # Ensure predictions are non-negative
    df['Predicted'] = predicted

    # -------------------------
    # User interaction via sliders to simulate inputs
    # -------------------------
    default_values = X.mean()
    input_values = {'const': 1.0}

    coefs = model.params.drop('const')
    sorted_features = coefs.abs().sort_values(ascending=False).index.tolist()
    positive_features = [f for f in sorted_features if coefs[f] > 0]
    negative_features = [f for f in sorted_features if coefs[f] < 0]

    st.markdown("### 📈 Positively Correlated Inputs")
    st.markdown("Increasing these will increase handover times, they are ordered in terms of significance")
    pos_cols = st.columns(2)
    for idx, col in enumerate(positive_features):
        col_index = idx % 2
        with pos_cols[col_index]:
            initial_val = float(default_values.get(col, 0))
            slider_val = st.slider(
                label=f"{col} (β={coefs[col]:.2f})",
                min_value=float(X[col].min() * 0.5),
                max_value=float(X[col].max() * 1.5),
                value=initial_val,
                key=f"{col}_pos"
            )
            delta_percent = ((slider_val - initial_val) / initial_val) * 100 if initial_val != 0 else 0
            
            input_values[col] = slider_val

    st.markdown("### 📉 Negatively Correlated Inputs")
    st.markdown("Increasing these will reduce handover times, they are ordered in terms of significance")
    neg_cols = st.columns(2)
    for idx, col in enumerate(negative_features):
        col_index = idx % 2
        with neg_cols[col_index]:
            initial_val = float(default_values.get(col, 0))
            slider_val = st.slider(
                label=f"{col} (β={coefs[col]:.2f})",
                min_value=float(X[col].min() * 0.5),
                max_value=float(X[col].max() * 1.5),
                value=initial_val,
                key=f"{col}_neg"
            )
            delta_percent = ((slider_val - initial_val) / initial_val) * 100 if initial_val != 0 else 0
            
            input_values[col] = slider_val

    input_series = pd.Series(input_values).reindex(X.columns, fill_value=0)  # Convert slider values to Series
    prediction = model.predict(input_series).clip(lower=0)[0]  # Predict new handover time
    default_pred = model.predict(default_values).clip(lower=0)[0]  # Base case prediction
    percent_change = ((prediction - default_pred) / default_pred) * 100  # Change from base case
    

    input_series = pd.Series(input_values).reindex(X.columns, fill_value=0)  # Convert slider values to Series
    prediction = model.predict(input_series).clip(lower=0)[0]  # Predict new handover time
    default_pred = model.predict(default_values).clip(lower=0)[0]  # Base case prediction
    percent_change = ((prediction - default_pred) / default_pred) * 100  # Change from base case

    # -------------------------
    # Adjust the plan based on % change from sliders
    # -------------------------
        

    st.markdown("## 📅 Annual Handover Time 2025/26 Plan Comparison")
    # -------------------------
    # Explanation of adjusted inputs (as a single sentence)
    # -------------------------
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
    # Monthly plot of handover time
    # -------------------------
    st.header("Monthly Plan Comparison")
    fig_plan, ax_plan = plt.subplots(figsize=(8, 3))
    ax_plan.plot(plan_df['MonthStr'], plan_df['Planned'], marker='o', label='Original Plan')
    ax_plan.plot(plan_df['MonthStr'], plan_df['Revised'], marker='o', label='Revised Plan')
    ax_plan.set_title("Monthly Handover Time Plan vs Revised")
    ax_plan.set_xlabel("Month")
    ax_plan.set_ylabel("Handover Time (minutes)")
    ax_plan.legend()
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_plan)

    # -------------------------
    # Monthly values table
    # -------------------------
    st.subheader("Monthly Comparison Table")
    st.dataframe(plan_df[['MonthStr', 'Planned', 'Revised']].rename(columns={
        'MonthStr': 'Month', 'Planned': 'Original Plan (mins)', 'Revised': 'Revised Plan (mins)'
    }))

    # -------------------------
    # Model performance and explanation
    # -------------------------
    st.header("Model Training Summary")
    st.markdown("""
This model is trained on hourly data extracted from SHREWD. The sample period is the past 3 months to the end of March 2025.  
Only complete daily samples are used in the model training. This is a basic machine learning model — it isn't artificial intelligence.  
It simply measures the relationship between each of the input variables and ambulance handover times from historic data and applies that proportionally to the 2025/26 annual plan.

Some of these relationships are likely to be counter-intuitive. Its intended use is to aid decision making by showing how sensitive handover times have historically been to changes in other key measures from across the system.  

Other measures can be added if the data is available. The model can also be refined in other ways such as including more training data and using more advanced techniques.  
Please consider it an initial release proof of concept.
""")

    st.header("Model Summary Stats")
    st.text("""R-squared: {:.2f} - This tells us that {:.0f}% of the variance in handover time is explained by the model.
Generally, an R-squared value above 0.6 is considered acceptable for operational models. Above 0.7 is good, and above 0.8 is very strong. Our model's R-squared score indicates a {} model fit.

Adjusted R-squared: {:.2f} - Adjusts for the number of predictors to avoid overfitting.
If this value is significantly lower than the R-squared, it suggests that some input variables are not contributing meaningful predictive power and may be adding noise rather than signal.

F-statistic: {:.2f} (p = {:.1e}) - Tests whether the model as a whole is statistically significant.
The F-statistic compares the model to one with no predictors. A high value with a very low p-value (typically < 0.05) suggests the model is statistically significant.
Our model's F-statistic indicates that the chosen predictors, taken together, explain a significant amount of variation in ambulance handover times.
A high F-statistic and low p-value suggest that the model provides a better fit than a model with no predictors.
""".format(
    model.rsquared, model.rsquared * 100,
    "very strong" if model.rsquared > 0.8 else "good" if model.rsquared > 0.7 else "reasonable" if model.rsquared > 0.6 else "weak",
    model.rsquared_adj, model.fvalue, model.f_pvalue))

    # -------------------------
    # Prediction output section
    # -------------------------
    delta_color = "normal" if percent_change <= 0 else "off"
    st.markdown("## 🧶 Predicted Handover Time For the Training Period")
    st.markdown("""This section shows what the model predicts when applied to the training data""")
    st.markdown(f"<h1 style='font-size: 48px;'>{prediction:.2f} mins</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {'red' if percent_change > 0 else 'green'};'>{percent_change:+.2f}% change</h3>", unsafe_allow_html=True)
  
    # -------------------------
    # Actual vs Predicted chart
    # -------------------------
    st.header("Actual vs Predicted Handover Time")
    st.markdown("""This section shows how well the model fits the historic data used to train it. Ideally it should be close but not too close which could be a sign of overfitting. Check for periods from the training data where the model doesnt fit the data well, these could be a caused by excpetional circumstances""")
    updated_X = X.copy()
    has_changes = False
    for col in selected_features:
        if not np.isclose(input_values[col], default_values[col]):
            updated_X[col] = input_values[col]
            has_changes = True
    model_output = model.predict(updated_X).clip(lower=0) if has_changes else predicted.copy()
    df['Model Output'] = model_output

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df.index, y, label='Actual')
    ax.plot(df.index, predicted, label='Predicted')
    ax.plot(df.index, df['Model Output'], label='Current Model Output')
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Handover Time (minutes)")
    ax.legend()
    st.pyplot(fig)

    # -------------------------
    # Final output table
    # -------------------------
    st.subheader("Predicted Values Table")
    predicted_df = pd.DataFrame({
        'Actual Handover Time': y,
        'Predicted Handover Time': predicted,
        'Current Model Output': df['Model Output']
    })
    st.dataframe(predicted_df)
else:
    st.warning("Please select at least one independent variable to build the model.")


