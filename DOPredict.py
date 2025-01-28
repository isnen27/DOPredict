import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_path = 'xgboost_sfs_model.pkl'
xgboost_model = joblib.load(model_path)

# Title and description
st.title("Student Dropout Prediction")
st.write("This application predicts whether a student is at risk of dropping out (DO) based on their academic and personal data.")

# Form for user input
with st.form("prediction_form"):
    st.header("Enter Student Details")

    # Input fields
    ccourse = st.number_input("1_Ccourse (Value between 0 and 1):", min_value=0.0, max_value=1.0, step=0.01)
    ccourse_good = st.number_input("2_CCourse_good (Value between 0 and 1):", min_value=0.0, max_value=1.0, step=0.01)
    toefl_score = st.number_input("3_TOEFL_score:", min_value=0.0, step=0.1)
    college_time = st.number_input("4_College_time (Hours per week):", min_value=0.0, step=0.1)
    gpa = st.number_input("5_GPA (Value between 0 and 4):", min_value=0.0, max_value=4.0, step=0.01)
    academic_leave = st.number_input("6_Academic_Leave (Number of times):", min_value=0.0, step=0.1)

    domicile = st.selectbox("7_domicile:", ["Urban", "Rural"])
    work = st.selectbox("8_work:", ["Yes", "No"])
    live_with_family = st.selectbox("9_live_with_family:", ["Yes", "No"])

    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode categorical features
    domicile_encoded = 1 if domicile == "Urban" else 0
    work_encoded = 1 if work == "Yes" else 0
    live_with_family_encoded = 1 if live_with_family == "Yes" else 0

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        "1_Ccourse": [ccourse],
        "2_CCourse_good": [ccourse_good],
        "3_TOEFL_score": [toefl_score],
        "4_College_time": [college_time],
        "5_GPA": [gpa],
        "6_Academic_Leave": [academic_leave],
        "7_domicile": [domicile_encoded],
        "8_work": [work_encoded],
        "9_live_with_family": [live_with_family_encoded]
    })

    # Make a prediction
    prediction = xgboost_model.predict(input_data)[0]

    # Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("The student is at risk of dropping out (DO).")
    else:
        st.success("The student is not at risk of dropping out (DO).")
