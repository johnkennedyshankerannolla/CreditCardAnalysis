# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 📌 Internship App Title and Header
# ==========================================================
st.set_page_config(page_title="Customer Fraud Detection App", layout="wide")

st.title("💼 John Kennedy Shankerannolla's Internship Application")
st.markdown("**Organization:** DataHill Solutions, Hyderabad")
st.markdown("**Project Title:** *End-to-End Customer Transaction Analytics & Fraud Detection System*")
st.markdown("---")

st.write("📁 Upload a CSV file with transaction data to predict whether each entry is fraudulent or genuine.")

# ==========================================================
# 🚀 Load the trained Keras model
# ==========================================================
model = tf.keras.models.load_model("fraud_detection_nn_model.h5")

# ==========================================================
# 🎯 Prediction Function — Corrected to expect 30 inputs
# ==========================================================
def predict_fraud(data):
    # Keep only Time, Amount, and V1-V28 columns
    base_cols = ['Time', 'Amount'] + [f"V{i}" for i in range(1, 29)]
    data = data[base_cols]

    # Scale Time and Amount
    scaler = StandardScaler()
    data[['scaled_amount', 'scaled_time']] = scaler.fit_transform(data[['Amount', 'Time']])

    # Drop unscaled Time and Amount
    data = data.drop(['Amount', 'Time'], axis=1)

    # Final column order: V1–V28 + scaled_amount + scaled_time (30 total)
    final_columns = [f"V{i}" for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
    data = data[final_columns]

    # Predict
    preds = model.predict(data)
    preds_binary = (preds > 0.5).astype(int)

    # Return dataframe with predictions
    data['Fraud_Probability'] = preds
    data['Prediction'] = preds_binary
    return data[['Fraud_Probability', 'Prediction']]

# ==========================================================
# 📤 File Upload Section
# ==========================================================
uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Transaction Data")
    st.dataframe(input_df.head())

    # Button to start prediction
    if st.button("🔍 Predict Fraud"):
        with st.spinner("Predicting..."):
            results = predict_fraud(input_df.copy())
            output_df = pd.concat([input_df, results], axis=1)

        st.success("✅ Prediction complete!")
        st.subheader("📊 Results Preview")
        st.dataframe(output_df.head(20))

        # Display fraud vs. genuine count
        fraud_count = output_df['Prediction'].value_counts()
        st.write(f"🔴 Fraudulent: {fraud_count.get(1, 0)} | 🟢 Genuine: {fraud_count.get(0, 0)}")

        # Optionally let user download
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv, file_name="predicted_results.csv", mime='text/csv')

else:
    st.warning("⚠️ Please upload a CSV file with columns: Time, Amount, V1–V28.")
