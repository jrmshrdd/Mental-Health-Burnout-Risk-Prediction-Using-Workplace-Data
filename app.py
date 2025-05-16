#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv("survey.csv")
    df = df.dropna(subset=['work_interfere'])
    
    # Fixed LabelEncoder for work_interfere
    work_interfere_labels = ['Never', 'Rarely', 'Sometimes', 'Often']
    le_work_interfere = LabelEncoder()
    le_work_interfere.classes_ = np.array(work_interfere_labels)
    df['work_interfere_enc'] = le_work_interfere.transform(df['work_interfere'])
    
    # Encode target variable
    le_target = LabelEncoder()
    df['target'] = le_target.fit_transform(df['mental_health_interview'].astype(str))
    
    features = ['work_interfere_enc', 'self_employed', 'remote_work', 'tech_company', 'benefits', 'care_options']
    
    # Encode other categorical features
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    X = df[features]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_work_interfere

model, label_encoder = load_model()

st.title("Mental Health Burnout Risk Predictor")

st.markdown("""
Enter your details below to get a personalized burnout risk prediction.
""")

def user_input_features():
    work_interfere = st.selectbox(
        "How often does your work interfere with your mental health?",
        ['Never', 'Rarely', 'Sometimes', 'Often']
    )
    self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])
    remote_work = st.selectbox("Do you work remotely?", ["No", "Yes"])
    tech_company = st.selectbox("Do you work for a tech company?", ["No", "Yes"])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["No", "Yes", "Don't know"])
    care_options = st.selectbox("Are you aware of the care options available?", ["No", "Yes", "Don't know"])

    input_dict = {
        'work_interfere_enc': label_encoder.transform([work_interfere])[0],
        'self_employed': 1 if self_employed == "Yes" else 0,
        'remote_work': 1 if remote_work == "Yes" else 0,
        'tech_company': 1 if tech_company == "Yes" else 0,
        'benefits': 2 if benefits == "Don't know" else (1 if benefits == "Yes" else 0),
        'care_options': 2 if care_options == "Don't know" else (1 if care_options == "Yes" else 0),
    }
    return pd.DataFrame([input_dict])

input_df = user_input_features()

st.subheader("Input Features")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

risk = "High Risk" if prediction[0] == 1 else "Low Risk"

st.subheader("Burnout Risk Prediction")
st.write(f"Predicted burnout risk level: **{risk}**")
st.write(f"Probability: {prediction_proba[0][prediction[0]]:.2f}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

st.subheader("Feature Impact on Prediction")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
st.pyplot(fig)






# In[ ]:




