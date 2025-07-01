import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import threading

# Lock to prevent race condition during model prediction
prediction_lock = threading.Lock()

# Cache model loading to avoid reloading on every rerun
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_safe.h5")

# Cache pickle file loading
@st.cache_resource
def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Load resources
model = load_model()
label_encoder_gender = load_pickle('label_encoder_gender.pkl')
onehot_encoder_geo = load_pickle('onehot_encoder_geo.pkl')
scaler = load_pickle('scaler.pkl')

# Streamlit UI
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850)
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_data_scaled = scaler.transform(input_data)

with prediction_lock:
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Probability: {prediction_proba:.2f}')
    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.') 
