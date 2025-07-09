import streamlit as st
import joblib
import plotly.express as px
import pandas as pd

data = px.data.tips()
model = joblib.load("tips_model.pkl")
label_encoders = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title('Tips Prediction App')

col1, col2 = st.columns(2)

with col1:
    total_bill = st.number_input(label = "total bil")
    sex = st.selectbox(label= "sex", options= ["Male", "Female"])
    smoker = st.selectbox(label= "smoker", options= ["Yes", "No"])
with col2:
    day = st.selectbox(label= "day", options= list(data['day'].unique()))
    time = st.selectbox(label= "time", options= list(data['time'].unique()))
    size = st.number_input(label = "size")


if st.button(label="predict"):
    data = [[total_bill, sex, smoker, day,time, size]]
    data = pd.DataFrame(data=data, columns = ["total_bill",'sex',
                                              'smoker','day','time','size'])
    cat_cols = data.select_dtypes(include= "object")
    for col in cat_cols:
        data[col] = label_encoders[col].transform(data[col])
    columns = data.columns
    data = scaler.transform(data)
    data = pd.DataFrame(data=data, columns=columns)
    pred = model.predict(data)
    st.success(f'This customer is expected to tip about ${round(pred[0], 2)}')

    

