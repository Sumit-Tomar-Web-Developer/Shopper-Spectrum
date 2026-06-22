
import streamlit as st
import pandas as pd
import joblib

st.title("Shopper Spectrum")

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
similarity_df = joblib.load("product_similarity.pkl")

st.header("Product Recommendation")

product = st.text_input("Enter Product Name")

if st.button("Get Recommendations"):
    if product in similarity_df.index:
        recs = similarity_df[product].sort_values(ascending=False)[1:6]
        for r in recs.index:
            st.write("•", r)
    else:
        st.error("Product not found")

st.header("Customer Segmentation")

r = st.number_input("Recency", min_value=0)
f = st.number_input("Frequency", min_value=0)
m = st.number_input("Monetary", min_value=0.0)

if st.button("Predict Cluster"):
    data = scaler.transform([[r,f,m]])
    cluster = kmeans.predict(data)[0]
    st.success(f"Cluster: {cluster}")
