import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation System")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="latin1")

    df.dropna(subset=["CustomerID", "Description"], inplace=True)

    df["CustomerID"] = df["CustomerID"].astype(int)

    return df

df = load_data()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler

kmeans, scaler = load_models()

# -----------------------------
# Create Similarity Matrix
# -----------------------------
@st.cache_resource
def create_similarity_matrix(data):

    customer_product = pd.crosstab(
        data["CustomerID"],
        data["Description"]
    )

    similarity = cosine_similarity(customer_product.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=customer_product.columns,
        columns=customer_product.columns
    )

    return similarity_df

similarity_df = create_similarity_matrix(df)

# -----------------------------
# Product Recommendation
# -----------------------------
st.header("📦 Product Recommendation")

product = st.text_input(
    "Enter Product Name",
    placeholder="Example: WHITE HANGING HEART T-LIGHT HOLDER"
)

if st.button("Get Recommendations"):

    if product in similarity_df.index:

        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.success("Recommended Products")

        for item in recommendations.index:
            st.write("✅", item)

    else:
        st.error("Product not found in dataset")

# -----------------------------
# Customer Segmentation
# -----------------------------
st.header("👤 Customer Segmentation")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency",
        min_value=0,
        value=10
    )

with col2:
    frequency = st.number_input(
        "Frequency",
        min_value=0,
        value=20
    )

with col3:
    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=1000.0
    )

if st.button("Predict Segment"):

    scaled_data = scaler.transform(
        [[recency, frequency, monetary]]
    )

    cluster = kmeans.predict(scaled_data)[0]

    segment_labels = {
        0: "🌟 High Value Customer",
        1: "👍 Regular Customer",
        2: "🛍️ Occasional Customer",
        3: "⚠️ At Risk Customer"
    }

    st.success(
        f"Predicted Segment: {segment_labels.get(cluster, f'Cluster {cluster}')}"
    )

# -----------------------------
# Dataset Overview
# -----------------------------
st.header("📊 Dataset Overview")

st.metric("Total Transactions", len(df))
st.metric("Unique Customers", df["CustomerID"].nunique())
st.metric("Unique Products", df["Description"].nunique())

st.dataframe(df.head())