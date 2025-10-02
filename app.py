import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data and model
df = pd.read_csv('Car_Sell_Dataset.csv')
model = joblib.load('pipe.pkl')

# ---- Page Configuration ----
st.set_page_config(page_title="Car Selling Price Prediction", layout="centered")

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
        section.main > div, .block-container, .css-1v0mbdj {
            background-color: #FFF1F5 !important;
        }

        .stSidebar > div:first-child {
            background-color: #FADADD !important;
            padding: 20px;
            border-radius: 10px;
        }

        h1, h2, h3 {
            color: #B76E79 !important;
        }

        .stSidebar label, .stSidebar span, .stSidebar p {
            color: #7A5C61 !important;
            font-weight: 500;
        }

        .stButton > button {
            background-color: #FF69B4 !important;
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            font-weight: bold;
        }

        .stButton > button:hover {
            background-color: #D94F70 !important;
            color: white;
        }

        .stAlert {
            background-color: #FFEFF5 !important;
            border-left: 5px solid #D4AF37 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Car_Silhouette.svg/512px-Car_Silhouette.svg.png", width=100)
st.title("🚗 Used Car Selling Price Prediction")

# ---- Sidebar ----
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["About", "Prediction", "Model Info"])
st.sidebar.markdown("---")

# ---- About Page ----
if page == "About":
    st.subheader("📘 About This App")
    st.write("""
        Welcome! This app predicts the **selling price** of a used car using a trained machine learning model.

        💡 Just enter the car features and get the price estimate!

        The model is trained using historical car sales data and includes preprocessing using `StandardScaler`, `OneHotEncoder`, and `LinearRegression`.

        ---
        Developed with ❤️ by **Shobhana**
    """)

# ---- Prediction Page ----
elif page == "Prediction":
    st.subheader("🛠️ Enter Car Details for Prediction")

    with st.form("car_input_form"):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Brand", df['Brand'].dropna().unique())
            model_name = st.selectbox("Model Name", df['Model Name'].dropna().unique())
            fuel = st.selectbox("Fuel Type", df['Fuel Type'].dropna().unique())
            state = st.selectbox("State", df['State'].dropna().unique())
            owner = st.selectbox("Owner", df['Owner'].dropna().unique())
            accidental = st.selectbox("Accidental", df['Accidental'].dropna().unique())

        with col2:
            model_variant = st.selectbox("Model Variant", df['Model Variant'].dropna().unique())
            transmission = st.selectbox("Transmission", df['Transmission'].dropna().unique())
            year = st.slider("Year of Purchase", 2000, 2024, 2015)
            kms = st.number_input("Kilometers Driven", min_value=0, step=1000)
            car_type = st.selectbox("Car Type", df['Car Type'].dropna().unique())

        submitted = st.form_submit_button("🔮 Predict Price")

    if submitted:
        input_df = pd.DataFrame({
            'Brand': [brand],
            'Model Name': [model_name],
            'Model Variant': [model_variant],
            'Fuel Type': [fuel],
            'Transmission': [transmission],
            'Year': [year],
            'Kilometers': [kms],
            'Owner': [owner],
            'Car Type': [car_type],
            'Accidental': [accidental],
            'State': [state],
            'Price': [0]
        })

        st.subheader("📋 Entered Car Details")
        st.write(input_df)

        try:
            result = model.predict(input_df)[0]
            st.success(f"💰 **Estimated Selling Price:** ₹ {round(result, 2)} Lakhs")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ---- Model Info Page ----
elif page == "Model Info":
    st.subheader("📈 Model & Data Info")
    st.write("The trained pipeline includes feature scaling, encoding, and a Linear Regression model for predicting selling price.")

    fig, ax = plt.subplots()
    sns.histplot(df['Price'], kde=True, ax=ax, color='#5C3B3B')
    ax.set_title("Selling Price Distribution")
    st.pyplot(fig)

    st.markdown("---")
    st.write("📊 **Available Columns in Dataset:**")
    st.write(df.columns.tolist())
