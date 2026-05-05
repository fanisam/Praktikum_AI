import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

@st.cache_data
def preprocess(df):
    # Encode gender
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df

@st.cache_data
def train_model(df):
    # Kita gunakan Spending Score sebagai target (klasifikasi sederhana)
    df['Spending_Category'] = pd.cut(df['Spending Score (1-100)'],
                                    bins=[0, 40, 70, 100],
                                    labels=["Low", "Medium", "High"])

    X = df[['Gender', 'Age', 'Annual Income (k$)']]
    y = df['Spending_Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy

def main():
    st.set_page_config(page_title="Mall Customer App", layout="centered")
    st.title("🛍️ Klasifikasi Customer Mall")
    st.write("Masukkan data pelanggan:")

    df = load_data()
    df = preprocess(df)
    model, accuracy = train_model(df)

    # Input user
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))
    income = st.slider("Annual Income (k$)", 
                       int(df['Annual Income (k$)'].min()),
                       int(df['Annual Income (k$)'].max()),
                       int(df['Annual Income (k$)'].mean()))

    gender_val = 0 if gender == "Male" else 1

    if st.button("Prediksi"):
        input_data = np.array([[gender_val, age, income]])
        prediction = model.predict(input_data)[0]

        st.success(f"🧾 Kategori Customer: **{prediction}**")
        st.info(f"Akurasi model: {accuracy*100:.2f}%")

    if st.checkbox("Tampilkan Dataset"):
        st.dataframe(df)

if __name__ == "__main__":
    main()
