import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import kagglehub
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Customer Segmentation Tool", page_icon="🛍️", layout="wide")

# --- FUNGSI LOAD DATA (DI-CACHE) ---
@st.cache_data
def load_data():
    # Download dataset menggunakan kagglehub
    path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
    dataset_path = os.path.join(path, "Mall_Customers.csv")
    df = pd.read_csv(dataset_path)
    return df

# Memuat data
df_raw = load_data()
X = df_raw.iloc[:, [3, 4]] # Mengambil Annual Income dan Spending Score

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Cluster")
st.sidebar.write("Sesuaikan parameter analisis di bawah ini:")

# Input jumlah cluster dari user
k_value = st.sidebar.slider("Jumlah Cluster (k)", 2, 10, 5)

st.sidebar.divider()
st.sidebar.info("""
**Info Fitur:**
- **Annual Income:** Pendapatan Tahunan (k$)
- **Spending Score:** Skor Pengeluaran (1-100)
""")

# --- MAIN PAGE ---
st.title("📊 Customer Segmentation Dashboard")
st.markdown("""
Aplikasi ini melakukan segmentasi pelanggan menggunakan algoritma **K-Means Clustering**. 
Membantu bisnis memahami kelompok pelanggan berdasarkan pendapatan dan kebiasaan belanja mereka.
""")

st.divider()

# --- BAGIAN 1: ELBOW METHOD & INFORMASI DATA ---
col_info, col_elbow = st.columns([1, 1.2])

with col_info:
    st.subheader("📋 Ringkasan Data")
    st.write(f"Total Baris Data: `{df_raw.shape[0]}`")
    st.write(f"Fitur yang digunakan: `Annual Income` & `Spending Score`")
    
    # Menghitung WCSS untuk Elbow Method
    wcss = []
    for i in range(1, 11):
        # Menambahkan n_init='auto' atau 10 untuk menghindari FutureWarning
        kmeans_elbow = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=14)
        kmeans_elbow.fit(X)
        wcss.append(kmeans_elbow.inertia_)
    
    st.success("Dataset berhasil dimuat dari Kaggle!")
    st.dataframe(df_raw.describe(), use_container_width=True)

with col_elbow:
    st.subheader("📈 The Elbow Method")
    fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
    ax_elbow.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax_elbow.set_title('Mencari Jumlah Cluster Optimal')
    ax_elbow.set_xlabel('Number of Clusters')
    ax_elbow.set_ylabel('WCSS')
    st.pyplot(fig_elbow)

st.divider()

# --- BAGIAN 2: PROSES K-MEANS ---
st.subheader("🚀 Hasil Segmentasi Pelanggan")

# Menjalankan K-Means berdasarkan input slider
kmeans = KMeans(n_clusters=k_value, init='k-means++', n_init=10, random_state=14)
with st.spinner('Menghitung cluster...'):
    time.sleep(0.5)
    y_kmeans = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

# Menambahkan hasil cluster ke dataframe
df_result = X.copy()
df_result['Cluster'] = y_kmeans
df_result['CustomerID'] = df_raw['CustomerID']

# Layout untuk Grafik Hasil
col_chart1, col_chart2 = st.columns([1.5, 1])

with col_chart1:
    # Visualisasi Scatter Plot
    st.write("**Visualisasi Sebaran Cluster**")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", n_colors=k_value)
    
    for i in range(k_value):
        ax_scatter.scatter(
            X.iloc[y_kmeans == i, 0], 
            X.iloc[y_kmeans == i, 1], 
            s=60, c=[colors[i]], 
            label=f'Cluster {i+1}',
            edgecolors='black', linewidth=0.5
        )
    
    # Plot Centroids
    ax_scatter.scatter(
        centroids[:, 0], centroids[:, 1], 
        s=200, c='red', marker='X', 
        label='Centroids', edgecolors='white'
    )
    
    ax_scatter.set_title("Segmentasi Pelanggan Berdasarkan Pendapatan & Pengeluaran")
    ax_scatter.set_xlabel("Annual Income (k$)")
    ax_scatter.set_ylabel("Spending Score (1-100)")
    ax_scatter.legend()
    st.pyplot(fig_scatter)

with col_chart2:
    # Visualisasi Bar Chart Frekuensi
    st.write("**Distribusi Jumlah Pelanggan**")
    fig_bar, ax_bar = plt.subplots(figsize=(7, 7.5))
    data_counts = df_result['Cluster'].value_counts().sort_index()
    
    sns.barplot(
        x=[f"C-{i+1}" for i in data_counts.index], 
        y=data_counts.values, 
        palette="viridis", ax=ax_bar
    )
    ax_bar.set_ylabel("Jumlah Customer")
    ax_bar.set_xlabel("Nama Cluster")
    st.pyplot(fig_bar)

st.divider()

# --- BAGIAN 3: TABEL DATA ---
with st.expander("🔍 Lihat Hasil Data Lengkap per Cluster"):
    st.write("Gunakan fitur filter di bawah ini untuk melihat member tiap cluster.")
    selected_cluster = st.selectbox("Pilih Cluster untuk ditampilkan:", [f"Cluster {i+1}" for i in range(k_value)])
    
    cluster_idx = int(selected_cluster.split(" ")[1]) - 1
    filtered_df = df_result[df_result['Cluster'] == cluster_idx]
    
    st.dataframe(filtered_df, use_container_width=True)

# Footer
st.caption("Dashboard Analisis K-Means | Dibuat untuk Eksplorasi Data Pelanggan")
