import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================================================
# KONFIGURASI HALAMAN STREAMLIT
# ====================================================
st.set_page_config(
    page_title="Dashboard Analisis Finansial (FINAL_USO)",
    page_icon="📈",
    layout="wide"
)

# Set style matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ====================================================
# LOAD DATASET (DENGAN CACHING UNTUK PERFORMA)
# ====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("FINAL_USO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ File 'FINAL_USO.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan `app.py`.")
    st.stop()

# ====================================================
# SIDEBAR / NAVIGASI
# ====================================================
st.sidebar.title("📌 Navigasi Dashboard")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Overview Dataset", "Exploratory Data Analysis (EDA)", "Model Regresi Linier"]
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini dibuat berdasarkan analisis data keuangan dari dataset **FINAL_USO.csv**.")

# ====================================================
# 1. HALAMAN: OVERVIEW DATASET
# ====================================================
if menu == "Overview Dataset":
    st.title("📊 Overview Dataset USO (United States Oil Fund)")
    st.write("Eksplorasi ringkasan data, dimensi dataset, tipe data, dan sampel baris pertama.")
    
    # Ringkasan Statistik Utama dalam Kartu (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris Data", f"{df.shape[0]:,}")
    col2.metric("Total Kolom Data", df.shape[1])
    col3.metric("Tanggal Mulai", df['Date'].min().strftime('%Y-%m-%d'))
    col4.metric("Tanggal Selesai", df['Date'].max().strftime('%Y-%m-%d'))
    
    st.markdown("---")
    
    # Preview Data
    st.subheader("📋 Preview Dataset")
    num_rows = st.slider("Pilih Jumlah Baris Tampil:", 5, 50, 5)
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    # Informasi Tipe Data dan Missing Value
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🍩 Distribusi Tipe Data")
        dtype_counts = df.dtypes.value_counts().astype(str)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(df.dtypes.value_counts(), labels=dtype_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
    with col_right:
        st.subheader("🔍 Pengecekan Missing Value")
        total_null = df.isnull().sum().sum()
        if total_null == 0:
            st.success("✅ Tidak ditemukan Missing Value pada dataset ini!")
        else:
            st.warning(f"⚠️ Ditemukan {total_null} missing value.")
            
        st.subheader("📈 Ringkasan Statistik Deskriptif")
        st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

# ====================================================
# 2. HALAMAN: EXPLORATORY DATA ANALYSIS (EDA)
# ====================================================
elif menu == "Exploratory Data Analysis (EDA)":
    st.title("📈 Exploratory Data Analysis (EDA)")
    
    # Visualisasi Tren Harga Berdasarkan Waktu
    st.subheader("📉 Tren Harga Historis")
    available_cols = [c for c in df.columns if c != 'Date']
    selected_cols = st.multiselect(
        "Pilih Indikator/Kolom untuk Ditampilkan pada Grafik Tren:",
        options=available_cols,
        default=["Close", "SP_close", "USO_Close"]
    )
    
    if selected_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in selected_cols:
            ax.plot(df['Date'], df[col], label=col)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Nilai / Harga")
        ax.set_title("Grafik Pergerakan Harga Finansial")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Pilih minimal satu kolom untuk menampilkan grafik.")
        
    st.markdown("---")
    
    # Visualisasi Korelasi
    st.subheader("🔥 Matriks Korelasi (Heatmap)")
    corr_cols = st.multiselect(
        "Pilih Kolom untuk Dihitung Korelasinya:",
        options=available_cols,
        default=["Open", "High", "Low", "Close", "Volume", "SP_close", "USO_Close", "GDX_Close"]
    )
    
    if len(corr_cols) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("Pilih minimal 2 kolom untuk melihat heatmap korelasi.")
   
    #Visualisasi Scaterplot

    st.subheader("3. Scatter Plot Hubungan Antar 2 Variabel (Visualisasi Baru)")
    col_x, col_y = st.columns(2)
    var_x = col_x.selectbox("Pilih Variabel Sumbu X:", options=numeric_df.columns, index=0)
    var_y = col_y.selectbox("Pilih Variabel Sumbu Y:", options=numeric_df.columns, index=min(1, len(numeric_df.columns)-1))

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4))
    sns.regplot(data=df, x=var_x, y=var_y, ax=ax_scatter, scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
    ax_scatter.set_title(f"Hubungan antara {var_x} dan {var_y}")
    ax_scatter.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_scatter)

    # Histogram / KDE Plot
    with col_hist:
        st.subheader(f"Histogram Sebaran Nilai ({selected_var})")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        sns.histplot(df[selected_var], kde=True, color="skyblue", ax=ax_hist)
        ax_hist.set_title(f"Distribusi {selected_var}")
        st.pyplot(fig_hist)
# ====================================================
# 3. HALAMAN: MODEL REGRESI LINIER
# ====================================================
elif menu == "Model Regresi Linier":
    st.title("🤖 Simulasi Regresi Linier (Machine Learning)")
    st.write("Lakukan eksperimen regresi linier sederhana menggunakan variabel independen pilihan Anda untuk memprediksi variabel target.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        target_var = st.selectbox("Pilih Variabel Target (Y):", numeric_cols, index=numeric_cols.index("Close") if "Close" in numeric_cols else 0)
    with col2:
        feature_vars = st.multiselect(
            "Pilih Variabel Fitur/Prediktor (X):",
            numeric_cols,
            default=["Open", "High", "Low", "SP_close"] if all(k in numeric_cols for k in ["Open", "High", "Low", "SP_close"]) else [numeric_cols[0]]
        )
        
    test_size = st.slider("Proporsi Data Uji (Test Size %):", 10, 40, 20) / 100.0
    
    if st.button("🚀 Latih Model Regresi"):
        if not feature_vars:
            st.error("Pilih setidaknya satu variabel fitur (X).")
        else:
            X = df[feature_vars]
            y = df[target_var]
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Model Training
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Prediction & Evaluation
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Tampilkan Hasil Evaluasi
            st.subheader("🎯 Hasil Evaluasi Model")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MAE", f"{mae:.4f}")
            m3.metric("MSE", f"{mse:.4f}")
            m4.metric("RMSE", f"{rmse:.4f}")
            
            # Plot Aktual vs Prediksi
            st.subheader("📊 Grafik Harga Aktual vs Hasil Prediksi")
            fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
            ax_pred.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Prediksi vs Aktual')
            ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal')
            ax_pred.set_xlabel("Nilai Aktual")
            ax_pred.set_ylabel("Nilai Prediksi")
            ax_pred.legend()
            st.pyplot(fig_pred)
