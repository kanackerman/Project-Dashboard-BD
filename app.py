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
    df.columns = df.columns.str.strip()  # Membersihkan spasi pada nama kolom
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ File 'FINAL_USO.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan `app.py`.")
    st.stop()

# ====================================================
# PERSIAPAN VARIABEL DEFAULT
# ====================================================
numeric_df = df.select_dtypes(include=[np.number])
numeric_cols = numeric_df.columns.tolist()

# Menentukan variabel target default jika ada kolom GLD atau Close
if 'GLD' in df.columns:
    target_col = 'GLD'
elif 'Close' in df.columns:
    target_col = 'Close'
else:
    target_col = numeric_cols[-1] if numeric_cols else ""

feature_cols = [col for col in numeric_cols if col != target_col]

# ====================================================
# SIDEBAR / NAVIGASI
# ====================================================
st.sidebar.title("📌 Navigasi Dashboard")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Overview Dataset", "Visualisasi Trend & Korelasi", "Model Regresi Linier"]
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
    
    if 'Date' in df.columns:
        col3.metric("Tanggal Mulai", df['Date'].min().strftime('%Y-%m-%d'))
        col4.metric("Tanggal Selesai", df['Date'].max().strftime('%Y-%m-%d'))
    else:
        col3.metric("Tanggal Mulai", "N/A")
        col4.metric("Tanggal Selesai", "N/A")
    
    st.markdown("---")
    
    # Preview Data
    st.subheader("📋 Preview Dataset")
    num_rows = st.slider("Pilih Jumlah Baris Tampil:", 5, 50, 5)
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    # Informasi Tipe Data dan Missing Value
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🍩 Distribusi Tipe Data")
        dtype_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(dtype_counts, labels=[str(dt) for dt in dtype_counts.index], autopct="%1.1f%%", startangle=90)
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
# 2. HALAMAN: VISUALISASI TREND & KORELASI
# ====================================================
elif menu == "Visualisasi Trend & Korelasi":
    st.title("📈 Visualisasi Pergerakan & Korelasi")

    # 1. Line Chart
    st.subheader("1. Pergerakan Harga Seiring Waktu")
    selected_metrics = st.multiselect(
        "Pilih kolom harga yang ingin ditampilkan:",
        options=numeric_cols,
        default=[target_col] if target_col in numeric_cols else [numeric_cols[0]]
    )

    if selected_metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        x_axis = df['Date'] if 'Date' in df.columns else df.index
        for metric in selected_metrics:
            ax.plot(x_axis, df[metric], label=metric)
        ax.set_title("Grafik Pergerakan Harga")
        ax.set_xlabel("Waktu / Indeks")
        ax.set_ylabel("Nilai")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    st.divider()

    # 2. Correlation Heatmap
    st.subheader("2. Heatmap Korelasi Antar Variabel")
    if not numeric_df.empty:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Korelasi Matriks")
        st.pyplot(fig_corr)

    st.divider()

    # 3. Scatter Plot Korelasi 2 Variabel
    st.subheader("3. Scatter Plot Hubungan Antar 2 Variabel")
    if not numeric_df.empty and len(numeric_cols) >= 2:
        col_x, col_y = st.columns(2)
        var_x = col_x.selectbox("Pilih Variabel Sumbu X:", options=numeric_cols, index=0)
        var_y = col_y.selectbox("Pilih Variabel Sumbu Y:", options=numeric_cols, index=min(1, len(numeric_cols)-1))

        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4))
        sns.regplot(data=df, x=var_x, y=var_y, ax=ax_scatter, scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
        ax_scatter.set_title(f"Hubungan antara {var_x} dan {var_y}")
        ax_scatter.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_scatter)
    else:
        st.warning("Tidak cukup kolom numerik untuk menampilkan Scatter Plot.")

# ====================================================
# 3. HALAMAN: MODEL REGRESI LINIER
# ====================================================
elif menu == "Model Regresi Linier":
    st.title("🤖 Simulasi Regresi Linier (Machine Learning)")
    st.write("Lakukan eksperimen regresi linier sederhana menggunakan variabel independen pilihan Anda untuk memprediksi variabel target.")
    
    col1, col2 = st.columns(2)
    with col1:
        default_target_idx = numeric_cols.index(target_col) if target_col in numeric_cols else 0
        target_var = st.selectbox("Pilih Variabel Target (Y):", numeric_cols, index=default_target_idx)
    with col2:
        default_features = [c for c in ["Open", "High", "Low", "SP_close"] if c in numeric_cols]
        if not default_features:
            default_features = [numeric_cols[0]] if numeric_cols else []
            
        feature_vars = st.multiselect(
            "Pilih Variabel Fitur/Prediktor (X):",
            numeric_cols,
            default=default_features
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
