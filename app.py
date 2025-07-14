import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Load model dan scaler ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Judul Aplikasi ---
st.title("K-Means Clustering Influencer Instagram")
st.markdown("Masukkan data influencer untuk melihat masuk ke dalam kategori klaster apa.")

# --- Input Data (Integer Only) ---
followers = st.number_input("Jumlah Followers", min_value=0, step=100, format="%d")
avg_likes = st.number_input("Rata-rata Likes", min_value=0, step=10, format="%d")
posts = st.number_input("Jumlah Postingan", min_value=0, step=1, format="%d")

# --- Saat Tombol Diklik ---
if st.button("Prediksi Cluster"):
    # Siapkan DataFrame dari input user
    data = pd.DataFrame([[followers, avg_likes, posts]], columns=['followers', 'avg_likes', 'posts'])

    # Normalisasi data input
    data_scaled = scaler.transform(data)

    # Prediksi klaster
    cluster = model.predict(data_scaled)[0]

    # Mapping hasil cluster ke label kategori
    cluster_labels = {
        0: "Micro Influencer",
        1: "Mid Influencer",
        2: "Mega Influencer"
    }
    label = cluster_labels.get(cluster, f"Cluster #{cluster}")

    # --- Tampilkan Hasil ---
    st.success(f"Influencer ini termasuk dalam: **{label} (Cluster #{cluster})**")

    # --- Visualisasi Data Input User ---
    st.markdown("---")
    st.subheader("Visualisasi Data Input")

    user_input = {
        "Followers": followers,
        "Rata-rata Likes": avg_likes,
        "Jumlah Postingan": posts
    }

    fig, ax = plt.subplots()
    ax.bar(user_input.keys(), user_input.values(), color=['#4CAF50', '#2196F3', '#FFC107'])
    ax.set_ylabel("Jumlah")
    ax.set_title("Data Influencer yang Dimasukkan")
    st.pyplot(fig)
