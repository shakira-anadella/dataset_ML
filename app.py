import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("K-Means Clustering Influencer Instagram")
st.markdown("Masukkan data influencer untuk melihat masuk cluster berapa.")

# Input fitur
influence_score = st.number_input("Influence Score", 0.0)
followers = st.number_input("Jumlah Followers", 0.0)
avg_likes = st.number_input("Rata-rata Likes", 0.0)
total_likes = st.number_input("Total Likes", 0.0)
posts = st.number_input("Jumlah Postingan", 0.0)
new_post_avg_like = st.number_input("Like Posting Terbaru", 0.0)
engagement_rate = st.number_input("Engagement Rate (0.01 = 1%)", 0.0)

if st.button("Prediksi Cluster"):
    data = pd.DataFrame([[
        influence_score, posts, followers, avg_likes,
        engagement_rate, new_post_avg_like, total_likes
    ]], columns=[
        'influence_score', 'posts', 'followers', 'avg_likes',
        '60_day_eng_rate', 'new_post_avg_like', 'total_likes'
    ])
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]
    st.success(f"Influencer ini termasuk dalam **Cluster #{cluster}**")
