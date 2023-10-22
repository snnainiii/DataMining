import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('https://raw.githubusercontent.com/alammahdum8/Project-KK-Data_Diabetes/main/data.csv')

def main():
    st.title("Contoh Navbar di Streamlit")

    menu = ["Beranda", "Profil", "Kontak"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Beranda":
        st.subheader("Beranda")
        data.head()
        data

    elif choice == "Profil":
        st.subheader("Profil")

        # Menghilangkan kolom 'diagnosis' jika ada
        if 'diagnosis' in data.columns:
            data_for_clustering = data.drop('diagnosis', axis=1)
        else:
            data_for_clustering = data

        # Normalisasi data menggunakan StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)

        # Melakukan clustering dengan DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(scaled_data)

        # Mendapatkan hasil label cluster
        cluster_labels = dbscan.labels_

        # Menghitung jarak antar data ke pusat cluster terdekat
        distances = dbscan.fit_transform(scaled_data)

        # Memplot hasil jarak
        plt.figure(figsize=(10, 6))
        plt.hist(distances.flatten(), bins=50, color='skyblue')
        plt.xlabel('Jarak')
        plt.ylabel('Frekuensi')
        plt.title('Distribusi Jarak ke Pusat Cluster Terdekat')
        st.pyplot()

    elif choice == "Kontak":
        st.subheader("Kontak")

if __name__ == "__main__":
    main()
