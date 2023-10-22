import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler

# Fungsi untuk menghitung dan memvisualisasikan hasil Fuzzy C-Means
def fuzzy_cmeans(data, n_clusters, m, max_iter, error):
    # Mengubah DataFrame menjadi array NumPy
    data_array = data.values

    # Inisialisasi variabel
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_array.T, n_clusters, m, error=error, maxiter=max_iter, init=None
    )

    # Menentukan klaster untuk setiap data
    cluster_membership = pd.Series(u.argmax(axis=0), index=data.index, name="Cluster")

    # Menambahkan kolom Cluster ke DataFrame
    data_clustered = pd.concat([data, cluster_membership], axis=1)

    # Memvisualisasikan hasil klasterisasi
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Fuzzy C-Means Clustering")
    for cluster in range(n_clusters):
        ax.scatter(
            data_clustered[data.columns[0]][data_clustered["Cluster"] == cluster],
            data_clustered[data.columns[1]][data_clustered["Cluster"] == cluster],
            label=f"Cluster {cluster}",
        )
    ax.legend()
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    st.pyplot(fig)

    return data_clustered

# Fungsi untuk melakukan normalisasi
def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return normalized_data

# Main program
def main():
    st.title("Fuzzy C-Means Clustering")

    # Meminta pengguna memasukkan URL file CSV dari GitHub
    url = st.text_input("Masukkan URL file CSV:", "")

    if url:
        try:
            # Membaca data dari file CSV
            data = pd.read_csv(url)

            # Menampilkan data asli
            st.subheader("Data Asli")
            st.write(data)

            # Meminta pengguna memasukkan kolom yang akan digunakan untuk klasterisasi
            selected_columns = st.multiselect(
                "Pilih kolom untuk klasterisasi:", data.columns
            )

            # Meminta pengguna memasukkan jumlah klaster
            n_clusters = st.number_input(
                "Masukkan jumlah klaster:", min_value=2, max_value=10, value=2, step=1
            )

            # Meminta pengguna memasukkan nilai keanggotaan
            m = st.number_input(
                "Masukkan nilai keanggotaan (m):", min_value=1.1, max_value=100.0, value=2.0, step=0.1
            )

            # Meminta pengguna memasukkan batasan maksimum iterasi
            max_iter = st.number_input(
                "Masukkan batasan maksimum iterasi:", min_value=1, max_value=1000, value=100, step=1
            )

            # Meminta pengguna memasukkan batasan kesalahan
            error = st.number_input(
                "Masukkan batasan kesalahan:", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001
            )

            # Mengambil subset data dengan kolom yang dipilih
            selected_data = data[selected_columns]

            # Normalisasi data
            normalized_data = normalize_data(selected_data)

            # Menampilkan data setelah normalisasi
            st.subheader("Data Setelah Normalisasi")
            st.write(normalized_data)

            # Menjalankan Fuzzy C-Means
            data_clustered = fuzzy_cmeans(normalized_data, n_clusters, m, max_iter, error)

            # Menampilkan data hasil klasterisasi
            st.subheader("Data Hasil Klasterisasi")
            st.write(data_clustered)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
