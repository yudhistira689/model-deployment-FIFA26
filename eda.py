import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run():
    st.title("FIFA Data Exploration")

    # #gambar 
    st.image("https://wallpapercave.com/wp/wp15596323.jpg",
            caption="source: Google image")

    # header
    st.header("Latar Belakang")
    # markdown
    st.markdown('''
            Menurut laporan [FIFA 2022](https://publications.fifa.com/en/annual-report-2021/around-fifa/professional-football-2021/), jumlah pemain sepakbola pada tahun 2021 kurang lebih sebanyak 130.000 pemain. Namun, dalam dataset yang digunakan pada kali ini, hanya mencakup 20.000 pemain saja.

    Project kali ini bertujuan untuk memprediksi rating pemain FIFA 2022 sehingga semua pemain sepak bola profesional dapat diketahui ratingnya dan tidak menutup kemungkinan untuk lahirnya talenta/wonderkid baru.

    Project ini akan dibuat menggunakan algoritma Linear Regresison dan akan dievaluasi dengan menggunakan metrics MAE (Mean Absolute Error). '''  )

    st.header('Dataset')
    st.markdown(' Rating dan atribut pemain fifa 2026')

    #load dataset
    data = pd.read_csv('https://raw.githubusercontent.com/FTDS-learning-materials/phase-1/refs/heads/v2.3/w1/P1W1D1PM%20-%20Machine%20Learning%20Problem%20Framing.csv')
    data.rename(columns={'ValueEUR': 'Price', 'Overall': 'Rating'}, inplace=True)
    # tampilkan dataframe
    st.dataframe(data)

    #EDA
    st.header('Exploratory Data Analysis')

    st.subheader('Player Rating Distribution')
    # rating histogram
    fig = plt.figure(figsize=(16, 5))
    sns.histplot(data['Rating'], kde=True, bins=30)
    plt.title('Histogram of Rating')

    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown('Terlihat dari Histogram Plot diatas bahwa `Rating` memiliki distribusi normal dengan mayoritas data berada pada rentang `60` hingga `70`. `Height` dan `Weight` mempunyai relasi yang searah. Artinya, semakin besar nilai `Height` maka nilai `Weight` juga akan semakin besar. Dapat disimpulkan bahwa mayoritas pemain sepak bola pada dataset ini memiliki kondisi tubuh yang proporsional')

    # weight vs height
    st.subheader('Weight vs Height Distribution')
    #plotly chart
    fig = px.scatter(data, x='Weight', y='Height', hover_name='Name')
    st.plotly_chart(fig)

    # viusalisasi based on user input
    st.subheader('Player Stat Distribution')

    # nama kolom yang ada Total-nya
    nama_kolom = data.columns
    total_cols = [col for col in nama_kolom if 'Total' in col]

    # user input
    pilihan = st.selectbox('Pilih atribut untuk divisualisasikan',
                        options = total_cols)

    # visualisasi
    fig = plt.figure(figsize=(16, 5))
    sns.histplot(data[pilihan], kde=True, bins=30)
    plt.title(f'Histogram of {pilihan}')
    # menampilkan visualisasi
    st.pyplot(fig)

    # plotly
    # fig = px.histogram(data, x=pilihan, nbins=30)
    # st.plotly_chart(fig)

if __name__ == '__main__':
    run()