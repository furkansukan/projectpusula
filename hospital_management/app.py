import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error


# Rastgele hasta verisi üreten fonksiyon
def generate_random_patient_data(start_date='2023-01-01', end_date=None,
                                 mean1=56, std_dev1=11, mean2=70, std_dev2=15, seed=42):
    """
    Rastgele hasta verisi üretir.

    Args:
        start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında).
        end_date (str or None): Bitiş tarihi (YYYY-MM-DD formatında). Varsayılan olarak bugünün tarihi.
        mean1 (float): İlk yarı için normal dağılımın ortalaması.
        std_dev1 (float): İlk yarı için normal dağılımın standart sapması.
        mean2 (float): İkinci yarı için normal dağılımın ortalaması.
        std_dev2 (float): İkinci yarı için normal dağılımın standart sapması.
        seed (int): Rastgele sayı üreticisi için sabitleyici.

    Returns:
        pd.DataFrame: `date` ve `count` sütunlarından oluşan bir DataFrame.
    """
    # Rastgeleliği sabitle
    np.random.seed(seed)

    # Bitiş tarihi varsayılan olarak bugünün tarihi olsun
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Tarihler arası günlük liste oluştur
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')

    # İlk ve ikinci yarı için normal dağılım ile rastgele değerler oluştur
    count_list = np.concatenate([
        np.random.normal(loc=mean1, scale=std_dev1, size=len(date_list) // 2),
        np.random.normal(loc=mean2, scale=std_dev2, size=-(-len(date_list) // 2))  # ceiling division
    ]).astype(int).tolist()

    # Negatif değerleri 1 ile değiştir
    count_list = [max(1, x) for x in count_list]

    # DataFrame oluştur
    df = pd.DataFrame({
        'ds': date_list,
        'y': count_list
    })

    return df

# Hastalar verisini analiz eden fonksiyon
def analyze_patient_data(df):
    """
    Hasta verilerini analiz eder ve çeşitli görseller oluşturur.

    Args:
        df (pd.DataFrame): 'ds' ve 'y' sütunlarını içeren bir DataFrame.

    Returns:
        dict: Analiz sonuçları (en yüksek hasta haftası, günü vb.).
    """
    results = {}

    # 'ds' sütununu dstime formatına çevir
    df['ds'] = pd.to_datetime(df['ds'])

    # Haftalık hasta sayısını hesapla
    df['week'] = df['ds'].dt.to_period('W')
    weekly_data = df.groupby('week')['y'].sum()

    # Aylık hasta sayısını hesapla
    df['month'] = df['ds'].dt.to_period('M')
    monthly_data = df.groupby('month')['y'].sum()

    # Mevsimsel hasta sayısını hesapla
    df['season'] = df['ds'].dt.month % 12 // 3 + 1  # 1: Kış, 2: İlkbahar, 3: Yaz, 4: Sonbahar
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    df['season'] = df['season'].map(seasons)
    seasonal_data = df.groupby('season')['y'].sum()

    # En yüksek hasta sayısının olduğu haftayı bul
    highest_week = weekly_data.idxmax()
    highest_week_y = weekly_data.max()
    results['highest_week'] = (highest_week, highest_week_y)

    # En fazla hasta gelen günü bul
    highest_day = df.loc[df['y'].idxmax()]
    results['highest_day'] = {'ds': highest_day['ds'], 'y': highest_day['y']}

    # Streamlit görselleştirmeleri
    st.subheader("Haftalık Hasta Sayısı")
    fig, ax = plt.subplots(figsize=(10, 6))
    weekly_data.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Haftalık Hasta Sayısı", fontsize=18)
    ax.set_xlabel('Hafta', fontsize=14)
    ax.set_ylabel('Toplam Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Aylık Hasta Sayısı")
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_data.plot(kind='bar', color='salmon', ax=ax)
    ax.set_title("Aylık Hasta Sayısı", fontsize=18)
    ax.set_xlabel('Ay', fontsize=14)
    ax.set_ylabel('Toplam Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Mevsimsel Hasta Sayısı")
    fig, ax = plt.subplots(figsize=(10, 6))
    seasonal_data.plot(kind='bar', color='lightgreen', ax=ax)
    ax.set_title("Mevsimsel Hasta Sayısı", fontsize=18)
    ax.set_xlabel('Mevsim', fontsize=14)
    ax.set_ylabel('Toplam Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Zaman Serisi: Günlük Hasta Sayısı")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.set_index('ds')['y'].plot(color='purple', ax=ax)
    ax.set_title("Zaman Serisi: Günlük Hasta Sayısı", fontsize=18)
    ax.set_xlabel('Tarih', fontsize=14)
    ax.set_ylabel('Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Hasta Sayısının Dağılımı (Histogram)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['y'], kde=True, color='orange', ax=ax)
    ax.set_title("Hasta Sayısının Dağılımı", fontsize=18)
    ax.set_xlabel('Hasta Sayısı', fontsize=14)
    ax.set_ylabel('Frekans', fontsize=14)
    st.pyplot(fig)

    st.subheader(f"En Fazla Hasta Gelen Gün: {highest_day['ds'].strftime('%Y-%m-%d')}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(highest_day['ds'], highest_day['y'], color='red')
    ax.set_title(f"En Fazla Hasta Gelen Gün: {highest_day['ds'].strftime('%Y-%m-%d')}", fontsize=18)
    ax.set_xlabel('Tarih', fontsize=14)
    ax.set_ylabel('Hasta Sayısı', fontsize=14)
    st.pyplot(fig)

    # Sonuçları Streamlit üzerinde yazdır
    st.write(f"Haftalık en yüksek hasta sayısı, {highest_week} haftasında: {highest_week_y}")
    st.write(f"En fazla hasta gelen gün: {highest_day['ds']} ({highest_day['y']} hasta)")

    return results

# Streamlit arayüzü
st.title("Randevusuz Hasta Yönetim ve Doktor Atama Sistemi")


# CSV dosyasını yükleme bölümü
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])

# Eğer dosya yüklenmişse, dosyayı oku
if uploaded_file is not None:
    # Dosyayı pandas ile oku
    df = pd.read_csv(uploaded_file)

    # Veriyi görüntüle
    st.write("Yüklenen Veri:", df.head())
    analyze_patient_data(df)
else:
    # Eğer dosya yüklenmemişse, rastgele veri üret
    st.write("CSV dosyası yüklenmedi, rastgele hasta verisi üretiliyor.")
    df = generate_random_patient_data()

    # Üretilen veriyi göster
    st.write("Rastgele Üretilen Veri:", df.head())
    analyze_patient_data(df)
# Veriyi görselleştir
st.line_chart(df.set_index('ds')['y'])


