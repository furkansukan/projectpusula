import random
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error


# Rastgele hasta verisi üreten fonksiyon
def generate_random_patient_data(start_date='2023-01-01', end_date=None,
                                 mean1=56, std_dev1=11, mean2=70, std_dev2=15):
    #np.random.seed(seed)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    count_list = np.concatenate([
        np.random.normal(loc=mean1, scale=std_dev1, size=len(date_list) // 2),
        np.random.normal(loc=mean2, scale=std_dev2, size=-(-len(date_list) // 2))
    ]).astype(int).tolist()
    count_list = [max(1, x) for x in count_list]
    df = pd.DataFrame({'ds': date_list, 'y': count_list})
    return df

def main_page():
    st.title("Ana Sayfa - Veri İnceleme")
    st.write("Şu anda rastgele oluşturulmuş veriler ile sayfayı görüntülemektesiniz, isterseniz kendi dosyanızı ekleyebilirsiniz.")
    uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])








    if uploaded_file is None:
        df = generate_random_patient_data()
        st.write(df.head())

        # Son 30 günü al
        df_last_30_days = df[df['ds'] >= (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')]

        # Veriyi görselleştir
        st.write("Son 30 Günün Hasta Verisinin Zaman İçindeki Değişimi:")
        plt.figure(figsize=(12, 6))
        plt.plot(df_last_30_days['ds'], df_last_30_days['y'], marker='o', linestyle='-', color='b')
        plt.title("Son 30 Günün Hasta Verisinin Zaman İçindeki Değişimi")
        plt.xlabel("Tarih")
        plt.ylabel("Hasta Sayısı")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Veriyi session state'e kaydediyoruz
        st.session_state.df = df

    else:
        df = pd.read_csv(uploaded_file)
        if 'ds' in df.columns and 'y' in df.columns:
            st.write("Yüklenen CSV dosyasının ilk 5 satırı:")
            st.write(df.head())

            # Son 30 günü al
            df_last_30_days = df[df['ds'] >= (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')]

            # Veriyi görselleştir
            st.write("Son 30 Günün Yüklenen Verisinin Zaman İçindeki Değişimi:")
            plt.figure(figsize=(12, 6))
            plt.plot(df_last_30_days['ds'], df_last_30_days['y'], marker='o', linestyle='-', color='g')
            plt.title("Son 30 Günün Yüklenen Verisinin Zaman İçindeki Değişimi")
            plt.xlabel("Tarih")
            plt.ylabel("Hasta Sayısı")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Veriyi session state'e kaydediyoruz
            st.session_state.df = df
        else:
            st.warning("CSV dosyasında 'ds' ve 'y' sütunları bulunamadı. Lütfen uygun dosyayı yükleyin.")
            df = generate_random_patient_data()
            st.write(df.head())

            # Son 30 günü al
            df_last_30_days = df[df['ds'] >= (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')]

            # Veriyi görselleştir
            st.write("Son 30 Günün Hasta Verisinin Zaman İçindeki Değişimi:")
            plt.figure(figsize=(12, 6))
            plt.plot(df_last_30_days['ds'], df_last_30_days['y'], marker='o', linestyle='-', color='b')
            plt.title("Son 30 Günün Hasta Verisinin Zaman İçindeki Değişimi")
            plt.xlabel("Tarih")
            plt.ylabel("Hasta Sayısı")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Veriyi session state'e kaydediyoruz
            st.session_state.df = df


def analyze_patient_data(df):
    results = {}
    df['ds'] = pd.to_datetime(df['ds'])
    df['week'] = df['ds'].dt.to_period('W')
    weekly_data = df.groupby('week')['y'].sum()

    df['month'] = df['ds'].dt.to_period('M')
    monthly_data = df.groupby('month')['y'].sum()

    df['season'] = df['ds'].dt.month % 12 // 3 + 1
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    df['season'] = df['season'].map(seasons)
    seasonal_data = df.groupby('season')['y'].sum()

    highest_week = weekly_data.idxmax()
    highest_week_y = weekly_data.max()
    results['highest_week'] = (highest_week, highest_week_y)

    highest_day = df.loc[df['y'].idxmax()]
    results['highest_day'] = {'ds': highest_day['ds'], 'y': highest_day['y']}

    # Görselleştirmeler
    # Son 10 haftayı almak
    last_10_weeks = weekly_data.tail(10)

    # Grafiği çizmek
    plt.figure(figsize=(10, 6))
    last_10_weeks.plot(kind='bar', color='skyblue')
    plt.title("Son 10 Haftalık Hasta Sayısı", fontsize=18)
    plt.xlabel('Hafta', fontsize=14)
    plt.ylabel('Toplam Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    monthly_data.plot(kind='bar', color='salmon')
    plt.title("Aylık Hasta Sayısı", fontsize=18)
    plt.xlabel('Ay', fontsize=14)
    plt.ylabel('Toplam Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    seasonal_data.plot(kind='bar', color='lightgreen')
    plt.title("Mevsimsel Hasta Sayısı", fontsize=18)
    plt.xlabel('Mevsim', fontsize=14)
    plt.ylabel('Toplam Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    df.set_index('ds')['y'].plot(color='purple')
    plt.title("Zaman Serisi: Günlük Hasta Sayısı", fontsize=18)
    plt.xlabel('Tarih', fontsize=14)
    plt.ylabel('Hasta Sayısı', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['y'], kde=True, color='orange')
    plt.title("Hasta Sayısının Dağılımı", fontsize=18)
    plt.xlabel('Hasta Sayısı', fontsize=14)
    plt.ylabel('Frekans', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    plt.bar(highest_day['ds'], highest_day['y'], color='red')
    plt.title(f"En Fazla Hasta Gelen Gün: {highest_day['ds'].strftime('%Y-%m-%d')}", fontsize=18)
    plt.xlabel('Tarih', fontsize=14)
    plt.ylabel('Hasta Sayısı', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

    return results

def analysis_page():
    st.title("Veri Görselleştirme")

    if 'df' in st.session_state:
        df = st.session_state.df
        results = analyze_patient_data(df)

    else:
        st.warning("Ana sayfada veri üretilmedi. Lütfen önce ana sayfayı ziyaret edin.")


# Prophet modelini ve MAE hesaplama fonksiyonunu tanımladığınız kod burada
def prophet_forecast_with_mae(df, date_col='ds', count_col='y', test_days=7, country='TR'):
    """
    Prophet modeli ile zaman serisi tahmini yapar, MAE hesaplar ve sonuçları yorumlar.
    """
    # Veriyi Train/Test olarak böl
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]

    # Prophet modelini oluştur ve eğit
    model = Prophet(weekly_seasonality=True, yearly_seasonality=False, changepoint_prior_scale=0.5,
                    uncertainty_samples=60000, interval_width=0.95)

    if country:
        model.add_country_holidays(country_name=country)  # Ülkeye özel tatilleri ekle

    model.fit(train)

    # Gelecekteki veriyi oluştur (test günlerini dahil et)
    future = model.make_future_dataframe(periods=test_days)

    # Prophet modeline tahmin yaptır
    forecast = model.predict(future)

    # Tahminleri gerçek değerlerle karşılaştır
    y_true = test[count_col].values
    y_pred = forecast['yhat'].iloc[-test_days:].values

    # MAE hesapla
    mae = mean_absolute_error(y_true, y_pred)

    # MAE'yi detaylı açıklama ile döndür
    explanation = (
        f"MAE (Mean Absolute Error): {mae:.2f}\n"
        f"Tanım: Tahmin edilen değerler ile gerçek değerler arasındaki mutlak hatanın ortalamasıdır.\n"
        f"Prophet modelimiz, tahminlerinde ortalama olarak gerçek değerden {mae:.2f} birim sapıyor.\n"
    )

    # Tahminleri "predict_table" adlı bir tabloya kaydet
    predict_table = forecast[['ds', 'yhat']].tail(test_days)

    # Görsel olarak tahminleri grafik üzerinde sun
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_col], df[count_col], label='Gerçek Veri', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Tahminler', color='red', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2,
                     label='Tahmin Aralığı')
    plt.xlabel('Tarih')
    plt.ylabel(count_col)
    plt.title('Prophet Modeli Tahminleri')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Streamlit'te görseli göster
    st.session_state.predict_table = predict_table

    return forecast, mae, explanation, predict_table


# Streamlit sayfasını oluştur
def model_page():
    st.title("Hasta Sayısı Tahminleme")

    # Sağ kenar için bir başlık
    with st.sidebar:
        st.header("Model Açıklaması")
        st.write(
            """
            Bu sayfa, zaman serisi verilerini tahmin etmek ve modelin performansını değerlendirmek için tasarlanmıştır. Kullanılan yöntem ve işlemler şunlardır:

            - **Prophet Modeli:** Facebook tarafından geliştirilmiş, zaman serisi tahminleri için kullanılan bir modeldir. Bu sayfada:
              - Haftalık ve yıllık sezonluk etkiler göz önüne alınmıştır.
              - Ülke bazlı tatiller (Türkiye için "TR") modele eklenmiştir.
              - Belirlenen bir tarih aralığı için tahmin yapılmıştır.

            - **Model Eğitimi ve Tahmin:**
              - Veriler, eğitim (train) ve test setlerine ayrılmıştır.
              - Eğitim seti üzerinde Prophet modeli eğitilmiş ve test seti için tahminler yapılmıştır.

            - **Hata Hesaplama:** Tahmin sonuçları, gerçek değerlerle karşılaştırılarak **Ortalama Mutlak Hata (Mean Absolute Error, MAE)** hesaplanmıştır. MAE, modelin ne kadar iyi çalıştığını ölçmek için kullanılır.
            """
        )

    if 'df' in st.session_state:
        df = st.session_state.df


    else:
        st.warning("Ana sayfada veri üretilmedi. Lütfen önce ana sayfayı ziyaret edin.")

    forecast, mae, explanation, predict_table = prophet_forecast_with_mae(df)

    # Sonuçları Streamlit'te göster
    st.subheader("Model Sonuçları")
    st.write(explanation)

    st.subheader("7 Günlük Tahmin Tablosu")
    st.write(predict_table)


# Fonksiyonlar
def assign_patients(unscheduled_patients, doctors):
    """
    Bu fonksiyon, belirli bir hasta sayısını doktora atamak için kullanılır.
    Her doktorun başlangıçta sahip olduğu hasta sayısına göre en az hasta bulunan doktora yeni hasta atanır.
    """
    doctor_costs = [doc['initial_patients'] for doc in doctors]
    detailed_assignments = []

    patient_number = 1
    while unscheduled_patients > 0:
        min_cost_index = doctor_costs.index(min(doctor_costs))
        doctor_name = doctors[min_cost_index]['name']

        detailed_assignments.append({
            'patient_number': patient_number,
            'doctor': doctor_name
        })

        doctor_costs[min_cost_index] += 1
        unscheduled_patients -= 1
        patient_number += 1

    return detailed_assignments


def process_predict_table(predict_table):
    """
    predict_table adlı tablodan ilk günü alır, int türüne çevirir ve unscheduled_patients değişkenine atar.
    """
    unscheduled_patients = int(predict_table['yhat'].iloc[0])
    return unscheduled_patients






# 4. Sayfa: Assignment
def assignment_page():
    st.title("Assignment Problem")
    st.write("Bu sayfada hastalar doktorlara eşit yük ve sıra ile atama yapılacaktır.")
    with st.sidebar:
        st.header("Model Açıklaması")
        st.write(
            """
            **Amacı:** Doktorların yükünü dengelemek ve atamaları olabildiğince adil yapmak.
            
            **Kullanılan Yöntem:** Her yeni hasta, o an en az hastası olan doktora atanır.
            
            **Pratik Sonuç**: Hastalar adil şekilde dağıtılırken, doktorların toplam hasta yükü hesaplanır ve detaylı bir şekilde raporlanır.
            
            Bu sayfa, özellikle hasta yoğunluğunun fazla olduğu durumlarda doktor iş yükünü optimize etmek ve eşitlik sağlamak için kullanılabilir.
            """
        )
    if 'predict_table' in st.session_state:
        predict_table = st.session_state.predict_table

        # Tahmini atanacak hasta sayısını hesaplayın
        unscheduled_patients = process_predict_table(predict_table)
        st.write(f"Tahmin edilen atanacak hasta sayısı: {unscheduled_patients}")

        # Doktor sayısını kullanıcıdan alın
        doctor_count = st.number_input("Doktor Sayısını Seçin:", min_value=1, max_value=50, step=1, value=3)

        # Doktor isimlerini ve başlangıç hasta sayılarını dinamik olarak oluşturun
        doctors = [{'name': f'Doktor {i + 1}', 'initial_patients': random.randint(0, 10)} for i in range(doctor_count)]

        # Hasta atamalarını gerçekleştir
        patient_assignments = assign_patients(unscheduled_patients, doctors)

        # Atamaları bir DataFrame olarak düzenle
        assignments_df = pd.DataFrame(patient_assignments)
        st.write("Hasta Atama Tablosu:")
        st.dataframe(assignments_df)

        # Doktor bazında toplam hasta dağılımını hesaplayın
        doctor_assignment_counts = {}
        for assignment in patient_assignments:
            doctor = assignment['doctor']
            doctor_assignment_counts[doctor] = doctor_assignment_counts.get(doctor, 0) + 1

        # Mevcut ve toplam yükleri hesapla
        doctor_totals = []
        for doctor in doctors:
            name = doctor['name']
            initial_patients = doctor['initial_patients']
            assigned_patients = doctor_assignment_counts.get(name, 0)
            total_patients = initial_patients + assigned_patients
            doctor_totals.append({'Doctor': name, 'Initial Patients': initial_patients,
                                  'Assigned Patients': assigned_patients, 'Total Patients': total_patients})

        doctor_totals_df = pd.DataFrame(doctor_totals)
        st.write("Doktor Bazında Toplam Hasta Dağılımı:")
        st.dataframe(doctor_totals_df)

def contact_page():
    st.title("İletişim")
    st.header("İletişim")
    st.write("""
            **Daha Fazla Soru ve İletişim İçin**  
            Bu projeyle ilgili herhangi bir sorunuz veya geri bildiriminiz olursa benimle iletişime geçmekten çekinmeyin! Aşağıdaki platformlar üzerinden ulaşabilirsiniz:
            """)

    st.write("📧 **E-posta**: furkansukan10@gmail.com")
    st.write("🪪 **LinkedIn**: https://www.linkedin.com/in/furkansukan/")
    st.write("🔗 **Kaggle**: https://www.kaggle.com/furkansukan")
    st.write("🐙 **GitHub**: https://github.com/furkansukan")  # Buraya bağlantı ekleyebilirsiniz
    st.write("🌐 **Proje Sitesi**: [Delhi Metro Operations](#)")  # Buraya bağlantı ekleyebilirsiniz

    st.write("""
            Görüş ve önerilerinizi duymaktan mutluluk duyarım!
            """)

# Ana Sayfa ve Diğer Sayfaların Seçimi
def main():
    st.sidebar.title("Pusula Yazılım")
    image = Image.open('hospital_management/logo.png')
    with st.sidebar:
        st.image(image, caption='Pusulayazilim.com', use_column_width=True)
        st.write("Hastane Yönetim Sistemi")  # Fotoğrafın üstüne isim ekleyin
        st.write("Türkiye'nin en büyük özel hastanelerine çözüm ürettiniz. Sıra bende...")

    page = st.sidebar.radio("Sayfa Seçin", ["Ana Sayfa", "Analiz", "Model", "Assignment", "İletişim"])

    if page == "Ana Sayfa":
        main_page()
    elif page == "Analiz":
        analysis_page()
    elif page == "Model":
        model_page()
    elif page == "Assignment":
        assignment_page()
    elif page == "İletişim":
        contact_page()

if __name__ == "__main__":
    main()
