import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score)
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    # Load dataset
    heart_data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/heart_disease.csv')
    
    # Handle missing values
    heart_data = heart_data.dropna()
    
    # Separate features and target
    X = heart_data.drop('target', axis=1)
    y = heart_data['target']
    
    # PCA reduction
    pca = PCA(n_components=9)
    X_reduced = pca.fit_transform(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return (model, scaler, pca, X, y, X_test, y_test, y_pred, 
            accuracy, precision, recall, f1, cv_mean, cv_std, 
            class_report, heart_data)

# Load data and model
try:
    (model, scaler, pca, X, y, X_test, y_test, y_pred, 
     accuracy, precision, recall, f1, cv_mean, cv_std, 
     class_report, heart_data) = load_and_prepare_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Menu
st.sidebar.title("🏥 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚠️ Faktor Risiko", "🔬 Prediksi Penyakit"]
)

# Menu: Tentang
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    
    st.markdown(f"""
    ## Sistem Prediksi Penyakit Jantung
    
    ### Versi 1.0
    
    **Dikembangkan oleh:** Tim Machine Learning
    
    **Teknologi yang Digunakan:**
    - Python 3.x
    - Streamlit (Framework Web)
    - Scikit-learn (Machine Learning)
    - Plotly (Visualisasi Data)
    - Pandas & NumPy (Pengolahan Data)
    
    ### Tentang Dataset
    Dataset yang digunakan adalah Heart Disease Dataset yang berisi data medis dari pasien dengan berbagai parameter kesehatan jantung.
    
    **Jumlah Data:** {len(heart_data)} sampel pasien
    
    **Jumlah Fitur:** 13 parameter medis
    
    **Target:** Prediksi ada/tidaknya penyakit jantung
    
    ### Model Machine Learning
    - **Algoritma:** Random Forest Classifier
    - **Akurasi Model:** {accuracy:.2%}
    - **Precision:** {precision:.2%}
    - **Recall:** {recall:.2%}
    - **F1-Score:** {f1:.2%}
    - **Cross-Validation:** {cv_mean:.2%} (±{cv_std:.2%})
    - **Preprocessing:** PCA (9 komponen) + Standardisasi
    
    ### Keunggulan Random Forest
    - ✅ **Akurasi Tinggi:** Ensemble learning meningkatkan performa
    - ✅ **Robust:** Tahan terhadap overfitting
    - ✅ **Stabil:** Prediksi lebih konsisten
    - ✅ **Handle Noise:** Dapat menangani data noisy dengan baik
    
    ### Disclaimer
    ⚠️ **Penting:** Aplikasi ini hanya untuk tujuan edukasi dan referensi. Hasil prediksi **TIDAK** dapat menggantikan diagnosis medis profesional. Selalu konsultasikan dengan dokter atau tenaga medis yang berkualifikasi untuk diagnosis dan pengobatan yang akurat.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Gunakan menu sidebar untuk navigasi ke berbagai fitur aplikasi.")

# Menu: Pengenalan Aplikasi
elif menu == "📚 Pengenalan Aplikasi":
    st.title("📚 Pengenalan Aplikasi")
    
    st.markdown("""
    ## Apa itu Sistem Prediksi Penyakit Jantung?
    
    Sistem Prediksi Penyakit Jantung adalah aplikasi berbasis **Machine Learning** yang dirancang untuk membantu memprediksi kemungkinan seseorang memiliki penyakit jantung berdasarkan parameter medis tertentu.
    
    ### 🎯 Tujuan Aplikasi
    
    1. **Deteksi Dini:** Membantu mendeteksi potensi penyakit jantung lebih awal
    2. **Edukasi:** Memberikan pemahaman tentang faktor-faktor risiko penyakit jantung
    3. **Kesadaran Kesehatan:** Meningkatkan awareness tentang pentingnya menjaga kesehatan jantung
    4. **Referensi Medis:** Memberikan informasi tambahan untuk konsultasi dengan dokter
    
    ### 🔬 Cara Kerja Aplikasi
    
    1. **Input Data:** Pengguna memasukkan 13 parameter medis di sidebar
    2. **Preprocessing:** Data dinormalisasi dan ditransformasi menggunakan PCA
    3. **Prediksi:** Model Random Forest menganalisis data dan memberikan prediksi
    4. **Hasil:** Aplikasi menampilkan hasil prediksi beserta tingkat kepercayaan dan rekomendasi
    
    ### 📊 Parameter yang Dianalisis
    
    Aplikasi menganalisis **13 parameter medis** yang meliputi:
    - Data demografis (usia, jenis kelamin)
    - Tekanan darah dan kolesterol
    - Hasil EKG
    - Detak jantung maksimum
    - Dan parameter medis lainnya
    
    ### 🤖 Tentang Random Forest
    
    **Random Forest** adalah algoritma ensemble learning yang:
    - Menggunakan banyak decision trees (100 trees)
    - Menggabungkan prediksi dari semua trees
    - Memberikan hasil yang lebih akurat dan stabil
    - Mengurangi risiko overfitting
    
    ### ✨ Fitur Utama
    
    - 🎨 **Interface User-Friendly:** Mudah digunakan dengan tampilan intuitif
    - 📊 **Visualisasi Interaktif:** Grafik dan chart yang informatif
    - 🔍 **Analisis Real-time:** Hasil prediksi langsung setelah input data
    - 📈 **Metrics Lengkap:** Accuracy, Precision, Recall, F1-Score
    - 🎯 **Cross-Validation:** Evaluasi model yang lebih reliable
    - 💾 **Berbasis Web:** Dapat diakses dari browser tanpa instalasi
    
    ### 🚀 Cara Menggunakan
    
    1. Pilih menu **"🔬 Prediksi Penyakit"** di sidebar
    2. Masukkan semua parameter medis pasien
    3. Klik tombol **"🔍 Prediksi"**
    4. Lihat hasil prediksi dan rekomendasi
    
    """)
    
    st.success("✅ Siap menggunakan aplikasi? Pilih menu **'🔬 Prediksi Penyakit'** untuk mulai!")

# Menu: Faktor Risiko
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Faktor Risiko Penyakit Jantung")
    
    st.markdown("""
    ## Memahami Faktor Risiko Penyakit Jantung
    
    Penyakit jantung adalah salah satu penyebab kematian tertinggi di dunia. Memahami faktor risikonya adalah langkah pertama untuk pencegahan.
    """)
    
    # Faktor Risiko yang Tidak Dapat Diubah
    st.subheader("🔒 Faktor Risiko yang TIDAK Dapat Diubah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. 👤 Usia
        - Risiko meningkat seiring bertambahnya usia
        - Pria ≥45 tahun, Wanita ≥55 tahun memiliki risiko lebih tinggi
        - Pembuluh darah menjadi kurang elastis seiring waktu
        
        ### 2. 👨👩 Jenis Kelamin
        - Pria memiliki risiko lebih tinggi di usia muda
        - Wanita: risiko meningkat setelah menopause
        - Hormon estrogen memberikan perlindungan pada wanita
        """)
    
    with col2:
        st.markdown("""
        ### 3. 🧬 Riwayat Keluarga
        - Genetik berperan penting
        - Risiko 2-3x lebih tinggi jika ada riwayat keluarga
        - Terutama jika anggota keluarga terdiagnosis di usia muda
        
        ### 4. 🩸 Ras/Etnis
        - Beberapa kelompok etnis memiliki risiko lebih tinggi
        - Faktor genetik dan sosial-ekonomi berperan
        """)
    
    st.markdown("---")
    
    # Faktor Risiko yang Dapat Diubah
    st.subheader("✅ Faktor Risiko yang DAPAT Diubah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. 🩺 Tekanan Darah Tinggi (Hipertensi)
        - **Normal:** <120/80 mmHg
        - **Tinggi:** ≥140/90 mmHg
        - Merusak arteri dan jantung
        - **Pencegahan:** Diet rendah garam, olahraga, obat
        
        ### 2. 🧈 Kolesterol Tinggi
        - **LDL (jahat):** Harus <100 mg/dL
        - **HDL (baik):** Harus >40 mg/dL (pria), >50 mg/dL (wanita)
        - Menyebabkan plak di arteri
        - **Pencegahan:** Diet sehat, olahraga, statin
        
        ### 3. 🚬 Merokok
        - Meningkatkan risiko 2-4x lipat
        - Merusak pembuluh darah
        - Menurunkan oksigen dalam darah
        - **Pencegahan:** BERHENTI merokok sekarang!
        
        ### 4. 🍔 Obesitas
        - **BMI Normal:** 18.5-24.9
        - **Obesitas:** BMI ≥30
        - Meningkatkan tekanan darah dan kolesterol
        - **Pencegahan:** Diet seimbang, olahraga rutin
        """)
    
    with col2:
        st.markdown("""
        ### 5. 💉 Diabetes
        - Merusak pembuluh darah dan saraf
        - Risiko 2-4x lebih tinggi
        - **Kontrol:** Gula darah <126 mg/dL (puasa)
        - **Pencegahan:** Diet, olahraga, obat
        
        ### 6. 🏃 Kurang Aktivitas Fisik
        - Gaya hidup sedenter = risiko tinggi
        - **Rekomendasi:** 150 menit/minggu aktivitas sedang
        - Olahraga memperkuat jantung
        - **Pencegahan:** Jalan kaki, jogging, berenang
        
        ### 7. 😰 Stress
        - Stress kronis merusak jantung
        - Meningkatkan tekanan darah
        - **Pencegahan:** Meditasi, yoga, hobi
        
        ### 8. 🍺 Alkohol Berlebihan
        - Meningkatkan tekanan darah
        - Merusak otot jantung
        - **Batasan:** Maksimal 1-2 gelas/hari
        - **Pencegahan:** Kurangi atau hindari alkohol
        """)
    
    st.markdown("---")
    
    # Parameter Medis dalam Aplikasi
    st.subheader("🔬 Parameter Medis dalam Aplikasi Ini")
    
    st.markdown("""
    Aplikasi ini menganalisis 13 parameter medis berikut:
    
    | Parameter | Deskripsi | Nilai Normal |
    |-----------|-----------|--------------|
    | **Usia** | Usia pasien dalam tahun | 20-100 tahun |
    | **Jenis Kelamin** | 0 = Wanita, 1 = Pria | - |
    | **Tipe Nyeri Dada** | 0-3 (dari ringan hingga parah) | 0 = Aman |
    | **Tekanan Darah** | Tekanan darah saat istirahat | 90-120 mmHg |
    | **Kolesterol** | Kolesterol serum | <200 mg/dL |
    | **Gula Darah Puasa** | >120 mg/dL = 1 | <100 mg/dL |
    | **EKG Istirahat** | Hasil elektrokardiogram | 0 = Normal |
    | **Detak Jantung Max** | Detak jantung maksimum | 60-100 bpm |
    | **Angina saat Olahraga** | Nyeri dada saat aktivitas | 0 = Tidak |
    | **ST Depression** | Depresi ST saat olahraga | 0-2 |
    | **Slope** | Kemiringan segmen ST | 0-2 |
    | **Jumlah Pembuluh** | Pembuluh darah mayor (0-3) | 0 = Aman |
    | **Thalassemia** | Kelainan darah | 1 = Normal |
    """)
    
    st.markdown("---")
    
    st.info("""
    ### 💡 Tips Pencegahan Penyakit Jantung
    
    1. ✅ **Makan Sehat:** Konsumsi buah, sayur, biji-bijian, ikan
    2. ✅ **Olahraga Rutin:** Minimal 30 menit/hari, 5 hari/minggu
    3. ✅ **Jaga Berat Badan:** Pertahankan BMI normal (18.5-24.9)
    4. ✅ **Stop Merokok:** Hentikan segera untuk mengurangi risiko
    5. ✅ **Kurangi Stress:** Meditasi, yoga, atau aktivitas menyenangkan
    6. ✅ **Cek Kesehatan Rutin:** Periksa tekanan darah dan kolesterol
    7. ✅ **Tidur Cukup:** 7-9 jam per malam
    8. ✅ **Batasi Alkohol:** Maksimal 1-2 gelas/hari
    """)

# Menu: Prediksi Penyakit
elif menu == "🔬 Prediksi Penyakit":
    st.title("🔬 Sistem Prediksi Penyakit Jantung")
    st.markdown(f"""
    Masukkan parameter medis pasien di sidebar untuk mendapatkan prediksi risiko penyakit jantung.
    
    **Model:** Random Forest Classifier | **Akurasi:** {accuracy:.2%} | **F1-Score:** {f1:.2%}
    """)
    
    # Sidebar for user input
    st.sidebar.header("📋 Data Pasien")
    st.sidebar.markdown("Masukkan parameter medis pasien:")
    
    # Create input fields
    age = st.sidebar.slider("Usia", 20, 100, 50)
    sex = st.sidebar.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Wanita" if x == 0 else "Pria")
    cp = st.sidebar.selectbox("Tipe Nyeri Dada", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Angina Tipikal", "Angina Atipikal", "Nyeri Non-anginal", "Asimtomatik"][x])
     # Penjelasan tipe nyeri dada
    chest_pain_info = {
        0: "**Angina Tipikal:** Nyeri dada khas jantung, terasa seperti ditekan/diremas, muncul saat aktivitas, hilang saat istirahat.",
        1: "**Angina Atipikal:** Nyeri dada tidak khas, gejalanya mirip tapi tidak semua kriteria angina tipikal terpenuhi.",
        2: "**Nyeri Non-anginal:** Nyeri dada bukan dari jantung, bisa dari otot, tulang rusuk, atau lambung.",
        3: "**Asimtomatik:** Tidak ada nyeri dada sama sekali."
    }
    st.sidebar.info(chest_pain_info[cp])
    trestbps = st.sidebar.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
    chol = st.sidebar.slider("Kolesterol Serum (mg/dL)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dl", options=[0, 1], 
                               format_func=lambda x: "Tidak" if x == 0 else "Ya")
    restecg = st.sidebar.selectbox("Hasil EKG Istirahat", options=[0, 1, 2],
                                   format_func=lambda x: ["Normal", "Kelainan Gelombang ST-T", "Hipertrofi Ventrikel Kiri"][x])
    ecg_info = {
        0: "**Normal:** Tidak ada kelainan pada elektrokardiogram.",
        1: "**Kelainan ST-T:** Ada kelainan gelombang ST atau T, bisa indikasi iskemia (kurang oksigen ke jantung).",
        2: "**Hipertrofi Ventrikel Kiri:** Penebalan otot jantung kiri, sering akibat tekanan darah tinggi."
    }
    st.sidebar.info(ecg_info[restecg])
    st.sidebar.markdown("---")
    thalach = st.sidebar.slider("Detak Jantung Maksimum", 60, 220, 150)
    st.sidebar.caption("💡 Normal istirahat: 60-100 bpm | Saat olahraga: 120-200 bpm")
    st.sidebar.markdown("---")
    exang = st.sidebar.selectbox("Angina saat Olahraga", options=[0, 1],
                                 format_func=lambda x: "Tidak" if x == 0 else "Ya")
    st.sidebar.caption("💡 Nyeri dada saat aktivitas fisik bisa indikasi masalah jantung")
    
    st.sidebar.markdown("---")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
    st.sidebar.caption("💡 Penurunan segmen ST pada EKG saat olahraga. Semakin tinggi = semakin berisiko")
    st.sidebar.markdown("---")
    slope = st.sidebar.selectbox("Slope Segmen ST", options=[0, 1, 2],
                                 format_func=lambda x: ["Naik", "Datar", "Turun"][x])
    slope_info = {
        0: "**Naik (Upsloping):** Segmen ST naik setelah olahraga - biasanya normal/aman.",
        1: "**Datar (Flat):** Segmen ST datar - bisa indikasi penyakit jantung.",
        2: "**Turun (Downsloping):** Segmen ST turun - risiko tinggi penyakit jantung koroner."
    }
    st.sidebar.info(slope_info[slope])
    
    st.sidebar.markdown("---")
    ca = st.sidebar.selectbox("Jumlah Pembuluh Darah Mayor (0-3)", options=[0, 1, 2, 3, 4])
    st.sidebar.caption("💡 0 = Tidak ada sumbatan | 1-4 = Jumlah pembuluh yang tersumbat")
    
    st.sidebar.markdown("---")
    thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])
     thal_info = {
        0: "**Normal:** Tidak ada kelainan darah thalassemia.",
        1: "**Cacat Tetap (Fixed Defect):** Kelainan permanen pada sel darah merah.",
        2: "**Cacat Reversibel (Reversible Defect):** Kelainan sementara yang bisa membaik.",
        3: "**Tidak Diketahui:** Status thalassemia belum diketahui atau tidak terdeteksi."
    }
    st.sidebar.info(thal_info[thal])
    
    st.sidebar.markdown("---")
    
    # Create prediction button
    predict_button = st.sidebar.button("🔍 Prediksi", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performa Model")
        
        # Display metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{accuracy:.2%}")
        col_m2.metric("Precision", f"{precision:.2%}")
        col_m3.metric("Recall", f"{recall:.2%}")
        col_m4.metric("F1-Score", f"{f1:.2%}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=['Tidak Sakit', 'Sakit'],
                           y=['Tidak Sakit', 'Sakit'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title="Confusion Matrix - Random Forest")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Cross-Validation Info
        st.info(f"**Cross-Validation Score:** {cv_mean:.2%} (±{cv_std:.2%})")
    
    with col2:
        st.subheader("📈 Ringkasan Dataset")
        
        # Target distribution
        target_counts = heart_data['target'].value_counts()
        fig_dist = go.Figure(data=[go.Pie(
            labels=['Tidak Sakit', 'Sakit'],
            values=target_counts.values,
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#e74c3c'])
        )])
        fig_dist.update_layout(title="Distribusi Target")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.metric("Total Sampel", len(heart_data))
        st.metric("Jumlah Fitur", len(heart_data.columns) - 1)
        st.metric("Algoritma", "Random Forest")
    
    # Prediction section
    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        
        # Prepare input features
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=X.columns)
        
        # Transform input
        input_reduced = pca.transform(input_data)
        input_scaled = scaler.transform(input_reduced)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display result
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.error("⚠️ RISIKO TINGGI: Terdeteksi Penyakit Jantung")
                st.markdown(f"**Tingkat Kepercayaan:** {prediction_proba[1]:.1%}")
                st.markdown("**Model:** Random Forest Classifier")
                st.markdown("""
                ### 📋 Rekomendasi:
                - 🏥 **Segera konsultasi ke dokter spesialis jantung**
                - 💊 Ikuti pengobatan yang diresepkan
                - 🥗 Terapkan pola hidup sehat
                - 📊 Monitoring rutin diperlukan
                - 🚨 Hindari aktivitas berat tanpa pengawasan medis
                """)
            else:
                st.success("✅ RISIKO RENDAH: Tidak Terdeteksi Penyakit Jantung")
                st.markdown(f"**Tingkat Kepercayaan:** {prediction_proba[0]:.1%}")
                st.markdown("**Model:** Random Forest Classifier")
                st.markdown("""
                ### 📋 Rekomendasi:
                - 💚 Lanjutkan gaya hidup sehat
                - 🏃 Olahraga teratur (150 menit/minggu)
                - 🥗 Konsumsi makanan bergizi seimbang
                - 🩺 Cek kesehatan rutin setiap 6-12 bulan
                - 🚭 Hindari merokok dan alkohol berlebihan
                """)
        
        # Probability chart
        st.markdown("---")
        st.subheader("📊 Probabilitas Prediksi")
        
        fig_proba = go.Figure(data=[
            go.Bar(x=['Tidak Sakit', 'Sakit'], 
                   y=prediction_proba,
                   marker=dict(color=['#2ecc71', '#e74c3c']),
                   text=[f'{p:.1%}' for p in prediction_proba],
                   textposition='auto')
        ])
        fig_proba.update_layout(
            yaxis_title="Probabilitas",
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        st.plotly_chart(fig_proba, use_container_width=True)
        
        # Display input summary
        st.markdown("---")
        st.subheader("📝 Ringkasan Data Input")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            - **Usia:** {age} tahun
            - **Jenis Kelamin:** {"Pria" if sex == 1 else "Wanita"}
            - **Tekanan Darah:** {trestbps} mmHg
            - **Kolesterol:** {chol} mg/dL
            - **Gula Darah Puasa:** {"Ya" if fbs == 1 else "Tidak"}
            """)
        
        with col2:
            st.markdown(f"""
            - **Detak Jantung Max:** {thalach} bpm
            - **Angina saat Olahraga:** {"Ya" if exang == 1 else "Tidak"}
            - **ST Depression:** {oldpeak}
            - **Jumlah Pembuluh:** {ca}
            """)
        
        with col3:
            tipe_cp = ["Angina Tipikal", "Angina Atipikal", "Nyeri Non-anginal", "Asimtomatik"][cp]
            tipe_restecg = ["Normal", "Kelainan ST-T", "Hipertrofi LV"][restecg]
            tipe_slope = ["Naik", "Datar", "Turun"][slope]
            tipe_thal = ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][thal]
            
            st.markdown(f"""
            - **Tipe Nyeri Dada:** {tipe_cp}
            - **Hasil EKG:** {tipe_restecg}
            - **Slope ST:** {tipe_slope}
            - **Thalassemia:** {tipe_thal}
            """)

# Menu: Beranda (Default)
else:
    st.title("❤️ Sistem Prediksi Penyakit Jantung")
    
    st.markdown(f"""
    ## Selamat Datang! 👋
    
    Aplikasi ini menggunakan **Random Forest Classifier** untuk memprediksi risiko penyakit jantung berdasarkan parameter medis.
    
    ### 🎯 Performa Model
    - **Accuracy:** {accuracy:.2%}
    - **F1-Score:** {f1:.2%}
    - **Cross-Validation:** {cv_mean:.2%} (±{cv_std:.2%})
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        ### 🎯 Akurat
        Random Forest dengan akurasi **{accuracy:.1%}**
        """)
    
    with col2:
        st.success("""
        ### 🚀 Cepat
        Hasil prediksi dalam hitungan detik
        """)
    
    with col3:
        st.warning(f"""
        ### 📊 Reliable
        CV Score: **{cv_mean:.1%}**
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Panduan Cepat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Untuk Pengguna Baru:
        
        1. **Baca Pengenalan** - Pahami cara kerja aplikasi
        2. **Pelajari Faktor Risiko** - Ketahui parameter yang dianalisis
        3. **Lakukan Prediksi** - Input data dan lihat hasilnya
        
        👉 Mulai dari menu **"📚 Pengenalan Aplikasi"**
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Untuk Pengguna Berpengalaman:
        
        1. Pilih menu **"🔬 Prediksi Penyakit"**
        2. Masukkan 13 parameter medis di sidebar
        3. Klik tombol **"Prediksi"**
        4. Lihat hasil dan rekomendasi
        
        👉 Langsung ke menu **"🔬 Prediksi Penyakit"**
        """)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistik Model")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    
    with col2:
        st.metric("Precision", f"{precision:.1%}")
    
    with col3:
        st.metric("Recall", f"{recall:.1%}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.1%}")
    
    with col5:
        st.metric("CV Score", f"{cv_mean:.1%}")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📈 Visualisasi Performa Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics bar chart
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [accuracy, precision, recall, f1]
        }
        
        fig_metrics = go.Figure(data=[
            go.Bar(
                x=metrics_data['Metric'],
                y=metrics_data['Score'],
                marker=dict(color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']),
                text=[f'{s:.2%}' for s in metrics_data['Score']],
                textposition='auto'
            )
        ])
        fig_metrics.update_layout(
            title="Performance Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # Target distribution
        target_counts = heart_data['target'].value_counts()
        fig_dist = go.Figure(data=[go.Pie(
            labels=['Tidak Sakit', 'Sakit'],
            values=target_counts.values,
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textinfo='label+percent',
            textfont_size=14
        )])
        fig_dist.update_layout(
            title="Distribusi Data Training",
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Important notice
    st.error("""
    ### ⚠️ DISCLAIMER PENTING
    
    Aplikasi ini **HANYA untuk tujuan edukasi dan referensi**. Hasil prediksi **TIDAK dapat menggantikan** diagnosis medis profesional dari dokter yang berkualifikasi.
    
    **Selalu konsultasikan dengan dokter atau tenaga medis profesional** untuk diagnosis, pengobatan, dan saran medis yang akurat.
    """)
    
    st.markdown("---")
    
    # Call to action
    st.success("""
    ### 💡 Siap Memulai?
    
    Pilih salah satu menu di sidebar untuk memulai:
    - **ℹ️ Tentang** - Informasi aplikasi dan teknologi
    - **📚 Pengenalan Aplikasi** - Cara kerja dan fitur
    - **⚠️ Faktor Risiko** - Pelajari faktor risiko penyakit jantung
    - **🔬 Prediksi Penyakit** - Mulai prediksi sekarang!
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>⚕️ Aplikasi ini untuk tujuan edukasi. Konsultasikan dengan tenaga medis profesional untuk diagnosis yang akurat.</p>
    <p>© 2024 Sistem Prediksi Penyakit Jantung v2.0 | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p><strong>Model: Random Forest Classifier</strong> | Accuracy: {accuracy:.2%} | F1-Score: {f1:.2%}</p>
</div>
""", unsafe_allow_html=True)