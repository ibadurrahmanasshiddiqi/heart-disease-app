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
    heart_data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/heart_disease.csv')
    heart_data = heart_data.dropna()
    
    X = heart_data.drop('target', axis=1)
    y = heart_data['target']
    
    pca = PCA(n_components=9)
    X_reduced = pca.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return (model, scaler, pca, X, y, X_test, y_test, y_pred, 
            accuracy, precision, recall, f1, cv_mean, cv_std, 
            class_report, heart_data)

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
    ["🏠 Beranda", "🔬 Prediksi Penyakit"]
)

# Menu: Prediksi Penyakit
if menu == "🔬 Prediksi Penyakit":
    st.title("🔬 Sistem Prediksi Penyakit Jantung")
    st.markdown(f"""
    Masukkan parameter medis pasien di sidebar untuk mendapatkan prediksi risiko penyakit jantung.
    
    **Model:** Random Forest Classifier | **Akurasi:** {accuracy:.2%} | **F1-Score:** {f1:.2%}
    """)
    
    # Sidebar for user input
    st.sidebar.header("📋 Data Pasien")
    st.sidebar.markdown("Masukkan parameter medis pasien:")
    
    age = st.sidebar.slider("Usia", 20, 100, 50)
    
    sex = st.sidebar.selectbox("Jenis Kelamin", options=[0, 1], 
                               format_func=lambda x: "Wanita" if x == 0 else "Pria")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tipe Nyeri Dada:**")
    cp = st.sidebar.selectbox("Pilih tipe nyeri dada", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Angina Tipikal", "Angina Atipikal", 
                                                     "Nyeri Non-anginal", "Asimtomatik"][x])
    
    chest_pain_info = {
        0: "**Angina Tipikal:** Nyeri dada khas jantung, terasa seperti ditekan/diremas, muncul saat aktivitas, hilang saat istirahat.",
        1: "**Angina Atipikal:** Nyeri dada tidak khas, gejalanya mirip tapi tidak semua kriteria angina tipikal terpenuhi.",
        2: "**Nyeri Non-anginal:** Nyeri dada bukan dari jantung, bisa dari otot, tulang rusuk, atau lambung.",
        3: "**Asimtomatik:** Tidak ada nyeri dada sama sekali."
    }
    st.sidebar.info(chest_pain_info[cp])
    
    st.sidebar.markdown("---")
    trestbps = st.sidebar.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
    st.sidebar.caption("💡 Normal: 90-120 mmHg | Tinggi: ≥140 mmHg")
    
    chol = st.sidebar.slider("Kolesterol Serum (mg/dL)", 100, 400, 200)
    st.sidebar.caption("💡 Normal: <200 mg/dL | Tinggi: ≥240 mg/dL")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Gula Darah Puasa:**")
    fbs = st.sidebar.selectbox("Apakah gula darah puasa > 120 mg/dl?", options=[0, 1], 
                               format_func=lambda x: "Tidak" if x == 0 else "Ya")
    st.sidebar.caption("💡 Normal: <100 mg/dL | Diabetes: ≥126 mg/dL")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Hasil EKG Istirahat:**")
    restecg = st.sidebar.selectbox("Pilih hasil EKG", options=[0, 1, 2],
                                   format_func=lambda x: ["Normal", "Kelainan Gelombang ST-T", 
                                                          "Hipertrofi Ventrikel Kiri"][x])
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
    st.sidebar.markdown("**Angina saat Olahraga:**")
    exang = st.sidebar.selectbox("Apakah mengalami nyeri dada saat olahraga?", options=[0, 1],
                                 format_func=lambda x: "Tidak" if x == 0 else "Ya")
    st.sidebar.caption("💡 Nyeri dada saat aktivitas fisik bisa indikasi masalah jantung")
    
    st.sidebar.markdown("---")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
    st.sidebar.caption("💡 Penurunan segmen ST pada EKG saat olahraga. Semakin tinggi = semakin berisiko")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Slope Segmen ST:**")
    slope = st.sidebar.selectbox("Pilih kemiringan segmen ST", options=[0, 1, 2],
                                 format_func=lambda x: ["Naik", "Datar", "Turun"][x])
    slope_info = {
        0: "**Naik (Upsloping):** Segmen ST naik setelah olahraga - biasanya normal/aman.",
        1: "**Datar (Flat):** Segmen ST datar - bisa indikasi penyakit jantung.",
        2: "**Turun (Downsloping):** Segmen ST turun - risiko tinggi penyakit jantung koroner."
    }
    st.sidebar.info(slope_info[slope])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Jumlah Pembuluh Darah Mayor:**")
    ca = st.sidebar.selectbox("Berapa pembuluh mayor yang tersumbat? (0-4)", options=[0, 1, 2, 3, 4])
    st.sidebar.caption("💡 0 = Tidak ada sumbatan | 1-4 = Jumlah pembuluh yang tersumbat")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Thalassemia:**")
    thal = st.sidebar.selectbox("Pilih status thalassemia", options=[0, 1, 2, 3],
                                format_func=lambda x: ["Normal", "Cacat Tetap", 
                                                       "Cacat Reversibel", "Tidak Diketahui"][x])
    thal_info = {
        0: "**Normal:** Tidak ada kelainan darah thalassemia.",
        1: "**Cacat Tetap (Fixed Defect):** Kelainan permanen pada sel darah merah.",
        2: "**Cacat Reversibel (Reversible Defect):** Kelainan sementara yang bisa membaik.",
        3: "**Tidak Diketahui:** Status thalassemia belum diketahui atau tidak terdeteksi."
    }
    st.sidebar.info(thal_info[thal])
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔍 Prediksi", use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performa Model")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{accuracy:.2%}")
        col_m2.metric("Precision", f"{precision:.2%}")
        col_m3.metric("Recall", f"{recall:.2%}")
        col_m4.metric("F1-Score", f"{f1:.2%}")
        
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=['Tidak Sakit', 'Sakit'],
                           y=['Tidak Sakit', 'Sakit'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title="Confusion Matrix - Random Forest")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.info(f"**Cross-Validation Score:** {cv_mean:.2%} (±{cv_std:.2%})")
    
    with col2:
        st.subheader("📈 Ringkasan Dataset")
        
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
    
    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=X.columns)
        
        input_reduced = pca.transform(input_data)
        input_scaled = scaler.transform(input_reduced)
        
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.error("⚠️ RISIKO TINGGI: Terdeteksi Penyakit Jantung")
                st.markdown(f"**Tingkat Kepercayaan:** {prediction_proba[1]:.1%}")
                st.markdown("""
                ### 📋 Rekomendasi:
                - 🏥 **Segera konsultasi ke dokter spesialis jantung**
                - 💊 Ikuti pengobatan yang diresepkan
                - 🥗 Terapkan pola hidup sehat
                - 📊 Monitoring rutin diperlukan
                """)
            else:
                st.success("✅ RISIKO RENDAH: Tidak Terdeteksi Penyakit Jantung")
                st.markdown(f"**Tingkat Kepercayaan:** {prediction_proba[0]:.1%}")
                st.markdown("""
                ### 📋 Rekomendasi:
                - 💚 Lanjutkan gaya hidup sehat
                - 🏃 Olahraga teratur (150 menit/minggu)
                - 🥗 Konsumsi makanan bergizi seimbang
                - 🩺 Cek kesehatan rutin setiap 6-12 bulan
                """)

else:  # Beranda
    st.title("❤️ Sistem Prediksi Penyakit Jantung")
    
    st.markdown(f"""
    ## Selamat Datang! 👋
    
    Aplikasi ini menggunakan **Random Forest Classifier** untuk memprediksi risiko penyakit jantung.
    
    ### 🎯 Performa Model
    - **Accuracy:** {accuracy:.2%}
    - **F1-Score:** {f1:.2%}
    - **Cross-Validation:** {cv_mean:.2%} (±{cv_std:.2%})
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"### 🎯 Akurat\nRandom Forest dengan akurasi **{accuracy:.1%}**")
    
    with col2:
        st.success("### 🚀 Cepat\nHasil prediksi dalam hitungan detik")
    
    with col3:
        st.warning(f"### 📊 Reliable\nCV Score: **{cv_mean:.1%}**")
    
    st.markdown("---")
    st.error("""
    ### ⚠️ DISCLAIMER PENTING
    
    Aplikasi ini **HANYA untuk tujuan edukasi**. Hasil prediksi **TIDAK dapat menggantikan** diagnosis medis profesional.
    
    **Selalu konsultasikan dengan dokter** untuk diagnosis dan pengobatan yang akurat.
    """)

st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>⚕️ Aplikasi ini untuk tujuan edukasi. Konsultasikan dengan tenaga medis profesional.</p>
    <p>© 2024 Sistem Prediksi Penyakit Jantung | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p><strong>Model: Random Forest</strong> | Accuracy: {accuracy:.2%} | F1-Score: {f1:.2%}</p>
</div>
""", unsafe_allow_html=True)