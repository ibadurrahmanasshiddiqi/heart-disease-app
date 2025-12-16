import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# Load and prepare data with model comparison
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
    
    # Train multiple models
    models = {
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
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
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
    
    return (results, scaler, pca, X, y, X_test, y_test, 
            heart_data, best_model)

# Load data and models
try:
    results, scaler, pca, X, y, X_test, y_test, heart_data, best_model_name = load_and_prepare_data()
    best_model = results[best_model_name]['model']
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Menu
st.sidebar.title("🏥 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚠️ Faktor Risiko", 
     "🔬 Prediksi Penyakit", "📊 Perbandingan Model"]
)

# Menu: Tentang
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    
    st.markdown(f"""
    ## Sistem Prediksi Penyakit Jantung
    
    ### Versi 2.0 (Improved)
    
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
    - **Model Terbaik:** {best_model_name}
    - **Akurasi:** {results[best_model_name]['accuracy']:.2%}
    - **F1-Score:** {results[best_model_name]['f1_score']:.2%}
    - **Cross-Validation:** {results[best_model_name]['cv_mean']:.2%} (±{results[best_model_name]['cv_std']:.2%})
    - **Preprocessing:** PCA (9 komponen) + Standardisasi
    
    ### Peningkatan Versi 2.0
    - ✅ Perbandingan Decision Tree vs Random Forest
    - ✅ Cross-Validation untuk evaluasi robust
    - ✅ Metrics lengkap (Precision, Recall, F1-Score)
    - ✅ Pemilihan model terbaik otomatis
    - ✅ Visualisasi perbandingan model
    
    ### Disclaimer
    ⚠️ **Penting:** Aplikasi ini hanya untuk tujuan edukasi dan referensi. Hasil prediksi **TIDAK** dapat menggantikan diagnosis medis profesional. Selalu konsultasikan dengan dokter atau tenaga medis yang berkualifikasi untuk diagnosis dan pengobatan yang akurat.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Lihat menu 'Perbandingan Model' untuk melihat performa semua model!")

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
    3. **Model Selection:** Sistem menggunakan model terbaik (dipilih otomatis)
    4. **Prediksi:** Model menganalisis data dan memberikan prediksi
    5. **Hasil:** Aplikasi menampilkan hasil prediksi beserta tingkat kepercayaan dan rekomendasi
    
    ### 🤖 Model Machine Learning
    
    Aplikasi ini menggunakan **2 algoritma** dan memilih yang terbaik:
    
    1. **Decision Tree Classifier**
       - Model berbasis pohon keputusan
       - Mudah diinterpretasi
       - Cepat dalam training dan prediksi
    
    2. **Random Forest Classifier** ⭐ (Biasanya lebih akurat)
       - Ensemble dari banyak decision trees
       - Lebih robust terhadap overfitting
       - Akurasi lebih tinggi
    
    ### ✨ Fitur Utama Versi 2.0
    
    - 🎨 **Interface User-Friendly:** Mudah digunakan dengan tampilan intuitif
    - 📊 **Visualisasi Interaktif:** Grafik dan chart yang informatif
    - 🔍 **Analisis Real-time:** Hasil prediksi langsung setelah input data
    - 📈 **Dashboard Performa:** Melihat akurasi dan performa model
    - 🆚 **Perbandingan Model:** Lihat performa Decision Tree vs Random Forest
    - 📉 **Metrics Lengkap:** Accuracy, Precision, Recall, F1-Score
    - 🎯 **Cross-Validation:** Evaluasi yang lebih reliable
    - 💾 **Berbasis Web:** Dapat diakses dari browser tanpa instalasi
    
    ### 🚀 Cara Menggunakan
    
    1. Pilih menu **"🔬 Prediksi Penyakit"** di sidebar
    2. Masukkan semua parameter medis pasien
    3. Klik tombol **"🔍 Prediksi"**
    4. Lihat hasil prediksi dan rekomendasi
    5. (Opsional) Lihat **"📊 Perbandingan Model"** untuk analisis detail
    
    """)
    
    st.success("✅ Siap menggunakan aplikasi? Pilih menu **'🔬 Prediksi Penyakit'** untuk mulai!")

# Menu: Faktor Risiko (sama seperti sebelumnya, tidak diubah)
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

# Menu: Perbandingan Model (BARU!)
elif menu == "📊 Perbandingan Model":
    st.title("📊 Perbandingan Model Machine Learning")
    
    st.markdown(f"""
    Aplikasi ini menggunakan **2 algoritma berbeda** dan secara otomatis memilih model terbaik.
    
    **Model Terbaik:** ⭐ **{best_model_name}**
    """)
    
    st.markdown("---")
    
    # Comparison Table
    st.subheader("📋 Tabel Perbandingan Metrics")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'CV Mean': f"{result['cv_mean']:.4f}",
            'CV Std': f"{result['cv_std']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📈 Visualisasi Perbandingan")
    
    # Metrics comparison chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for name, result in results.items():
        fig.add_trace(go.Bar(
            name=name,
            x=metrics_names,
            y=[result['accuracy'], result['precision'], result['recall'], result['f1_score']],
            text=[f"{result['accuracy']:.2%}", f"{result['precision']:.2%}", 
                  f"{result['recall']:.2%}", f"{result['f1_score']:.2%}"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Perbandingan Metrics Model",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.subheader("🔍 Confusion Matrix Per Model")
    
    col1, col2 = st.columns(2)
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=['Tidak Sakit', 'Sakit'],
                           y=['Tidak Sakit', 'Sakit'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title=f"Confusion Matrix - {name}")
        
        if idx == 0:
            col1.plotly_chart(fig_cm, use_container_width=True)
        else:
            col2.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Classification Report
    st.subheader("📄 Classification Report Detail")
    
    for name, result in results.items():
        with st.expander(f"📊 {name} - Classification Report"):
            report_df = pd.DataFrame(result['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
    
    st.markdown("---")
    
    # Interpretation
    st.subheader("💡 Interpretasi Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Penjelasan Metrics
        
        **Accuracy** 🎯
        - Persentase prediksi yang benar
        - Formula: (TP + TN) / Total
        - Semakin tinggi semakin baik
        
        **Precision** 🔍
        - Dari yang diprediksi sakit, berapa yang benar sakit
        - Formula: TP / (TP + FP)
        - Penting untuk menghindari false alarm
        
        **Recall (Sensitivity)** 🎭
        - Dari yang benar-benar sakit, berapa yang terdeteksi
        - Formula: TP / (TP + FN)
        - Penting untuk deteksi dini
        
        **F1-Score** ⚖️
        - Harmonic mean dari Precision dan Recall
        - Balance antara keduanya
        - Baik untuk imbalanced dataset
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 Kesimpulan
        
        **Model Terbaik:** {model}
        
        **Alasan:**
        - Accuracy: {acc:.2%}
        - F1-Score: {f1:.2%}
        - Cross-Validation: {cv:.2%}
        
        **Kelebihan Random Forest:**
        - ✅ Lebih robust terhadap overfitting
        - ✅ Dapat menangani noise lebih baik
        - ✅ Ensemble learning = prediksi lebih stabil
        - ✅ Feature importance analysis
        
        **Kelebihan Decision Tree:**
        - ✅ Lebih cepat training
        - ✅ Mudah diinterpretasi
        - ✅ Tidak memerlukan banyak memori
        - ✅ Visualisasi tree yang jelas
        """.format(
            model=best_model_name,
            acc=results[best_model_name]['accuracy'],
            f1=results[best_model_name]['f1_score'],
            cv=results[best_model_name]['cv_mean']
        ))

# Menu: Prediksi Penyakit
elif menu == "🔬 Prediksi Penyakit":
    st.title("🔬 Sistem Prediksi Penyakit Jantung")
    st.markdown(f"""
    Masukkan parameter medis pasien di sidebar untuk mendapatkan prediksi risiko penyakit jantung.
    
    **Model yang digunakan:** {best_model_name} (Akurasi: {results[best_model_name]['accuracy']:.2%})
    """)
    
    # Sidebar for user input
    st.sidebar.header("📋 Data Pasien")
    st.sidebar.markdown("Masukkan parameter medis pasien:")
    
    # Create input fields
    age = st.sidebar.slider("Usia", 20, 100, 50)
    sex = st.sidebar.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Wanita" if x == 0 else "Pria")
    cp = st.sidebar.selectbox("Tipe Nyeri Dada", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Angina Tipikal", "Angina Atipikal", "Nyeri Non-anginal", "Asimtomatik"][x])
    trestbps = st.sidebar.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
    chol = st.sidebar.slider("Kolesterol Serum (mg/dL)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dl", options=[0, 1], 
                               format_func=lambda x: "Tidak" if x == 0 else "Ya")
    restecg = st.sidebar.selectbox("Hasil EKG Istirahat", options=[0, 1, 2],
                                   format_func=lambda x: ["Normal", "Kelainan Gelombang ST-T", "Hipertrofi Ventrikel Kiri"][x])
    thalach = st.sidebar.slider("Detak Jantung Maksimum", 60, 220, 150)
    exang = st.sidebar.selectbox("Angina saat Olahraga", options=[0, 1],
                                 format_func=lambda x: "Tidak" if x == 0 else "Ya")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
    slope = st.sidebar.selectbox("Slope Segmen ST", options=[0, 1, 2],
                                 format_func=lambda x: ["Naik", "Datar", "Turun"][x])
    ca = st.sidebar.selectbox("Jumlah Pembuluh Darah Mayor (0-3)", options=[0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])
    
    # Create prediction button
    predict_button = st.sidebar.button("🔍 Prediksi", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performa Model Terpilih")
        
        # Display metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{results[best_model_name]['accuracy']:.2%}")
        col_m2.metric("Precision", f"{results[best_model_name]['precision']:.2%}")
        col_m3.metric("Recall", f"{results[best_model_name]['recall']:.2%}")
        col_m4.metric("F1-Score", f"{results[best_model_name]['f1_score']:.2%}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=['Tidak Sakit', 'Sakit'],
                           y=['Tidak Sakit', 'Sakit'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title=f"Confusion Matrix - {best_model_name}")
        st.plotly_chart(fig_cm, use_container_width=True)
    
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
        st.metric("Model Terbaik", best_model_name)
    
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
        
        # Make prediction using best model
        prediction = best_model.predict(input_scaled)[0]
        prediction_proba = best_model.predict_proba(input_scaled)[0]
        
        # Display result
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.error("⚠️ RISIKO TINGGI: Terdeteksi Penyakit Jantung")
                st.markdown(f"**Tingkat Kepercayaan:** {prediction_proba[1]:.1%}")
                st.markdown(f"**Model:** {best_model_name}")
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
                st.markdown(f"**Model:** {best_model_name}")
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
    
    Aplikasi ini menggunakan **Machine Learning** untuk memprediksi risiko penyakit jantung berdasarkan parameter medis.
    
    ### 🆕 Versi 2.0 - Improved!
    - ✅ Perbandingan Decision Tree vs Random Forest
    - ✅ Model terbaik dipilih otomatis: **{best_model_name}**
    - ✅ Metrics lengkap dengan Cross-Validation
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        ### 🎯 Akurat
        Model {best_model_name} dengan akurasi **{results[best_model_name]['accuracy']:.1%}**
        """)
    
    with col2:
        st.success("""
        ### 🚀 Cepat
        Hasil prediksi dalam hitungan detik
        """)
    
    with col3:
        st.warning(f"""
        ### 📊 Informatif
        F1-Score: **{results[best_model_name]['f1_score']:.1%}**
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
        3. **Lihat Perbandingan Model** - Pahami model yang digunakan
        4. **Lakukan Prediksi** - Input data dan lihat hasilnya
        
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
        st.metric("Akurasi", f"{results[best_model_name]['accuracy']:.1%}")
    
    with col2:
        st.metric("F1-Score", f"{results[best_model_name]['f1_score']:.1%}")
    
    with col3:
        st.metric("Total Data", len(heart_data))
    
    with col4:
        st.metric("Jumlah Fitur", 13)
    
    with col5:
        st.metric("Model Terbaik", best_model_name.split()[0])
    
    st.markdown("---")
    
    # Model comparison preview
    st.subheader("🆚 Preview Perbandingan Model")
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for name, result in results.items():
        fig.add_trace(go.Bar(
            name=name,
            x=metrics_names,
            y=[result['accuracy'], result['precision'], result['recall'], result['f1_score']],
            text=[f"{result['accuracy']:.2%}", f"{result['precision']:.2%}", 
                  f"{result['recall']:.2%}", f"{result['f1_score']:.2%}"],
            textposition='auto',
        ))
    
    fig.update_layout(
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Lihat detail lengkap di menu **'📊 Perbandingan Model'**")
    
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
    - **📊 Perbandingan Model** - Lihat performa semua model
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>⚕️ Aplikasi ini untuk tujuan edukasi. Konsultasikan dengan tenaga medis profesional untuk diagnosis yang akurat.</p>
    <p>© 2024 Sistem Prediksi Penyakit Jantung v2.0 | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p><strong>Model Terbaik: {best_model_name}</strong> | Akurasi: {results[best_model_name]['accuracy']:.2%}</p>
</div>
""", unsafe_allow_html=True)