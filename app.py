import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application predicts the likelihood of heart disease based on various medical parameters.
Please enter the patient's information in the sidebar to get a prediction.
""")

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
    
    # Train model
    model = DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, pca, X, y, X_test, y_test, y_pred, accuracy, heart_data

# Load data and model
try:
    model, scaler, pca, X, y, X_test, y_test, y_pred, accuracy, heart_data = load_and_prepare_data()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar for user input
st.sidebar.header("📋 Patient Information")
st.sidebar.markdown("Enter the patient's medical parameters:")

# Create input fields based on original features
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (chol)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
thalach = st.sidebar.slider("Maximum Heart Rate (thalach)", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST (slope)", options=[0, 1, 2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

# Create prediction button
predict_button = st.sidebar.button("🔍 Predict", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Model Performance")
    
    # Display accuracy
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['No Disease', 'Disease'],
                       y=['No Disease', 'Disease'],
                       text_auto=True,
                       color_continuous_scale='Blues')
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("📈 Dataset Overview")
    
    # Target distribution
    target_counts = heart_data['target'].value_counts()
    fig_dist = go.Figure(data=[go.Pie(
        labels=['No Disease', 'Disease'],
        values=target_counts.values,
        hole=0.4,
        marker=dict(colors=['#2ecc71', '#e74c3c'])
    )])
    fig_dist.update_layout(title="Target Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.metric("Total Samples", len(heart_data))
    st.metric("Features", len(heart_data.columns) - 1)

# Prediction section
if predict_button:
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
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
            st.error("⚠️ HIGH RISK: Heart Disease Detected")
            st.markdown(f"**Confidence:** {prediction_proba[1]:.1%}")
            st.markdown("""
            ### Recommendations:
            - Consult a cardiologist immediately
            - Follow prescribed medication
            - Maintain a healthy lifestyle
            - Regular monitoring required
            """)
        else:
            st.success("✅ LOW RISK: No Heart Disease Detected")
            st.markdown(f"**Confidence:** {prediction_proba[0]:.1%}")
            st.markdown("""
            ### Recommendations:
            - Continue healthy lifestyle
            - Regular check-ups recommended
            - Maintain balanced diet
            - Stay physically active
            """)
    
    # Probability chart
    st.markdown("---")
    st.subheader("📊 Prediction Probability")
    
    fig_proba = go.Figure(data=[
        go.Bar(x=['No Disease', 'Disease'], 
               y=prediction_proba,
               marker=dict(color=['#2ecc71', '#e74c3c']),
               text=[f'{p:.1%}' for p in prediction_proba],
               textposition='auto')
    ])
    fig_proba.update_layout(
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    st.plotly_chart(fig_proba, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚕️ This application is for educational purposes only. Please consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)