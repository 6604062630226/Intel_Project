import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder,RobustScaler

# โหลดโมเดล
model = joblib.load("models/ensemble_model.joblib")

# UI ของ Streamlit
st.title("Heart Disease Prediction")

# อินพุตจากผู้ใช้
age = st.slider("📅 อายุ", 10, 100, 30)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
rbp = st.slider("🩸 RestingBloodPressure", 50, 250, 100)
chol = st.slider("Cholesterol", 0, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
MaxHR = st.slider("Maximum Heart Rate Achieved", 0, 250, 150)
exang = st.selectbox("Exercise Induced Angina", ['N', 'Y'])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Up", "Flat", "Down"])



# สร้าง DataFrame จากอินพุตของผู้ใช้
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "ChestPainType": cp,
    "RestingBP": rbp,
    "Cholesterol": chol,
    "FastingBS": fbs,
    "RestingECG": restecg,
    "MaxHR": MaxHR,
    "ExerciseAngina": exang,
    "Oldpeak": oldpeak,
    "ST_Slope": slope
}])

# 🔹 **One-Hot Encoding ให้ตรงกับโมเดล**
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=False)

# 🔹 **โหลดฟีเจอร์ที่โมเดลเคยเห็น**
expected_features = model.feature_names_in_

# เติมค่า 0 ให้ฟีเจอร์ที่ขาดหาย
for feature in expected_features:
    if feature not in input_data.columns:
        input_data[feature] = 0

# ลบฟีเจอร์ที่โมเดลไม่เคยเห็น
input_data = input_data[expected_features]

# **🔹 ใช้ RobustScaler (เหมือนตอนเทรนโมเดล)**
scaler = RobustScaler()
input_data_scaled = scaler.fit_transform(input_data)

# **🔍 ตรวจสอบอินพุตก่อนทำนาย**
st.subheader("✅ ตรวจสอบข้อมูลที่ป้อน")
st.write(input_data)

# **🧠 ทำนายผล**
if st.button("🔮 Predict"):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[:, 1]

    st.subheader("📊 ผลลัพธ์การทำนาย")
    if prediction[0] == 1:
        st.error(f"🛑 มีความเสี่ยงเป็นโรคหัวใจ! (โอกาส: {probability[0]:.2%})")
    else:
        st.success(f"✅ ไม่มีความเสี่ยงโรคหัวใจ (โอกาส: {probability[0]:.2%})")



