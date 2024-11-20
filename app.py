import streamlit as st
import joblib
import numpy as np

# Modelni yuklash
MODEL_PATH = 'lung_cancer_prediction_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    st.error(f"Modelni yuklashda xatolik: {e}")

def predict(features):
    prediction = model.predict(features)
    result = 'Lung Cancer Detected' if prediction[0] == 1 else 'No Lung Cancer Detected'
    return result

def app():
    st.title("Lung Cancer Prediction")

    # Form ma'lumotlarini olish
    gender = st.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
    age = st.number_input("Age", min_value=18, max_value=100)
    smoking = st.selectbox("Smoking", [0, 1])  # 0: No, 1: Yes
    yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure", [0, 1])
    chronic_disease = st.selectbox("Chronic Disease", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])
    allergy = st.selectbox("Allergy", [0, 1])
    wheezing = st.selectbox("Wheezing", [0, 1])
    alcohol_consuming = st.selectbox("Alcohol Consuming", [0, 1])
    coughing = st.selectbox("Coughing", [0, 1])
    shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", [0, 1])
    chest_pain = st.selectbox("Chest Pain", [0, 1])

    # Modelga kiritish uchun ma'lumotlarni tayyorlash
    features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                          chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                          coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

    if st.button("Predict"):
        result = predict(features)
        st.write(f"Prediction: {result}")

# This line should be removed
# app.run(debug=True)

if __name__ == '__main__':
    app()
