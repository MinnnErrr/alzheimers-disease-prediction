import streamlit as st
import pandas as pd
import joblib

model=joblib.load("final_model.pkl")

st.title("Alzheimer's Disease Prediction App")

age = st.number_input("Age", min_value=60, max_value=90, step=1)

gender = st.selectbox("Gender", ["Male", "Female"])

ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African America", "Asian", "Other"])

education = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"])

bmi = st.number_input("BMI", min_value=15.00, max_value=40.00, format="%.2f")

smoking = st.selectbox("Smoking Status", ["No", "Yes"])

alcohol = st.number_input("Weekly Alchohol Consumption", min_value=0.00, max_value=20.00, format="%.2f")

phy = st.number_input("Physical Activity Hours", min_value=0.00, max_value=10.00, format="%.2f")

diet = st.number_input("Diet Quality Score", min_value=0.00, max_value=10.00, format="%.2f")

sleep = st.number_input("Sleep Quality Score", min_value=4.00, max_value=10.00, format="%.2f")

history = st.selectbox("Family History of Alzheimer's Disease", ["No", "Yes"])

cardio = st.selectbox ("Presence of Cardiovascular Disease", ["No", "Yes"])

diabetes = st.selectbox("Presence of Diabetes", ["No", "Yes"])

depression = st.selectbox("Presence of Depression", ["No", "Yes"])

head = st.selectbox("History of Head Injury", ["No", "Yes"])

hypertension = st.selectbox("Hypertension", ["No", "Yes"])

systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=90, max_value=180, step=1)

diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=60, max_value=120, step=1)

cholesterolTotal = st.number_input("Total Cholesterol Level (mg/dL)", min_value=150.00, max_value=300.00, format="%.2f")

cholesterolLDL = st.number_input("Low-density Lipoprotein Cholesterol Level (mg/dL)", min_value=50.00, max_value=200.00, format="%.2f")

cholesterolHDL = st.number_input("High-density Lipoprotein Cholesterol Level (mg/dL)", min_value=20.00, max_value=100.00, format="%.2f")

cholesterolTri = st.number_input("Triglycerides Level (mg/dL)", min_value=50.00, max_value=400.00, format="%.2f")

mmse = st.number_input("Mini-Mental State Examination Score (MMSE)", min_value=0.00, max_value=30.00, format="%.2f")

func = st.number_input("Functional Assessment Score", min_value=0.00, max_value=10.00, format="%.2f")

memory = st.selectbox("MemoryComplaints", ["No", "Yes"])

behavioral = st.selectbox("BehaviouralProblems", ["No", "Yes"])

adl = st.number_input("Activities of Daily Living Score", min_value=0.00, max_value=10.00, format="%.2f")

confusion = st.selectbox("Presnece of Confusion", ["No", "Yes"])

disorientation = st.selectbox("Presence of Disorientation", ["No", "Yes"])

personality = st.selectbox("Presence of Personality Changes", ["No", "Yes"])

difficulty = st.selectbox("Presence of Difficulty Completing Tasks", ["No", "Yes"])

forget = st.selectbox("Presence of Forgetfullness", ["No", "Yes"])

#labelling categorical festures
gender_label = {"Male": 0, "Female": 1}[gender]
ethnicity_label = {"Caucasian": 0, "African America": 1, "Asian": 2, "Other": 4}[ethnicity]
edu_label = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}[education]
smoking_label = {"No": 0, "Yes": 1}[smoking]
history_label = {"No": 0, "Yes": 1}[history]
cardio_label = {"No": 0, "Yes": 1}[cardio]
diabetes_label = {"No": 0, "Yes": 1}[diabetes]
depression_label = {"No": 0, "Yes": 1}[depression]
head_label = {"No": 0, "Yes": 1}[head]
hypertension_label = {"No": 0, "Yes": 1}[hypertension]
memory_label = {"No": 0, "Yes": 1}[memory]
behavioural_label = {"No": 0, "Yes": 1}[behavioral]
confusion_label = {"No": 0, "Yes": 1}[confusion]
disorientation_label = {"No": 0, "Yes": 1}[disorientation]
personality_label = {"No": 0, "Yes": 1}[personality]
difficulty_label = {"No": 0, "Yes": 1}[difficulty]
forget_label = {"No": 0, "Yes": 1}[forget]

if st.button("Predict"):
    input_data=pd.DataFrame({
        'Age': [age],
        'Gender': [gender_label],
        'Ethnicity': [ethnicity_label],
        'EducationLevel': [edu_label],
        'BMI': [bmi],
        'Smoking': [smoking_label],
        'AlcoholConsumption': [alcohol],
        'PhysicalActivity': [phy],
        'DietQuality': [diet],
        'SleepQuality': [sleep],
        'FamilyHistoryAlzheimers': [history_label],
        'CardiovascularDisease': [cardio_label],
        'Diabetes': [diabetes_label],
        'Depression': [depression_label],
        'HeadInjury': [head_label],
        'Hypertension': [hypertension_label],
        'SystolicBP': [systolic],
        'DiastolicBP': [diastolic],
        'CholesterolTotal': [cholesterolTotal],
        'CholesterolLDL': [cholesterolLDL],
        'CholesterolHDL': [cholesterolHDL],
        'CholesterolTriglycerides': [cholesterolTri],
        'MMSE': [mmse],
        'FunctionalAssessment': [func],
        'MemoryComplaints': [memory_label],
        'BehavioralProblems': [behavioural_label],
        'ADL': [adl],
        'Confusion': [confusion_label],
        'Disorientation': [disorientation_label],
        'PersonalityChanges': [personality_label],
        'DifficultyCompletingTasks': [difficulty_label],
        'Forgetfulness': [forget_label]
    })

    st.subheader("Input Data")
    st.dataframe(input_data)
    
    prediction=model.predict(input_data)[0]
    prediction_prob=model.predict_proba(input_data)[0]

    st.subheader("Result")
    st.write("Prediction (1: likely Alzhemier's, 0: unlikely Alzheimer's): ", prediction)
    st.write("Confidence: {:.2f}%".format(100*max(prediction_prob)))