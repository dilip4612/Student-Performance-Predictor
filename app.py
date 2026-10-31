
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the expected exam score.")

st.divider()

st.subheader("📚 Student Details")

# Daily study hours
study_hours = st.number_input(
    "Daily Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Previous exam marks
st.subheader("Previous Exam Marks")

previous_marks = st.number_input(
    "Marks Obtained",
    min_value=0.0,
    value=500.0,
    step=1.0
)

maximum_marks = st.number_input(
    "Maximum Marks",
    min_value=1.0,
    value=600.0,
    step=1.0
)

# Attendance
attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

# Assignments
assignments = st.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=10,
    value=8,
    step=1
)

st.divider()

if st.button("Predict Exam Score", type="primary"):

    if previous_marks > maximum_marks:
        st.error("Marks obtained cannot exceed maximum marks.")

    else:
        # Convert previous marks into percentage
        previous_percentage = (
            previous_marks / maximum_marks
        ) * 100

        # Prepare input for the model
        input_data = pd.DataFrame([{
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Score": previous_percentage,
            "Assignments_Completed": assignments
        }])

        # Predict exam score
        prediction = model.predict(input_data)[0]

        prediction = max(0, min(100, prediction))

        st.success("Prediction Completed!")

        st.metric(
            "Predicted Exam Score",
            f"{prediction:.2f} / 100"
        )

        st.info(
            f"Previous exam percentage: "
            f"{previous_percentage:.2f}%"
        )

        st.caption(
            "This is an educational estimate based on a "
            "synthetic training dataset, not a guaranteed result."
        )

st.divider()

st.caption("Student Performance Predictor | Machine Learning Project")

