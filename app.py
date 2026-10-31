
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write("Predict student exam performance using Machine Learning.")

MODEL_FILE = "student_performance_model.pkl"
DATASET_FILE = "student_performance_dataset.csv"
HISTORY_FILE = "prediction_history.csv"

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

# Load dataset
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)

df = load_dataset()

# Create prediction history file
if not os.path.exists(HISTORY_FILE):
    history_columns = [
        "Date",
        "Study_Hours",
        "Previous_Score",
        "Attendance",
        "Assignments_Completed",
        "Predicted_Score"
    ]
    pd.DataFrame(columns=history_columns).to_csv(
        HISTORY_FILE, index=False
    )

# Load prediction history
def load_history():
    return pd.read_csv(HISTORY_FILE)

# Dashboard and prediction tabs
dashboard_tab, prediction_tab = st.tabs(
    ["📊 Dashboard", "🔮 Prediction"]
)

# ---------------- DASHBOARD ----------------

with dashboard_tab:

    st.header("Student Performance Dashboard")

    history = load_history()

    total_students = len(df)
    average_score = df["Exam_Score"].mean()
    highest_score = df["Exam_Score"].max()
    average_attendance = df["Attendance"].mean()

    total_predictions = len(history)

    if total_predictions > 0:
        average_predicted_score = history["Predicted_Score"].mean()
    else:
        average_predicted_score = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Average Exam Score", f"{average_score:.2f}")
    col3.metric("Highest Exam Score", f"{highest_score:.2f}")
    col4.metric("Average Attendance", f"{average_attendance:.2f}%")

    st.divider()

    col5, col6 = st.columns(2)

    col5.metric("Total Predictions", total_predictions)
    col6.metric(
        "Average Predicted Score",
        f"{average_predicted_score:.2f}"
    )

    st.divider()

    st.subheader("📈 Exam Score Distribution")

    score_distribution = (
        df["Exam_Score"]
        .value_counts(bins=10, sort=False)
        .sort_index()
    )

    st.bar_chart(score_distribution)

    st.subheader("📚 Study Hours vs Exam Score")

    st.scatter_chart(
        df,
        x="Study_Hours",
        y="Exam_Score"
    )

    st.subheader("⭐ Feature Importance")

    feature_names = [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Assignments_Completed"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.divider()

    st.subheader("📝 Prediction History")

    if not history.empty:

        history["Date"] = pd.to_datetime(
            history["Date"],
            errors="coerce"
        )

        chart_history = history.dropna(
            subset=["Date", "Predicted_Score"]
        ).set_index("Date")

        st.line_chart(
            chart_history["Predicted_Score"]
        )

        st.dataframe(
            history,
            use_container_width=True
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "No predictions yet. Go to the Prediction tab "
            "to make your first prediction."
        )

    st.divider()

    st.subheader("📋 Original Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

with prediction_tab:

    st.header("Predict Student Exam Performance")

    st.write(
        "Enter the student's academic details below."
    )

    study_hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    col1, col2 = st.columns(2)

    with col1:
        previous_marks = st.number_input(
            "Previous Marks Obtained",
            min_value=0.0,
            value=400.0,
            step=1.0
        )

    with col2:
        maximum_marks = st.number_input(
            "Maximum Marks",
            min_value=1.0,
            value=500.0,
            step=1.0
        )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=100,
        value=8,
        step=1
    )

    if st.button(
        "🔮 Predict Exam Score",
        use_container_width=True
    ):

        if previous_marks > maximum_marks:

            st.error(
                "Previous marks cannot exceed maximum marks."
            )

        else:

            previous_percentage = (
                previous_marks / maximum_marks
            ) * 100

            input_data = pd.DataFrame([{
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Previous_Score": previous_percentage,
                "Assignments_Completed": assignments
            }])

            prediction = model.predict(input_data)[0]

            prediction = max(0, min(100, prediction))

            new_record = pd.DataFrame([{
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Study_Hours": study_hours,
                "Previous_Score": previous_percentage,
                "Attendance": attendance,
                "Assignments_Completed": assignments,
                "Predicted_Score": round(prediction, 2)
            }])

            new_record.to_csv(
                HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )

            st.success("Prediction completed successfully!")

            st.metric(
                "Predicted Exam Score",
                f"{prediction:.2f} / 100"
            )

            st.info(
                "This is an estimated score based on the "
                "trained machine learning model."
            )

            
