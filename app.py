
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict student exam performance using Machine Learning."
)

MODEL_FILE = "student_performance_model.pkl"
DATASET_FILE = "student_performance_dataset.csv"


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# Load original dataset
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)


model = load_model()
df = load_dataset()


# Initialize temporary session history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


dashboard_tab, prediction_tab = st.tabs(
    ["📊 Dashboard", "🔮 Prediction"]
)


# ---------------- DASHBOARD ----------------

with dashboard_tab:

    st.header("Student Performance Dashboard")

    history = st.session_state.prediction_history

    total_students = len(df)
    average_score = df["Exam_Score"].mean()
    highest_score = df["Exam_Score"].max()
    average_attendance = df["Attendance"].mean()

    total_predictions = len(history)

    if total_predictions > 0:
        average_predicted_score = sum(
            item["Predicted_Score"] for item in history
        ) / total_predictions
    else:
        average_predicted_score = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Average Exam Score", f"{average_score:.2f}")
    col3.metric("Highest Exam Score", f"{highest_score:.2f}")
    col4.metric(
        "Average Attendance",
        f"{average_attendance:.2f}%"
    )

    st.divider()

    col5, col6 = st.columns(2)

    col5.metric("Total Predictions", total_predictions)

    col6.metric(
        "Average Predicted Score",
        f"{average_predicted_score:.2f} / 100"
    )

    st.caption(
        "Prediction statistics are temporary and belong "
        "to your current website session."
    )

    st.divider()

    st.subheader("📈 Exam Score Distribution")

    score_distribution = df["Exam_Score"].value_counts(
        bins=10,
        sort=False
    ).sort_index()

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

    st.subheader("📝 Your Prediction History")

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True
        )

        history_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Save All My Predictions",
            data=history_csv,
            file_name="my_prediction_history.csv",
            mime="text/csv",
            key="download_all"
        )

        st.divider()

        if st.button(
            "🗑️ Clear My Prediction History",
            type="secondary",
            use_container_width=True
        ):
            st.session_state.prediction_history = []
            st.session_state.last_prediction = None
            st.rerun()

    else:

        st.info(
            "You have not made any predictions in this session yet."
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
        step=0.5,
        key="study_hours"
    )

    col1, col2 = st.columns(2)

    with col1:
        previous_marks = st.number_input(
            "Previous Marks Obtained",
            min_value=0.0,
            value=400.0,
            step=1.0,
            key="previous_marks"
        )

    with col2:
        maximum_marks = st.number_input(
            "Maximum Marks",
            min_value=1.0,
            value=500.0,
            step=1.0,
            key="maximum_marks"
        )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        key="attendance"
    )

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=100,
        value=8,
        step=1,
        key="assignments"
    )

    if st.button(
        "🔮 Predict Exam Score",
        use_container_width=True
    ):

        if previous_marks > maximum_marks:

            st.session_state.last_prediction = None

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

            prediction = max(
                0,
                min(100, float(prediction))
            )

            prediction = round(prediction, 2)

            new_record = {
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Study_Hours": study_hours,
                "Previous_Marks": previous_marks,
                "Maximum_Marks": maximum_marks,
                "Previous_Percentage": round(
                    previous_percentage, 2
                ),
                "Attendance": attendance,
                "Assignments_Completed": assignments,
                "Predicted_Score": prediction,
                "Predicted_Percentage": prediction
            }

            st.session_state.prediction_history.append(
                new_record
            )

            st.session_state.last_prediction = new_record

            st.rerun()

    # Display latest prediction
    last_prediction = st.session_state.last_prediction

    if last_prediction is not None:

        st.success("Prediction completed successfully!")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Predicted Exam Score",
            f"{last_prediction['Predicted_Score']:.2f} / 100"
        )

        result_col2.metric(
            "Predicted Exam Percentage",
            f"{last_prediction['Predicted_Percentage']:.2f}%"
        )

        st.info(
            "This is an estimated result generated "
            "by the trained machine learning model."
        )

        latest_df = pd.DataFrame([last_prediction])

        latest_csv = latest_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="💾 Save This Prediction",
            data=latest_csv,
            file_name="my_student_prediction.csv",
            mime="text/csv",
            key="download_latest"
        )

        st.caption(
            "Your prediction is temporary unless you download and save it."
        )
