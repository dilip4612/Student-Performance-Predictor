
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

model = joblib.load("student_performance_model.pkl")
df = pd.read_csv("student_performance_dataset.csv")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


st.sidebar.title("🎓 Student Performance Predictor")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🔮 Prediction"]
)


if page == "🏠 Home":

    st.title("🎓 Student Performance Predictor")

    st.write(
        "Predict a student's exam performance using study hours, "
        "attendance, previous exam performance and assignments."
    )

    st.info(
        "Use the Prediction page to enter student details and generate a prediction."
    )


elif page == "📊 Dashboard":

    st.title("📊 Dashboard")

    total_predictions = len(st.session_state.prediction_history)

    if total_predictions > 0:

        predicted_scores = [
            item["Predicted Score"]
            for item in st.session_state.prediction_history
        ]

        attendances = [
            item["Attendance"]
            for item in st.session_state.prediction_history
        ]

        highest_score = max(predicted_scores)
        average_score = sum(predicted_scores) / len(predicted_scores)
        average_attendance = sum(attendances) / len(attendances)

    else:

        predicted_scores = []
        average_score = 0
        highest_score = 0
        average_attendance = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", total_predictions)

    with col2:
        st.metric(
            "Average Predicted Score",
            f"{average_score:.2f}"
        )

    with col3:
        st.metric(
            "Highest Score",
            f"{highest_score:.2f}"
        )

    with col4:
        st.metric(
            "Average Attendance",
            f"{average_attendance:.2f}%"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 Score Distribution")

        if predicted_scores:

            chart_data = pd.DataFrame(
                {"Predicted Score": predicted_scores}
            )

            st.bar_chart(chart_data)

        else:

            st.info("No predictions available yet.")

    with col2:

        st.subheader("📚 Study Hours vs Score")

        if total_predictions > 0:

            study_score_data = pd.DataFrame({
                "Study Hours": [
                    item["Study Hours"]
                    for item in st.session_state.prediction_history
                ],
                "Predicted Score": predicted_scores
            })

            st.line_chart(
                study_score_data.set_index("Study Hours")
            )

        else:

            st.info("No prediction data available yet.")

    st.divider()

    st.subheader("⭐ Feature Importance")

    feature_names = [
        "Study Hours",
        "Attendance",
        "Previous Score",
        "Assignments Completed"
    ]

    importance_data = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("🕘 Prediction History")

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("🗑️ Delete History"):

            st.session_state.prediction_history = []
            st.session_state.last_prediction = None

            st.success("Prediction history deleted.")

            st.rerun()

    else:

        st.info("No prediction history yet.")


elif page == "🔮 Prediction":

    st.title("🔮 Predict Student Performance")

    st.write("Enter the student's details below.")

    col1, col2 = st.columns(2)

    with col1:

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        previous_marks = st.number_input(
            "Previous Exam Marks",
            min_value=0.0,
            value=500.0,
            step=1.0
        )

        previous_max_marks = st.number_input(
            "Previous Exam Maximum Marks",
            min_value=1.0,
            value=625.0,
            step=1.0
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col2:

        assignments = st.number_input(
            "Assignments Completed",
            min_value=0,
            max_value=10,
            value=8,
            step=1
        )

        actual_marks = st.number_input(
            "Actual Exam Marks",
            min_value=0.0,
            value=540.0,
            step=1.0
        )

        actual_max_marks = st.number_input(
            "Actual Exam Maximum Marks",
            min_value=1.0,
            value=600.0,
            step=1.0
        )

    st.write("")

    if st.button(
        "🔮 Predict Exam Score",
        use_container_width=True
    ):

        if previous_marks > previous_max_marks:

            st.error(
                "Previous Exam Marks cannot be greater than "
                "Previous Exam Maximum Marks."
            )

        elif actual_marks > actual_max_marks:

            st.error(
                "Actual Exam Marks cannot be greater than "
                "Actual Exam Maximum Marks."
            )

        else:

            previous_percentage = (
                previous_marks / previous_max_marks
            ) * 100

            actual_percentage = (
                actual_marks / actual_max_marks
            ) * 100

            input_data = pd.DataFrame({
                "Study_Hours": [study_hours],
                "Attendance": [attendance],
                "Previous_Score": [previous_percentage],
                "Assignments_Completed": [assignments]
            })

            predicted_score = float(
                model.predict(input_data)[0]
            )

            predicted_score = max(
                0,
                min(100, predicted_score)
            )

            predicted_percentage = predicted_score

            if actual_percentage > previous_percentage:

                improvement_message = (
                    "🎉 Yes! You improved! Keep up the great work!"
                )

            elif actual_percentage < previous_percentage:

                improvement_message = (
                    "💪 No, you still need to improve. "
                    "Keep working hard—you can do it!"
                )

            else:

                improvement_message = (
                    "📚 You maintained your performance. "
                    "Keep working to improve!"
                )

            prediction = {
                "Study Hours": study_hours,
                "Previous Exam Percentage": round(
                    previous_percentage,
                    2
                ),
                "Attendance": attendance,
                "Assignments Completed": assignments,
                "Predicted Score": round(
                    predicted_score,
                    2
                ),
                "Predicted Percentage": round(
                    predicted_percentage,
                    2
                ),
                "Actual Exam Percentage": round(
                    actual_percentage,
                    2
                )
            }

            st.session_state.last_prediction = prediction

            st.session_state.prediction_history.append(
                prediction
            )

            st.success(
                "Prediction completed successfully!"
            )

            st.subheader("📊 Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Previous Exam Percentage",
                    f"{previous_percentage:.2f}%"
                )

                st.metric(
                    "Predicted Exam Score",
                    f"{predicted_score:.2f}"
                )

            with result_col2:

                st.metric(
                    "Predicted Exam Percentage",
                    f"{predicted_percentage:.2f}%"
                )

                st.metric(
                    "Actual Exam Percentage",
                    f"{actual_percentage:.2f}%"
                )

            st.info(improvement_message)

            st.divider()

            st.subheader("💾 Save Prediction")

            save_data = pd.DataFrame([prediction])

            csv_data = save_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "💾 Save This Prediction",
                data=csv_data,
                file_name="student_prediction.csv",
                mime="text/csv"
            )

            if st.session_state.prediction_history:

                all_data = pd.DataFrame(
                    st.session_state.prediction_history
                )

                all_csv = all_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "📥 Save All Predictions",
                    data=all_csv,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )

            st.success(
                "Prediction added to your current session history."
            )
