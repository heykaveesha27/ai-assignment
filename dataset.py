import streamlit as sl


tab1, tab2, tab3 = sl.tabs(["Home", "Data Analysis","Dataset"])

with tab1:
	sl.subheader("Home")
	sl.write("Welcome to the Students Result Predictor")
	sl.title("Dataset")


with tab2:
	sl.subheader("Data Analysis")
	sl.write("Data Analysis section")

with tab3:
	sl.subheader("Dataset")
	sl.write("View the dataset")

	
sl.title("Dataset")




attendance = sl.number_input("Attendence Percentage",min_value=0.00)
quizz_avg = sl.number_input("Average Quizz Score", min_value=0.00)
project_score = sl.number_input("Project Score", min_value=0.00)
assignment_avg = sl.number_input("Average Assignment Score", min_value=0.00)
participation_score = sl.number_input("Participation Score", min_value=0.00)
study = sl.number_input("Study Hours", min_value=0)
sleep = sl.number_input("Sleep Hours",min_value=0)
stress = sl.number_input("Stress Level", min_value=0)


if sl.button("Predict"):
	#For now just a placeholder formula
	result = study * 10 + sleep * - stress * 2
	sl.success(f"Predicted Result : {result:.2f}")