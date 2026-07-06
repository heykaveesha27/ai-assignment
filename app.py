from altair import Color, value
from narwhals._compliant import column
import streamlit as sl


tab1, tab2, tab3 = sl.tabs(["Home", "Data Analysis","Dataset"])

with tab1:
		
	sl.title(" Students Result Predictor")
	for key in ["attendance", "quizz_avg", "project_score", "assignment_avg", "participation_score", "study_hours", "sleep", "stress"]:
		if key not in sl.session_state:
			sl.session_state[key]= 0

	

	attendance = sl.number_input("Attendence Percentage",min_value=0,value=sl.session_state.attendance)
	sl.session_state.attendance=attendance
	quizz_avg = sl.number_input("Average Quizz Score", min_value=0,value=sl.session_state.quizz_avg)
	sl.session_state.quizz_avg=quizz_avg
	project_score = sl.number_input("Project Score", min_value=0,value=sl.session_state.project_score)
	sl.session_state.project_score=project_score
	assignment_avg = sl.number_input("Average Assignment Score", min_value=0 ,value=sl.session_state.assignment_avg)
	sl.session_state.assignment_avg = assignment_avg
	participation_score = sl.number_input("Participation Score", min_value=0, value=sl.session_state.participation_score)
	sl.session_state.participation_score = participation_score
	study_hours = sl.number_input("Study Hours", min_value=0,value=sl.session_state.study_hours)
	sl.session_state.study_hours = study_hours
	sleep = sl.number_input("Sleep Hours",min_value=0,value=sl.session_state.sleep)
	sl.session_state.sleep=sleep
	stress = sl.number_input("Stress Level", min_value=0,value=sl.session_state.stress)
	sl.session_state.stress = stress

	col1, col2 =sl.columns([1,1])

	with col1:
		predict=sl.button("Predict",use_container_width=True)
			#For now just a placeholder formula
				
	with col2:
		clear=sl.button("Clear All",use_container_width=True)


	sl.markdown("""
    <style>
    /* First button (Predict) */
    div.stButton:nth-child(1) > button {
        background-color: #4CAF50; /* Green */
        color: white;
        height: 3em;
        border-radius: 8px;
        border: none;
    }
    div.stButton:nth-child(1) > button:hover {
        background-color: #45a049;
    }

    /* Second button (Clear All) */
    div.stButton:nth-child(2) > button {
        background-color: #f44336; /* Red */
        color: white;
        height: 3em;
        border-radius: 8px;
        border: none;
    }
    div.stButton:nth-child(2) > button:hover {
        background-color: #da190b;
    }
    </style>
    """, unsafe_allow_html=True)


	if predict:
		result = sl.session_state.study_hours * 10 + sl.session_state.sleep - sl.session_state.stress * 2
		sl.success(f"Predicted Result : {result:.2f}")

	if clear:
		for key in ["attendance", "quizz_avg", "project_score", "assignment_avg", "participation_score", "study_hours", "sleep", "stress"]:
	        	sl.session_state[key] = 0
     
		





with tab2:
	sl.subheader("Data Analysis")
	sl.write("Data Analysis section")

with tab3:
	sl.subheader("Dataset")
	sl.write("View the dataset")

