import streamlit as st
import streamlit.components.v1 as components
import joblib

import pandas as pd
import numpy as np

# Load predictive model & Its Supoorting files
@st.cache_resource
def load_model():
    return joblib.load("medibot_model.pkl"), joblib.load("preprocess_medibot.pkl"), joblib.load("medibot_le.pkl")

model, preprocessor, le = load_model()
df1=pd.read_csv('expanded_medical_dataset_500.csv')
# Questions to ask
questions = [
    "What is your age?",
    "What are your symptoms?",
    "What is severity(Low,Moderate,High)?"
]

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "completed" not in st.session_state:
    st.session_state.completed = False

# Function to run prediction
def run_model(answers):
    try:
        age = int(answers["What is your age?"])
        symptoms = answers["What are your symptoms?"]
        severity = answers["What is severity(Low,Moderate,High)?"]
        
        details = {
                    'Age': age,
                    'Symptoms': symptoms,
                    'Severity': severity,
                    
                    }

        #details['Skills_Interests'] = details['Skills'] + ' ' + details['Interests']
        details_df = pd.DataFrame([{
            'Age': details['Age'],
            'Symptoms': details['Symptoms'],
            'Severity': details['Severity'],
            
        }])

        # Preprocess new user
        x = preprocessor.transform(details_df)

        # Predict probabilities
        proba = model.predict_proba(x)

        # Top-3 predictions
       
        
        top3_indices = np.argsort(proba[0])[::-1][:3]
        top3_diseases = le.inverse_transform(top3_indices)

        result = "Your expected diseases:\n"
    
        for disease_name in top3_diseases:
            match1 = df1[df1["Expected Disease"].str.contains(disease_name, case=False, na=False)].head(1)
        
            advice = match1.iloc[0]["Initial Doctor Advice"]
        
            
        
            result += f"\n- {disease_name}\n * Doctor's Advice: {advice}\n"

        return result

    except Exception as e:
        return f"Error during prediction: {e}"
       
    
    
    



# Header
st.subheader(":orange[AI-Powered MediBot]")
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:

    # Intro message outside scrollable container to keep it pinned
    intro_html = """<div style='text-align: left; margin-bottom: 10px;'>
        <span style='background-color: red; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>
        🤖: Hello! I'm your AI assistant to help you predict the disease.
        </span>
        </div>"""

    chat_html = """<div id="chat-container" style="height: 300px; overflow-y: auto; border: 1px solid #ccc;
    border-radius: 10px; padding: 10px; background-color: black; margin-bottom: 10px;">"""

    for i in range(st.session_state.step):
        q = questions[i]
        a = st.session_state.answers.get(q, "")
        chat_html += f"""<div style='text-align: left; margin-bottom: 10px;'>
            <span style='background-color: blue; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🤖: {q}</span>
            </div>
            <div style='text-align: right; margin-bottom: 10px;'>
            <span style='background-color: green; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🧑: {a}</span>
            </div>"""

    if not st.session_state.completed:
        current_question = questions[st.session_state.step]
        chat_html += f"""<div style='text-align: left; margin-bottom: 10px;'>
            <span style='background-color: blue; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🤖: {current_question}</span>
            </div>"""
    else:
        prediction = run_model(st.session_state.answers)
        chat_html += "<div style='text-align: left; margin-bottom: 10px;'>"
        chat_html += f"""<div style='margin-bottom: 5px;'>
                <span style='background-color: purple; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>🤖: {prediction.split("-")[0]}</span>
                </div>"""
        for rec in prediction.split("-")[1:]:
            chat_html += f"""<div style='margin-bottom: 5px;'>
                <span style='background-color: purple; padding: 8px 12px; border-radius: 10px; display: inline-block; color: white;'>{rec}</span>
                </div>"""
        chat_html += "</div>"

    chat_html += """</div><script>document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;</script>"""
    components.html(intro_html + chat_html, height=400)

# Chat input
cola, colb, colc = st.columns([0.2,0.6,0.2])
with colb:
    if not st.session_state.completed:
        user_input = st.chat_input("Your answer:")
        if user_input:
            q = questions[st.session_state.step]
            st.session_state.answers[q] = user_input
            st.session_state.step += 1
            if st.session_state.step == len(questions):
                st.session_state.completed = True
            st.rerun()
