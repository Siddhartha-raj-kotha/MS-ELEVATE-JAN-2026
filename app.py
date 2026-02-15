import streamlit as st
import pandas as pd
import pickle
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ProFit AI",
    page_icon="💪",
    layout="wide"
)

# --- 2. ADAPTIVE THEME CSS ---
st.markdown("""
    <style>
    .main-header {
        color: var(--text-color) !important;
        font-weight: 800;
        font-size: 38px;
        text-align: center;
        margin-bottom: 20px;
    }
    .recommendation-card {
        background-color: var(--background-color);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        border-top: 5px solid #10b981;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .recommendation-card h3, .recommendation-card p, .recommendation-card b {
        color: var(--text-color) !important;
    }
    .workout-box {
        background-color: #1e293b;
        color: #ffffff !important;
        padding: 20px;
        border-radius: 12px;
    }
    .workout-box * {
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #10b981 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    label, .stSelectbox, .stSlider {
        color: var(--text-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & AI LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('models/fitness_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        data = pd.read_csv('data/diet_data.csv')
        return model, scaler, data
    except:
        return None, None, None

model, scaler, df = load_assets()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>ProFit AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Physical Profile")
    weight = st.number_input("Weight (kg)", 40, 150, 70)
    height = st.number_input("Height (cm)", 120, 220, 175)
    st.markdown("---")
    st.subheader("Preferences")
    diet_pref = st.selectbox("Dietary Choice", ["Veg", "Non-Veg"])
    budget = st.slider("Budget/Meal (₹)", 10, 150, 60)
    protein = st.slider("Goal Protein (g)", 5, 60, 25)
    st.markdown("---")
    location = st.radio("Workout Context", ["Hostel/Home", "Gym"])

# --- 5. MAIN LOGIC & BMI CALCULATION ---
if model is None:
    st.warning("⚠️ Training Data not detected. Please run 'train_ai.py' first.")
    st.stop()

bmi = weight / ((height/100)**2)

# Dynamic rep and goal definition
if bmi < 18.5:
    goal_text = "Focus on Hypertrophy (Muscle Gain)"
    reps = "8-10 Reps"
    status = "Underweight"
elif 18.5 <= bmi <= 24.9:
    goal_text = "Focus on Strength & Maintenance"
    reps = "10-12 Reps"
    status = "Healthy Range"
else:
    goal_text = "Focus on Metabolic Conditioning (Fat Loss)"
    reps = "15-20 Reps"
    status = "Overweight"

# --- DYNAMIC EXERCISE LIST GENERATION ---
if bmi < 18.5: # Muscle Gain Protocol
    if location == "Gym":
        workout_list = ["Barbell Squats", "Incline Bench Press", "Deadlifts", "Lat Pulldowns", "Bicep Curls"]
    else:
        workout_list = ["Bulgarian Split Squats", "Diamond Pushups", "Doorway Rows", "Plank", "Bodyweight Lunges"]
elif 18.5 <= bmi <= 24.9: # Strength Protocol
    if location == "Gym":
        workout_list = ["Overhead Press", "Leg Press", "Seated Rows", "Chest Flys", "Tricep Extensions"]
    else:
        workout_list = ["Standard Pushups", "Air Squats", "Bicycle Crunches", "Pike Pushups", "Superman Extensions"]
else: # Fat Loss Protocol
    if location == "Gym":
        workout_list = ["Kettlebell Swings", "Rowing Machine", "Goblet Squats", "Burpees", "Mountain Climbers"]
    else:
        workout_list = ["Jumping Jacks", "Burpees", "Mountain Climbers", "High Knees", "Plank Jacks"]

st.markdown("<h1 class='main-header'>AI Fitness Strategy Lab</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b;'>Optimizing student nutrition and fitness through personalized AI planning.</p>", unsafe_allow_html=True)

# Metric Display
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Current BMI", f"{bmi:.1f}")
with m2:
    analysis_label = "Healthy Range" if status == "Healthy Range" else "Adjustment Advised"
    st.metric("Analysis", analysis_label, delta=status if status != "Healthy Range" else None, delta_color="inverse")
with m3:
    st.metric("Est. Daily Cost", f"₹{budget * 3}")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. MAIN CONTENT COLUMNS ---
col_diet, col_work = st.columns([1.6, 1])

with col_diet:
    st.subheader("📋 AI Nutritional Recommendations")
    if st.button("Run AI Recommendation Engine"):
        # Suppress feature name warning
        query_df = pd.DataFrame([[protein, budget]], columns=['Protein', 'Cost'])
        query = scaler.transform(query_df)
        
        dist, idx = model.kneighbors(query)
        results = df.iloc[idx[0]]
        matches = results[results['Type'] == diet_pref]
        
        if not matches.empty:
            # Building report for download
            report_text = f"PROFIT AI - PERSONALIZED FITNESS REPORT\n"
            report_text += f"BMI: {bmi:.1f} ({status})\n"
            report_text += f"Diet Preference: {diet_pref}\n"
            report_text += "="*40 + "\n\nNUTRITION PLAN:\n"

            for i, row in matches.iterrows():
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0;">{row['Food']}</h3>
                        <span style="background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:bold;">{row['Type']}</span>
                    </div>
                    <p style="margin:10px 0;">
                        <b>Nutritional Value:</b> {row['Protein']}g Protein<br>
                        <b>Financial Impact:</b> ₹{row['Cost']} per serving
                    </p>
                </div>
                """, unsafe_allow_html=True)
                report_text += f"- {row['Food']}: {row['Protein']}g Protein (₹{row['Cost']})\n"

            # Adding the Dynamic Workout to the report text
            report_text += f"\nWORKOUT PROTOCOL ({location}):\n"
            report_text += f"Target Goal: {goal_text}\n"
            report_text += f"Rep Range: {reps}\n"
            for ex in workout_list:
                report_text += f" [ ] {ex}\n"

            st.download_button(
                label="📥 Download Plan as Text",
                data=report_text,
                file_name="my_fitness_plan.txt",
                mime="text/plain"
            )
        else:
            st.info("No exact matches found. Try adjusting your protein/budget sliders.")

with col_work:
    st.subheader("🏋️ AI Exercise Protocol")
    st.markdown(f"""
    <div class="workout-box">
        <h4 style="color:#10b981; margin-top:0;">{location} Strategy</h4>
        <p style="font-size: 14px; color: #94a3b8; margin-bottom: 5px;"><b>AI Goal:</b> {goal_text}</p>
        <p style="font-size: 13px; color: #cbd5e1; font-style: italic;">Intensity: High (Based on {status})</p>
        <hr style="border-color:#334155; margin: 10px 0;">
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.8;">
            {"".join([f"<li>✅ {item} ({reps})</li>" for item in workout_list])}
        </ul>
        <hr style="border-color:#334155; margin: 10px 0;">
        <p style="font-size:11px; font-style:italic; color:#94a3b8;">
            AI Note: These exercises were selected specifically for an '{status}' profile to maximize effective body composition changes.
        </p>
    </div>
    """, unsafe_allow_html=True)