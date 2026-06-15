# AI-Powered Dietary Monitor Dashboard
# Run using: streamlit run app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tempfile
from pathlib import Path
from datetime import date

from src.utils.database import init_db, add_meal_to_db, get_all_meals_from_db, clear_all_meals_from_db, register_user, authenticate_user

# Page config
st.set_page_config(
    page_title="AI Dietary Monitor",
    page_icon="https://cdn-icons-png.flaticon.com/512/3448/3448065.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

DAILY_RDA = {
    "calories": 2000, "protein_g": 50, "carbs_g": 275, "fat_g": 78,
    "fiber_g": 28, "sugar_g": 50, "sodium_mg": 2300, "iron_mg": 18,
    "calcium_mg": 1000, "vitamin_c_mg": 90,
}
NUTRIENT_LABELS = {
    "calories": "Calories (kcal)", "protein_g": "Protein (g)",
    "carbs_g": "Carbs (g)", "fat_g": "Fat (g)", "fiber_g": "Fiber (g)",
    "sugar_g": "Sugar (g)", "sodium_mg": "Sodium (mg)", "iron_mg": "Iron (mg)",
    "calcium_mg": "Calcium (mg)", "vitamin_c_mg": "Vitamin C (mg)",
}

# Load custom styles
def inject_css():
    import textwrap
    st.html(textwrap.dedent("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    /* Hide Streamlit header anchor links */
    .element-container a.anchor, .element-container a {
        display: none !important;
    }
    
    /* Style only the login container card specifically */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#login_username_input),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#signup_username_input) {
        background: #111124 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
    }

    /* ── Base & background ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0A0A1A 0%, #0D0D2B 40%, #0A1628 100%);
        min-height: 100vh;
    }

    /* Limit image display size to keep page from scrolling excessively */
    [data-testid="stImage"] img, .stImage img {
        max-height: 250px !important;
        object-fit: cover !important;
        border-radius: 14px !important;
        width: auto !important;
        margin: 0 auto;
    }

    /* Responsive adjustments for mobile and tablets */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px !important;
        }
        .gradient-title {
            font-size: 1.8rem !important;
        }
        .hero-stat .value {
            font-size: 1.5rem !important;
        }
        [data-testid="metric-container"] {
            padding: 12px !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        .glass-card {
            padding: 16px !important;
        }
    }

    /* Animated gradient orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -20%;
        left: -10%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -20%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(6,182,212,0.05) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0d0d1f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }

    /* Card style */
    .glass-card {
        background: #111124;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    /* Metric cards styling */
    [data-testid="metric-container"] {
        background: #111124 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(124, 58, 237, 0.2) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.08) !important;
    }
    [data-testid="metric-container"] label {
        color: rgba(226,232,240,0.6) !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E2E8F0 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Gradient titles */
    .gradient-title {
        background: linear-gradient(135deg, #A78BFA 0%, #60A5FA 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .gradient-subtitle {
        background: linear-gradient(90deg, #7C3AED, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .section-header {
        color: #A78BFA;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(167,139,250,0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(124,58,237,0.4) !important;
        border-radius: 16px;
        padding: 20px;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(124,58,237,0.7) !important;
        background: rgba(124,58,237,0.05);
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7C3AED, #06B6D4) !important;
        border-radius: 999px;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 999px;
    }

    /* Hide Streamlit default input instructions ("Press Enter to apply") */
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Inputs and selects style wrapper */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        background: #111124 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
    }
    .stSelectbox > div > div *,
    .stTextInput > div > div *,
    .stNumberInput > div > div * {
        background-color: transparent !important;
        color: #E2E8F0 !important;
    }
    .stTextInput input,
    .stNumberInput input {
        background: transparent !important;
        border: none !important;
        color: #E2E8F0 !important;
        box-shadow: none !important;
        padding-right: 42px !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 14px !important;
        border: none !important;
        backdrop-filter: blur(10px);
    }
    .stSuccess { background: rgba(52,211,153,0.12) !important; border-left: 3px solid #34D399 !important; }
    .stWarning { background: rgba(251,191,36,0.10) !important; border-left: 3px solid #FBBF24 !important; }
    .stError   { background: rgba(239,68,68,0.10)  !important; border-left: 3px solid #EF4444  !important; }
    .stInfo    { background: rgba(6,182,212,0.10)   !important; border-left: 3px solid #06B6D4  !important; }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: #A78BFA !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }

    /* Radio buttons */
    .stRadio > label { color: #A78BFA !important; font-weight: 600; }
    .stRadio [data-baseweb="radio"] span { border-color: #7C3AED !important; }
    .stRadio [aria-checked="true"] span { background: #7C3AED !important; }

    /* Charts */
    .js-plotly-plot { border-radius: 16px; overflow: hidden; }

    /* Captions */
    .stCaption { color: rgba(226,232,240,0.45) !important; font-size: 0.78rem !important; }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }

    /* Badges */
    .badge {
        display: inline-block;
        background: rgba(124,58,237,0.2);
        border: 1px solid rgba(124,58,237,0.4);
        color: #A78BFA;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-cyan {
        background: rgba(6,182,212,0.15);
        border-color: rgba(6,182,212,0.35);
        color: #67E8F9;
    }
    .badge-green {
        background: rgba(52,211,153,0.15);
        border-color: rgba(52,211,153,0.35);
        color: #6EE7B7;
    }

    /* Hero stats */
    .hero-stat {
        text-align: center;
        padding: 20px;
        background: #111124;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        margin-bottom: 12px;
    }
    .hero-stat .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-stat .label {
        color: rgba(226,232,240,0.55);
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Rec cards */
    .rec-card {
        background: #111124;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .rec-card:hover {
        background: rgba(124, 58, 237, 0.05);
        border-color: rgba(124, 58, 237, 0.2);
        transform: translateY(-2px);
    }
    .rec-card .food-name {
        font-weight: 700;
        color: #6EE7B7;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .rec-card .score {
        color: rgba(226,232,240,0.6);
        font-size: 0.82rem;
    }

    /* Deficiency alerts */
    .deficiency-item {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        border-left: 3px solid #EF4444;
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 8px;
        color: #FCA5A5;
        font-weight: 500;
    }
    .anomaly-ok {
        background: rgba(52,211,153,0.05);
        border: 1px solid rgba(52,211,153,0.15);
        border-radius: 12px;
        padding: 10px 16px;
        color: #6EE7B7;
        font-weight: 500;
        text-align: center;
    }
    .anomaly-warn {
        background: rgba(251,191,36,0.05);
        border: 1px solid rgba(251,191,36,0.2);
        border-left: 3px solid #FBBF24;
        border-radius: 12px;
        padding: 10px 16px;
        color: #FCD34D;
        font-weight: 500;
    }
    </style>
    """))


inject_css()

# Plotly dark theme config
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#E2E8F0"),
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
)

# Load models
@st.cache_resource(show_spinner="Loading AI models...")
def load_pipeline():
    from src.inference.pipeline import DietaryPipeline
    return DietaryPipeline()

def get_pipeline():
    try:
        return load_pipeline()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return None

# Session state & database init
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "meal_log" not in st.session_state:
    st.session_state.meal_log = []

def load_user_meals():
    if st.session_state.logged_in and st.session_state.username:
        st.session_state.meal_log = get_all_meals_from_db(st.session_state.username)
    else:
        st.session_state.meal_log = []

# Load meals if logged in
if st.session_state.logged_in and not st.session_state.meal_log:
    load_user_meals()

def add_meal(food_name, meal_type, nutrition):
    if st.session_state.logged_in and st.session_state.username:
        today_str = str(date.today())
        add_meal_to_db(st.session_state.username, today_str, meal_type, food_name, nutrition)
        load_user_meals()

def show_login_page():
    # Centered title
    st.markdown("""
    <div style="text-align:center; margin-top:8vh; margin-bottom: 20px;">
        <h1 style="background:linear-gradient(135deg,#A78BFA,#60A5FA);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    font-size:2.8rem; font-weight:800; letter-spacing:-0.02em; margin-top: 10px;">
            AI Dietary Monitor
        </h1>
        <p style="color:rgba(226,232,240,0.55); font-size:1.1rem; margin-top:5px;">
            Your personal local health and nutrition analytics system
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])
            
            with tab_login:
                st.markdown("<br>", unsafe_allow_html=True)
                login_user = st.text_input("Username", key="login_username_input").strip()
                login_pass = st.text_input("Password", type="password", key="login_password_input")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Sign In", type="primary", use_container_width=True):
                    if not login_user or not login_pass:
                        st.error("Please fill in all fields.")
                    elif authenticate_user(login_user, login_pass):
                        st.session_state.logged_in = True
                        st.session_state.username = login_user
                        load_user_meals()
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
                        
            with tab_signup:
                st.markdown("<br>", unsafe_allow_html=True)
                signup_user = st.text_input("Choose Username", key="signup_username_input").strip()
                signup_pass = st.text_input("Choose Password", type="password", key="signup_password_input")
                signup_pass_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm_input")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Create Account", use_container_width=True):
                    if not signup_user or not signup_pass:
                        st.error("Please fill in all fields.")
                    elif signup_pass != signup_pass_confirm:
                        st.error("Passwords do not match.")
                    elif len(signup_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif register_user(signup_user, signup_pass):
                        st.success("Account created successfully! You can now sign in.")
                    else:
                        st.error("Username already exists. Please choose another one.")

# Run login screen if not authenticated
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

def get_today_totals():
    today = str(date.today())
    totals = {k: 0.0 for k in DAILY_RDA}
    for meal in st.session_state.meal_log:
        if meal.get("date") == today:
            for k in DAILY_RDA:
                totals[k] += meal.get(k, 0.0)
    return totals

def get_meal_icon(meal_type):
    mt = str(meal_type).lower()
    if mt == "breakfast":
        return '<i class="fa-solid fa-mug-hot" style="color:#FBBF24; margin-right:8px;"></i>'
    elif mt == "lunch":
        return '<i class="fa-solid fa-sun" style="color:#F59E0B; margin-right:8px;"></i>'
    elif mt == "dinner":
        return '<i class="fa-solid fa-moon" style="color:#A78BFA; margin-right:8px;"></i>'
    else: # snack
        return '<i class="fa-solid fa-apple-whole" style="color:#EF4444; margin-right:8px;"></i>'

# Sidebar setup
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <div style="font-size:2.5rem; color:#A78BFA;"><i class="fa-solid fa-bowl-food"></i></div>
        <div style="background:linear-gradient(135deg,#A78BFA,#60A5FA);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    font-size:1.2rem; font-weight:800; letter-spacing:-0.01em;">
            AI Dietary Monitor
        </div>
        <div style="color:rgba(226,232,240,0.4); font-size:0.72rem; margin-top:4px; letter-spacing:0.05em;">
            HEALTH ANALYTICS SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07); margin:12px 0'>", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "Food Recognition",
        "Daily Dashboard",
        "Health Analytics",
        "Meal History",
    ])

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07); margin:16px 0'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 4px 0;">
        <div style="margin-bottom:8px;">
            <span class="badge">EfficientNet-B3</span>
            <span style="color:rgba(226,232,240,0.5); font-size:0.75rem; margin-left:6px;">84.3% Top-1</span>
        </div>
        <div style="margin-bottom:8px;">
            <span class="badge badge-cyan">181 Classes</span>
            <span style="color:rgba(226,232,240,0.5); font-size:0.75rem; margin-left:6px;">Food-101 + Indian</span>
        </div>
        <div style="margin-bottom:8px;">
            <span class="badge badge-green">RandomForest</span>
            <span style="color:rgba(226,232,240,0.5); font-size:0.75rem; margin-left:6px;">Deficiency</span>
        </div>
        <div style="margin-bottom:8px;">
            <span class="badge badge-cyan">LSTM</span>
            <span style="color:rgba(226,232,240,0.5); font-size:0.75rem; margin-left:6px;">Trend Forecast</span>
        </div>
        <div style="margin-bottom:16px;">
            <span class="badge">IsolationForest</span>
            <span style="color:rgba(226,232,240,0.5); font-size:0.75rem; margin-left:6px;">Anomaly</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07); margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); 
                border-radius:12px; padding:10px 14px; margin-bottom:12px; text-align:center;">
        <div style="color:rgba(226,232,240,0.5); font-size:0.75rem; text-transform:uppercase; font-weight:500;">
            User Profile
        </div>
        <div style="font-weight:700; color:#A78BFA; font-size:1.0rem; margin-top:4px;">
            <i class="fa-solid fa-user" style="margin-right:6px;"></i>{st.session_state.username.title()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.meal_log = []
        st.rerun()

# Page 1: Food Recognition
if page == "Food Recognition":
    st.markdown('<div class="gradient-title">Food Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:rgba(226,232,240,0.55); margin-bottom:28px;">'
        'Upload a food photo — EfficientNet-B3 identifies it and estimates macronutrients instantly.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

            if st.button("Identify Food", type="primary", use_container_width=True):
                pipeline = get_pipeline()
                if pipeline:
                    with st.spinner("Running model inference..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            image.save(tmp.name)
                            result = pipeline.predict_food(tmp.name)
                    st.session_state["last_result"] = result
                    st.session_state["last_meal_type"] = meal_type.lower()

    with col2:
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            food_label = result["predicted_food"].replace("_", " ").title()
            confidence = result["confidence"] * 100

            st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="hero-stat">
                <div class="value">{food_label}</div>
                <div class="label">Identified Food</div>
            </div>
            """, unsafe_allow_html=True)

            conf_color = "#34D399" if confidence >= 70 else "#FBBF24" if confidence >= 40 else "#EF4444"
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:20px;">
                <span style="font-size:2rem; font-weight:800; color:{conf_color};">{confidence:.1f}%</span>
                <span style="color:rgba(226,232,240,0.5); font-size:0.85rem; margin-left:8px;">confidence</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Top 5 Predictions</div>', unsafe_allow_html=True)
            for i, p in enumerate(result["top_predictions"], 1):
                pct = p["confidence"] * 100
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#E2E8F0; font-size:0.85rem;">
                        <span style="color:#7C3AED; font-weight:700;">#{i}</span>
                        &nbsp;{p['food'].replace('_',' ').title()}
                    </span>
                    <span style="color:#A78BFA; font-size:0.85rem; font-weight:600;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(p["confidence"])

            st.markdown('<div class="section-header" style="margin-top:20px;">Nutrition Estimate</div>', unsafe_allow_html=True)
            n = result["nutrition"]
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("Calories", f"{n.get('calories',0):.0f} kcal")
            nc2.metric("Protein", f"{n.get('protein_g',0):.1f} g")
            nc3.metric("Carbs", f"{n.get('carbs_g',0):.1f} g")
            nc4, nc5, _ = st.columns(3)
            nc4.metric("Fat", f"{n.get('fat_g',0):.1f} g")
            nc5.metric("Fiber", f"{n.get('fiber_g',0):.1f} g")

            if st.button("Add to Meal Log", use_container_width=True):
                add_meal(result["predicted_food"], st.session_state["last_meal_type"], n)
                st.success(f"Added **{food_label}** to your meal log!")

# Page 2: Daily Dashboard
elif page == "Daily Dashboard":
    today_str = date.today().strftime("%A, %B %d %Y")
    st.markdown('<div class="gradient-title">Daily Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:rgba(226,232,240,0.55); margin-bottom:28px;">{today_str}</div>',
        unsafe_allow_html=True,
    )

    totals = get_today_totals()
    meals_today = [m for m in st.session_state.meal_log if m.get("date") == str(date.today())]

    # Hero metric display
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Meals Logged", len(meals_today))
    cal_pct = totals["calories"] / DAILY_RDA["calories"] * 100
    hc2.metric("Calories", f"{totals['calories']:.0f}", f"{cal_pct:.0f}% of RDA")
    hc3.metric("Protein", f"{totals['protein_g']:.1f}g", f"/ {DAILY_RDA['protein_g']}g RDA")
    hc4.metric("Carbs", f"{totals['carbs_g']:.1f}g", f"/ {DAILY_RDA['carbs_g']}g RDA")

    st.markdown("<br>", unsafe_allow_html=True)

    if not meals_today:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:3rem; margin-bottom:12px; color:#7C3AED;"><i class="fa-solid fa-utensils"></i></div>
            <div style="color:#A78BFA; font-size:1.1rem; font-weight:600;">No meals logged today</div>
            <div style="color:rgba(226,232,240,0.45); margin-top:8px;">
                Use Food Recognition to log your first meal!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([1.4, 1], gap="large")

        with col1:
            st.markdown('<div class="section-header">Nutrients vs Daily RDA</div>', unsafe_allow_html=True)
            nutrients = list(DAILY_RDA.keys())
            pct_values = [min(totals[n] / DAILY_RDA[n] * 100, 150) for n in nutrients]
            colors = [
                "#EF4444" if p < 50 else
                "#FBBF24" if p < 80 else
                "#34D399" if p <= 110 else
                "#F97316"
                for p in pct_values
            ]

            fig = go.Figure(go.Bar(
                x=pct_values,
                y=[NUTRIENT_LABELS[n] for n in nutrients],
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(255,255,255,0.05)", width=1),
                ),
                text=[f"{p:.0f}%" for p in pct_values],
                textposition="outside",
                textfont=dict(color="#E2E8F0", size=11),
            ))
            fig.add_vline(x=100, line_dash="dash", line_color="rgba(167,139,250,0.6)", line_width=1.5)
            fig.add_vline(x=70, line_dash="dot", line_color="rgba(251,191,36,0.5)", line_width=1)
            fig.update_layout(
                height=380,
                xaxis_title="% of Daily RDA",
                showlegend=False,
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                margin=dict(l=10, r=70, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            p_cal = totals["protein_g"] * 4
            c_cal = totals["carbs_g"] * 4
            f_cal = totals["fat_g"] * 9
            total_cal = p_cal + c_cal + f_cal

            if total_cal > 0:
                st.markdown('<div class="section-header">Macro Split</div>', unsafe_allow_html=True)
                fig2 = go.Figure(go.Pie(
                    labels=["Protein", "Carbs", "Fat"],
                    values=[p_cal, c_cal, f_cal],
                    hole=0.6,
                    marker=dict(
                        colors=["#7C3AED", "#06B6D4", "#F97316"],
                        line=dict(color="rgba(0,0,0,0)", width=0),
                    ),
                    textinfo="label+percent",
                    textfont=dict(color="#E2E8F0", size=12),
                ))
                fig2.update_layout(
                    height=240,
                    showlegend=False,
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-header">Today\'s Meals</div>', unsafe_allow_html=True)
            for m in meals_today:
                icon_html = get_meal_icon(m.get("meal_type", ""))
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                            border-radius:12px; padding:12px 14px; margin-bottom:8px;">
                    <div style="font-weight:600; color:#E2E8F0;">
                        {icon_html}{m['food'].replace('_',' ').title()}
                    </div>
                    <div style="color:rgba(226,232,240,0.45); font-size:0.78rem; margin-top:4px;">
                        {m.get('calories',0):.0f} kcal &nbsp;·&nbsp;
                        P: {m.get('protein_g',0):.0f}g &nbsp;·&nbsp;
                        C: {m.get('carbs_g',0):.0f}g &nbsp;·&nbsp;
                        F: {m.get('fat_g',0):.0f}g
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Manual input form
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Add Meal Manually"):
        mc1, mc2 = st.columns(2)
        with mc1:
            food_name = st.text_input("Food name")
            mt = st.selectbox("Meal type", ["breakfast", "lunch", "dinner", "snack"])
            cal = st.number_input("Calories (kcal)", 0.0, 5000.0, 0.0)
            pro = st.number_input("Protein (g)", 0.0, 300.0, 0.0)
            crb = st.number_input("Carbs (g)", 0.0, 500.0, 0.0)
        with mc2:
            fat = st.number_input("Fat (g)", 0.0, 200.0, 0.0)
            fib = st.number_input("Fiber (g)", 0.0, 100.0, 0.0)
            sug = st.number_input("Sugar (g)", 0.0, 200.0, 0.0)
            sod = st.number_input("Sodium (mg)", 0.0, 5000.0, 0.0)
        if st.button("Add Meal", use_container_width=True):
            if food_name:
                add_meal(food_name, mt, {
                    "calories": cal, "protein_g": pro, "carbs_g": crb, "fat_g": fat,
                    "fiber_g": fib, "sugar_g": sug, "sodium_mg": sod,
                    "iron_mg": 0, "calcium_mg": 0, "vitamin_c_mg": 0,
                })
                st.success(f"Added **{food_name}**!")
                st.rerun()

# Page 3: Health Analytics
elif page == "Health Analytics":
    st.markdown('<div class="gradient-title">Health Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:rgba(226,232,240,0.55); margin-bottom:28px;">'
        'Powered by RandomForest · LSTM · Isolation Forest'
        '</div>',
        unsafe_allow_html=True,
    )

    meals = st.session_state.meal_log
    if len(meals) < 3:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:3rem; margin-bottom:12px; color:#FBBF24;"><i class="fa-solid fa-brain"></i></div>
            <div style="color:#FBBF24; font-size:1.1rem; font-weight:600;">Not enough data yet</div>
            <div style="color:rgba(226,232,240,0.45); margin-top:8px;">
                Log at least 3 meals to enable health analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(meals)
        for col in DAILY_RDA:
            if col not in df.columns:
                df[col] = 0.0
        daily = df.groupby("date")[list(DAILY_RDA.keys())].sum().reset_index()
        daily_records = daily[list(DAILY_RDA.keys())].values.astype(np.float32)

        pipeline = get_pipeline()
        if pipeline:
            with st.spinner("Running health models..."):
                health = pipeline.analyze_health(daily_records)

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown('<div class="section-header">Deficiency Detection</div>', unsafe_allow_html=True)
                st.caption("RandomForest — trained on 24,000 synthetic daily logs")
                deficient = health["deficient_nutrients"]
                if deficient:
                    for n in deficient:
                        st.markdown(
                            f'<div class="deficiency-item"><i class="fa-solid fa-triangle-exclamation" style="margin-right:8px;"></i> Low {NUTRIENT_LABELS.get(n, n)}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="anomaly-ok"><i class="fa-solid fa-circle-check" style="margin-right:8px;"></i> No deficiencies detected — great job!</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
                st.caption("Isolation Forest — flags unusual eating patterns")
                if health["is_anomaly"]:
                    st.markdown(
                        '<div class="anomaly-warn"><i class="fa-solid fa-bolt" style="margin-right:8px;"></i> Unusual eating pattern detected!</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="anomaly-ok"><i class="fa-solid fa-circle-check" style="margin-right:8px;"></i> Eating patterns look normal</div>',
                        unsafe_allow_html=True,
                    )

            with col2:
                st.markdown('<div class="section-header">Tomorrow\'s Nutrient Forecast</div>', unsafe_allow_html=True)
                st.caption("LSTM — 14-day rolling window predictor")
                pred = health["predicted_tomorrow"]
                forecast_data = {
                    "Nutrient": [NUTRIENT_LABELS[k] for k in DAILY_RDA],
                    "Predicted": [max(0, pred.get(k, 0)) for k in DAILY_RDA],
                    "RDA": list(DAILY_RDA.values()),
                }
                fdf = pd.DataFrame(forecast_data)
                fdf["pct"] = (fdf["Predicted"] / fdf["RDA"] * 100).clip(0, 140)

                fig = go.Figure(go.Bar(
                    x=fdf["Nutrient"],
                    y=fdf["pct"],
                    marker=dict(
                        color=fdf["pct"],
                        colorscale=[[0, "#EF4444"], [0.6, "#FBBF24"], [1, "#34D399"]],
                        cmin=0, cmax=120,
                        line=dict(color="rgba(255,255,255,0.05)", width=1),
                    ),
                    text=[f"{p:.0f}%" for p in fdf["pct"]],
                    textposition="outside",
                    textfont=dict(color="#E2E8F0", size=10),
                ))
                fig.add_hline(y=100, line_dash="dash", line_color="rgba(167,139,250,0.6)", line_width=1.5)
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_tickangle=-45,
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                    margin=dict(l=10, r=10, t=20, b=80),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Recommended Foods to Fill Gaps</div>', unsafe_allow_html=True)
            st.caption("Content-based recommender — cosine similarity on nutrition vectors")
            totals = get_today_totals()
            recs = pipeline.get_recommendations(
                deficient_nutrients=deficient or list(DAILY_RDA.keys())[:3],
                consumed_today=totals, top_k=6,
            )
            rec_cols = st.columns(3)
            for i, rec in enumerate(recs[:6]):
                with rec_cols[i % 3]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="food-name">{rec['food'].replace('_',' ').title()}</div>
                        <div class="score">Match Score: {rec['score']*100:.0f}%</div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

# Page 4: Meal History
elif page == "Meal History":
    st.markdown('<div class="gradient-title">Meal History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:rgba(226,232,240,0.55); margin-bottom:28px;">'
        'Your complete food diary and calorie trends.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.meal_log:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:3rem; margin-bottom:12px; color:#A78BFA;"><i class="fa-solid fa-clipboard-list"></i></div>
            <div style="color:#A78BFA; font-size:1.1rem; font-weight:600;">No meals logged yet</div>
            <div style="color:rgba(226,232,240,0.45); margin-top:8px;">
                Start logging meals using Food Recognition.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(st.session_state.meal_log)
        for col in DAILY_RDA:
            if col not in df.columns:
                df[col] = 0.0

        st.markdown('<div class="section-header">Calorie Trend</div>', unsafe_allow_html=True)
        daily = df.groupby("date")["calories"].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["calories"],
            mode="lines+markers",
            line=dict(color="#7C3AED", width=2.5),
            marker=dict(color="#A78BFA", size=8, line=dict(color="#7C3AED", width=2)),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.08)",
            name="Calories",
        ))
        fig.add_hline(y=DAILY_RDA["calories"], line_dash="dash", line_color="rgba(239,68,68,0.6)",
                      line_width=1.5, annotation_text="RDA Target",
                      annotation_font_color="rgba(239,68,68,0.8)")
        fig.update_layout(
            height=260,
            xaxis_title="Date",
            yaxis_title="Calories (kcal)",
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">All Meals</div>', unsafe_allow_html=True)
        show = df[["date", "meal_type", "food", "calories", "protein_g", "carbs_g", "fat_g"]].copy()
        show["food"] = show["food"].str.replace("_", " ").str.title()
        show.columns = ["Date", "Meal", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]
        st.dataframe(show, use_container_width=True, hide_index=True)

        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("Clear All Meals"):
                clear_all_meals_from_db(st.session_state.username)
                st.session_state.meal_log = []
                st.rerun()
