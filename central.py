import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from datetime import datetime
from models.data_models import *
from data_loader import DataLoader
import config 

st.set_page_config(
    page_title="🥛 Dairy Production Scheduler",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4169E1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


