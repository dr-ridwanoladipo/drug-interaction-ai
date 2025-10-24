"""
💊 Drug Interaction Checker - UI Helper Functions
Complete UI system with chat interface, streaming, and clinical display

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

import streamlit as st
import json
import time
from pathlib import Path


# ============================================================================
#  CUSTOM CSS - PROFESSIONAL MEDICAL + CHAT INTERFACE
# ============================================================================

def load_custom_css():
    """Load custom CSS for chat-based medical interface"""
    st.markdown("""
    <style>
    /* Import medical-grade fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu, footer, .stAppDeployButton {display: none !important;}

    .stApp {
        background: #f8fafc;
    }

    /* Page layout */
    .block-container {
        padding-top: 2.7rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }

    .hero-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .hero-header p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin: 0.5rem 0;
    }

    /* Scenario Grid */
    .scenario-grid {
        margin: 2rem 0;
    }

    /* Chat Interface */
    .chat-container {
        max-width: 900px;
        margin: 2rem auto;
        padding-bottom: 0.5rem !important;
    }

    .user-message {
        background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%);
        color: #1e3a8a;
        padding: 1.2rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        margin-left: auto;
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        animation: slideIn 0.3s ease;
    }

    .user-message-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    .user-message-text {
        font-size: 1.1rem;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
    }

    .risk-high {
        background: #fee2e2;
        color: #dc2626;
    }

    .risk-moderate {
        background: #fef3c7;
        color: #d97706;
    }

    .risk-low {
        background: #d1fae5;
        color: #059669;
    }

    .section-label {
        font-weight: 700;
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .section-content {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #1f2937;
    }

    .alert-box {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Check Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 15px 40px !important;
        border-radius: 25px !important;
        border: none;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.5rem 0;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }

    /* Footer */
    .medical-footer {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
    }

    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4C63D2 0%, #1E293B 100%) !important;
        backdrop-filter: blur(8px);
        box-shadow: inset -2px 0 8px rgba(0,0,0,0.25);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 500;
    }

    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_scenario' not in st.session_state:
        st.session_state.selected_scenario = None
    if 'show_user_message' not in st.session_state:
        st.session_state.show_user_message = False
    if 'show_check_button' not in st.session_state:
        st.session_state.show_check_button = False
    if 'show_ai_response' not in st.session_state:
        st.session_state.show_ai_response = False