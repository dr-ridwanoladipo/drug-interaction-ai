"""
💊 Drug Interaction Checker - Main Application
AI-Powered Clinical Decision Support System

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

import streamlit as st
from drug_ui_helpers import load_custom_css, initialize_session_state


# ============================================================================
# INITIALIZATION
# ============================================================================
st.set_page_config(
    page_title="AI Drug Interaction Checker",
    page_icon="💊",
    layout="wide"
)

initialize_session_state()
load_custom_css()


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("AI Drug Interaction Checker")
    st.write("This application provides intelligent analysis of potential "
             "drug–drug interactions and therapeutic guidance for clinicians.")


if __name__ == "__main__":
    main()
