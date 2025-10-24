"""
💊 Drug Interaction Checker - Main Application
AI-Powered Clinical Decision Support System

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

import streamlit as st
from drug_ui_helpers import (
    load_custom_css, initialize_session_state,
    load_precomputed_results, display_hero_section,
    render_sidebar, display_scenario_grid,
    display_user_message
)


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
load_precomputed_results()


# ============================================================================
# CLINICAL SCENARIOS
# ============================================================================
SCENARIOS = [
    ("Hypertensive patient on anticoagulation",
     "Can I safely combine Warfarin, Lisinopril and Aspirin?"),

    ("Post-MI patient with pain",
     "Is it safe to take Clopidogrel, Aspirin and Diclofenac together?"),

    ("Type 2 Diabetes with dyslipidemia",
     "Can I take Metformin with Atorvastatin?"),

    ("Type 2 Diabetic patient with hypertension",
     "Can I take Metformin with Lisinopril and Hydrochlorothiazide?"),

    ("Depression with chronic pain",
     "Is Sertraline safe with Tramadol and Ibuprofen?"),

    ("Rheumatoid arthritis with gastroprotection",
     "Can I combine Methotrexate, Prednisone and Omeprazole?"),

    ("Breast cancer chemotherapy support",
     "Is Tamoxifen safe with Ondansetron?"),

    ("Atrial fibrillation with antiplatelet therapy",
     "Is Warfarin safe with Aspirin?"),

    ("Elderly patient with polypharmacy",
     "Can I take Warfarin with Omeprazole?"),

    ("Cold & flu relief (OTC combo)",
     "Is it safe to take Tylenol with Benadryl?"),
]


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    render_sidebar()
    display_hero_section()

    # Display scenario selection grid
    display_scenario_grid(SCENARIOS)

    # Display user message for selected scenario
    if st.session_state.selected_scenario is not None:
        idx = st.session_state.selected_scenario
        title, query = SCENARIOS[idx]
        display_user_message(title, query)


if __name__ == "__main__":
    main()
