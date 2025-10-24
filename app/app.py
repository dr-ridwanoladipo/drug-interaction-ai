"""
💊 Drug Interaction Checker - Main Application
AI-Powered Clinical Decision Support System

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

import streamlit as st
from drug_ui_helpers import *

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
    # Page configuration
    st.set_page_config(
        page_title="Drug Interaction Checker - AI Powered",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load CSS
    load_custom_css()

    # Initialize session state
    initialize_session_state()

    # Load precomputed results
    results = load_precomputed_results()

    if results is None:
        st.error("Failed to load analysis results")
        return

    # Render sidebar
    render_sidebar()

    # Display hero section
    display_hero_section()

    # Display clinical scenario grid
    display_scenario_grid(SCENARIOS)

    # Footer
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()
