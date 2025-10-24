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

    render_sidebar()

    # Display hero section
    display_hero_section()

    # ========================================================================
    # CHAT INTERFACE
    # ========================================================================

    if st.session_state.selected_scenario is None:
        # Display scenario selection grid
        display_scenario_grid(SCENARIOS)

    else:
        # Get selected scenario
        idx = st.session_state.selected_scenario
        title, query = SCENARIOS[idx]

        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display user message
        if st.session_state.show_user_message:
            display_user_message(title, query)

        # Display check button
        if st.session_state.show_check_button and not st.session_state.show_ai_response:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Check Interaction", key="check_btn", use_container_width=True):
                    st.session_state.show_ai_response = True
                    st.rerun()

        # Display AI response
        if st.session_state.show_ai_response:
            analysis = results.get(idx)

            if analysis:
                st.markdown("---")
                display_ai_response(analysis, stream=True)
            else:
                st.error(f"Analysis not found for: {title}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Back button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("← Back to Scenarios", use_container_width=True):
                if "check_btn" in st.session_state:
                    del st.session_state["check_btn"]
                reset_chat()
                st.rerun()

    # Display footer
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()