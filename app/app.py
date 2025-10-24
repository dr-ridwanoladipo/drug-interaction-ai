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
    display_user_message, stream_text,
    display_ai_response, display_pair_analysis,
    reset_chat, display_footer
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

# Load precomputed results and store in session
results = load_precomputed_results()

if results is None or len(results) == 0:
    st.error("🚨 Failed to load drug interaction data. Please verify data source.")
    st.stop()

st.session_state.precomputed_results = results


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

    # Handle user selection
    if st.session_state.selected_scenario is not None:
        idx = st.session_state.selected_scenario
        title, query = SCENARIOS[idx]

        # Display user message
        display_user_message(title, query)

        # Interaction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Check Interaction", use_container_width=True):
                st.session_state.show_ai_response = True
                st.rerun()

        # Display AI response and pair analysis
        if st.session_state.get("show_ai_response", False):
            if "precomputed_results" not in st.session_state:
                st.error("❌ No precomputed data found. Please reload the app.")
                return

            analysis = st.session_state.precomputed_results.get(idx)
            if analysis:
                st.markdown("---")
                display_ai_response(analysis, stream=True)
                display_pair_analysis(analysis.get('pair_analyses', []))
            else:
                st.warning(f"⚠️ No analysis data found for: **{title}**")

            # Back button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("← Back to Scenarios", use_container_width=True):
                    reset_chat()
                    st.rerun()

    # ============================================================================
    # FOOTER
    # ============================================================================
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()
