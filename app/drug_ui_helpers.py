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
        margin-bottom: 5rem;
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


# ============================================================================
#  DATA LOADING
# ============================================================================

def load_precomputed_results():
    """Load precomputed analysis results from JSON"""
    results_path = Path("data/precomputed_samples.json")

    if not results_path.exists():
        st.error("⚠️ Precomputed results file not found")
        return None

    try:
        with open(results_path, 'r') as f:
            data = json.load(f)

        # Index results by scenario number
        results_dict = {item['scenario']: item['analysis'] for item in data}
        return results_dict

    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None


# ============================================================================
#  DISPLAY COMPONENTS
# ============================================================================

def display_hero_section():
    """Display hero header"""
    st.markdown("""
    <div class="hero-header">
        <h1>💊 Drug Interaction Checker</h1>
        <p>AI-Powered Clinical Decision Support</p>
        <p><strong>by Ridwan Oladipo, MD | Medical AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)


def display_scenario_grid(scenarios):
    """Display clinical scenario selection grid"""
    st.markdown("### 🔍 Select a Clinical Scenario")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
                border: 1px solid #a5b4fc; color: #1e3a8a;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 0.6rem; font-size: 0.9rem;">
    💡 <em>Demo subset — production version supports real-time polypharmacy analysis for any drug combination.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='margin-top:0.4rem;'>Click any scenario below to analyze drug interactions</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    cols = st.columns(2)

    for idx, (title, query) in enumerate(scenarios):
        with cols[idx % 2]:
            if st.button(
                    f"📋 {title}",
                    key=f"scenario_{idx}",
                    use_container_width=True,
                    help=query
            ):
                st.session_state.selected_scenario = idx
                st.session_state.show_user_message = True
                st.session_state.show_check_button = True
                st.session_state.show_ai_response = False
                st.rerun()

            st.markdown(
                f'<div style="font-size: 0.9rem; color: #374151; text-align: center; '
                f'padding: 0.4rem 0.6rem; background: #f9fafb; border-radius: 8px; '
                f'margin-top: 0.1rem; margin-bottom: 2.4rem; '
                f'box-shadow: 0 1px 4px rgba(0,0,0,0.05); word-wrap: break-word;">'
                f'"{query}"</div>',
                unsafe_allow_html=True
            )


def display_user_message(title, query):
    """Display user message bubble"""
    st.markdown(f"""
    <div class="user-message">
        <div class="user-message-title">{title}</div>
        <div class="user-message-text">"{query}"</div>
    </div>
    """, unsafe_allow_html=True)


def stream_text(text, delay=0.02):
    """Stream text character by character"""
    placeholder = st.empty()
    displayed_text = ""

    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(delay)


def display_ai_response(analysis, stream=True):
    """Display AI response with streaming"""

    st.markdown('<div class="ai-message">', unsafe_allow_html=True)

    # Show analysis indicator
    if stream and "check_btn" in st.session_state:
        with st.spinner("💊 Analyzing interactions..."):
            time.sleep(3.5)

    # Overall Risk Badge
    overall = analysis['overall_assessment']
    flag = overall['overall_flag']
    risk_level = overall['risk_level']

    if flag == '🟥':
        badge_class = 'risk-high'
    elif flag == '🟨':
        badge_class = 'risk-moderate'
    else:
        badge_class = 'risk-low'

    st.markdown(
        f'<div class="risk-badge {badge_class}">{flag} {risk_level}</div>',
        unsafe_allow_html=True
    )

    # Clinical Assessment
    st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 CLINICAL ASSESSMENT</div>', unsafe_allow_html=True)

    if stream and "check_btn" in st.session_state:
        stream_text(overall['clinical_synthesis'], delay=0.01)
    else:
        st.markdown(f'<div class="section-content">{overall["clinical_synthesis"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Cumulative Concerns
    if overall.get('cumulative_concerns'):
        st.markdown(
            f'<div class="alert-box">'
            f'<span style="font-size: 1.2rem; margin-right: 0.5rem;">⚠️</span>'
            f'<strong>Cumulative Concerns:</strong> ',
            unsafe_allow_html=True
        )
        if stream and "check_btn" in st.session_state:
            stream_text(overall['cumulative_concerns'], delay=0.02)
        else:
            st.markdown(overall['cumulative_concerns'])
        st.markdown('</div>', unsafe_allow_html=True)

    # Action Required
    if overall.get('action_required'):
        st.markdown(
            f'<div class="alert-box" style="border-left-color: #dc2626;">'
            f'<span style="font-size: 1.2rem; margin-right: 0.5rem;">🚨</span>'
            f'<strong>Action Required:</strong> ',
            unsafe_allow_html=True
        )
        if stream and "check_btn" in st.session_state:
            stream_text(overall['action_required'], delay=0.02)
        else:
            st.markdown(overall['action_required'])
        st.markdown('</div>', unsafe_allow_html=True)

    # Expandable Details
    with st.expander("📋 See Detailed Pair-by-Pair Analysis"):
        display_pair_analysis(analysis['pair_analyses'])

    st.markdown('</div>', unsafe_allow_html=True)


def display_pair_analysis(pair_analyses):
    """Display pair-by-pair breakdown"""
    st.markdown(f"**Analyzed {len(pair_analyses)} drug pair(s)**")

    for pair in pair_analyses:
        flag = pair['flag']

        if flag == '🟥':
            border_class = 'red-border'
        elif flag == '🟨':
            border_class = 'yellow-border'
        else:
            border_class = 'green-border'

        st.markdown(f'<div class="pair-card {border_class}">', unsafe_allow_html=True)

        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem;">'
            f'<span style="font-size: 1.5rem;">{flag}</span>'
            f'<span style="font-weight: 600; font-size: 1.1rem;">{pair["query"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">'
            f'Tier {pair["tier"]} | Confidence: {pair["confidence"] * 100:.0f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div style="font-size: 1rem; line-height: 1.6; color: #374151;">{pair["reasoning"]}</div>',
            unsafe_allow_html=True
        )

        if pair.get('monitoring'):
            st.markdown(
                f'<div style="margin-top: 0.5rem; padding: 0.5rem; background: #f9fafb; '
                f'border-radius: 6px; font-size: 0.9rem;">'
                f'<strong>Monitor:</strong> {pair["monitoring"]}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)


def display_footer():
    """Display footer"""
    html = """
<div class="medical-footer">
  <p style="font-size:0.95rem; opacity:0.9; margin-bottom:0.8rem;">
    Powered by DrugBank · RxNorm · OpenAI GPT-5 · AWS Bedrock
  </p>
  <div style="margin-top:0.3rem;">
    <a href="https://github.com/dr-ridwanoladipo/drug-interaction-ai#readme" target="_blank"
       style="color:#a5b4fc; text-decoration:none; margin-right:1.2rem; font-weight:500;">💻 GitHub</a>
    <a href="https://huggingface.co/spaces/dr-ridwanoladipo/drug-interaction-api" target="_blank"
       style="color:#a5b4fc; text-decoration:none; font-weight:500;">🔗 Live API Demo</a>
  </div>
  <p style="margin-top:1rem; font-size:0.95rem; color:rgba(255,255,255,0.85);">
    © 2025 Ridwan Oladipo, MD — Medical AI Specialist
  </p>
  <p style="margin-top:0.5rem; font-size:0.8rem; opacity:0.8;">
    ⚠️ Built to FDA-grade standards for clinical deployment.  
    All medical decisions should be made in consultation with qualified healthcare providers.
  </p>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with system information"""
    with st.sidebar:
        st.markdown("### ⚙️ System Overview")
        st.markdown("""
        **Production-Grade RAG Clinical Intelligence**

        - 170K+ curated DrugBank pairs  
        - RxNorm-standardized mapping (77K brand→ingredient)  
        - GPT-5 clinical reasoning + rule logic  
        - 3-Tier safety confidence (Direct ▸ Semantic ▸ None)
        """)
        st.markdown(
            "<p style='font-size:0.85rem; color:rgba(255,255,255,0.8); font-style:italic; "
            "margin-top:-0.5rem;'>Tier 1: Direct KB match | Tier 2: Semantic similarity | "
            "Tier 3: No evidence</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("### 🧠 AI Stack")
        st.markdown("""
        - **LLM:** OpenAI **GPT-5**  
        - **Embeddings:** OpenAI *text-embedding-3-large (3072-dim)*  
          + AWS **Bedrock Titan V2 (512-dim)** benchmark  
        - **Retrieval:** FAISS similarity search  
        - **Pipeline:** Polypharmacy RAG + severity classifier
        """)

        st.markdown("---")

        st.markdown("### 💡 Key Features")
        st.markdown("""
        - Dual-embedding benchmarking (OpenAI vs AWS)  
        - Clinical-grade safety synthesis (🟥 / 🟨 / 🟩)  
        - Confidence-weighted reasoning  
        - Hospital-ready CDSS integration
        """)


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


def reset_chat():
    """Reset chat state"""
    st.session_state.selected_scenario = None
    st.session_state.show_user_message = False
    st.session_state.show_check_button = False
    st.session_state.show_ai_response = False