"""
app.py
Streamlit Web Application: Autonomous Text-to-SQL Agent
Theme: Dark AI / Futuristic Dashboard
Built for Elevvo Internship & Streamlit Community Cloud Deployment
"""

import os
import sys
import pandas as pd
import streamlit as st

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import AVAILABLE_MODELS, DEFAULT_MODEL_NAME, get_groq_api_key
from src.database import check_database_connection, DEFAULT_DB_PATH
from src.agent import run_agent_workflow, MAX_RETRIES
from src.visualization import create_chart, detect_chart_suitability

# 1. Page Configuration
st.set_page_config(
    page_title="Autonomous Text-to-SQL Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Dark AI / Futuristic Theme
CUSTOM_CSS = """
<style>
/* Overall Theme Variables */
:root {
    --bg-dark: #0b0f19;
    --card-bg: #111827;
    --card-border: #1f2937;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-glow: rgba(139, 92, 246, 0.2);
    --text-main: #f9fafb;
    --text-sub: #9ca3af;
}

/* Base Body Adjustments */
.stApp {
    background-color: var(--bg-dark);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Top Hero Header Card */
.hero-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(30, 41, 59, 0.85));
    border: 1px solid #374151;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 20px var(--accent-glow);
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c084fc, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-sub);
    margin-bottom: 14px;
}

/* Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-purple { background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-cyan { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }
.badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Sidebar Card */
.sidebar-section {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
}

/* Self-Correction Banner */
.correction-banner {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 16px;
    color: #fef3c7;
}

/* Metric / Stat Mini Card */
.stat-pill {
    background: rgba(31, 41, 55, 0.7);
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 14px;
    display: inline-block;
    font-size: 0.88rem;
    margin-right: 8px;
}

/* Social / Connect Buttons */
.social-buttons {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 10px;
}

.social-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.2s ease-in-out;
    border: 1px solid transparent;
}

.social-btn-github {
    background: #1e293b;
    color: #f1f5f9 !important;
    border-color: #334155;
}

.social-btn-github:hover {
    background: #0f172a;
    border-color: #a855f7;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(168, 85, 247, 0.25);
    color: #ffffff !important;
}

.social-btn-linkedin {
    background: rgba(10, 102, 194, 0.15);
    color: #38bdf8 !important;
    border-color: rgba(14, 165, 233, 0.4);
}

.social-btn-linkedin:hover {
    background: #0a66c2;
    color: #ffffff !important;
    border-color: #0284c7;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.35);
}

/* Footer */
.footer-container {
    margin-top: 48px;
    padding: 24px 16px;
    border-top: 1px solid #1f2937;
    text-align: center;
    color: #64748b;
    font-size: 0.84rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Session State Initialization
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "current_question" not in st.session_state:
    st.session_state.current_question = "Which country generated the highest total revenue?"
if "chart_type_choice" not in st.session_state:
    st.session_state.chart_type_choice = "Auto"

# 4. Sidebar: Configuration & Controls
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")
    
    # Model Selector
    model_labels = list(AVAILABLE_MODELS.keys())
    selected_label = st.selectbox(
        "🧠 Select Groq LLM Model",
        options=model_labels,
        index=model_labels.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in model_labels else 0,
        help="Switch dynamically between Groq's high-speed text models without restarting."
    )
    
    active_model_info = AVAILABLE_MODELS[selected_label]
    active_model_id = active_model_info["id"]
    
    # Model Details Card
    st.markdown(f"""
    <div class="sidebar-section">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <strong style="color: #c084fc;">{active_model_info['badge']}</strong>
            <code style="font-size: 0.75rem; color: #94a3b8;">{active_model_id}</code>
        </div>
        <p style="font-size: 0.84rem; color: #9ca3af; margin: 0;">
            {active_model_info['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("💡 *If one model reaches its rate limit, try another available model.*")
    
    st.markdown("---")
    
    # Database Status
    st.markdown("### 🗄️ Database Status")
    db_connected, db_msg = check_database_connection(DEFAULT_DB_PATH)
    if db_connected:
        st.success(f"**Chinook SQLite** Connected\n\n`{db_msg}`")
    else:
        st.error(f"**Database Issue**:\n\n{db_msg}")
        
    st.markdown("""
    <div class="sidebar-section">
        <div style="font-size: 0.85rem; line-height: 1.6;">
            <div><strong>Database:</strong> Chinook Digital Store</div>
            <div><strong>Connection Mode:</strong> <span style="color: #34d399;">Read-Only (mode=ro)</span></div>
            <div><strong>Max Autonomous Retries:</strong> 3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Security Badges
    st.markdown("### 🛡️ Security Checkpoints")
    st.markdown("""
    <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.8;">
        <div>✅ <strong>Connection Level:</strong> Read-only URI mode</div>
        <div>✅ <strong>Query Level:</strong> SELECT-only verified</div>
        <div>✅ <strong>Injection Filter:</strong> Multi-statement blocked</div>
        <div>✅ <strong>Destructive Filter:</strong> DDL/DML blocked</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Elevvo Internship & Developer Information
    st.markdown("### 👨‍💻 Developer & Connect")
    st.markdown("""
    <div class="sidebar-section">
        <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc; margin-bottom: 2px;">
            Ahmed Wael
        </div>
        <div style="font-size: 0.8rem; color: #a855f7; margin-bottom: 8px;">
            Elevvo AI Intern &bull; AI / Software Engineer
        </div>
        <p style="font-size: 0.8rem; color: #94a3b8; margin: 0 0 10px 0;">
            Autonomous agent built for the <strong>Elevvo AI Internship</strong> with LangGraph, Groq, SQLite & Streamlit.
        </p>
        <div class="social-buttons">
            <a href="https://github.com/Abo0wael" target="_blank" class="social-btn social-btn-github">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                GitHub: @Abo0wael
            </a>
            <a href="https://www.linkedin.com/in/ahmed-wael-9a6a5938a" target="_blank" class="social-btn social-btn-linkedin">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                LinkedIn: Ahmed Wael
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 5. Top Hero Header
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🧠 Autonomous Text-to-SQL Agent</div>
    <div class="hero-subtitle">Ask questions in natural language. Get real answers from your relational database.</div>
    <div class="badge-container">
        <span class="badge badge-purple">LangGraph State Machine</span>
        <span class="badge badge-cyan">Groq LLM Acceleration</span>
        <span class="badge badge-emerald">SQLite Read-Only Engine</span>
        <span class="badge badge-amber">Self-Correction Loop</span>
        <span class="badge badge-purple">Zero-Trust Security</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Check API Key availability
api_key_present = bool(get_groq_api_key())
if not api_key_present:
    st.warning("""
    ⚠️ **Groq API Key Not Detected!**
    Please set `GROQ_API_KEY` in your `.env` file for local use, or under **Streamlit Cloud Settings > Secrets** when deployed.
    """)

# 6. Clickable Example Questions
st.markdown("##### 💡 Example Questions (Click to populate):")
example_col1, example_col2, example_col3 = st.columns(3)
example_col4, example_col5 = st.columns(2)

example_questions = [
    "What are the top 5 customers by total spending?",
    "Which country has the highest total invoice revenue?",
    "Show the 10 best-selling tracks.",
    "How many customers are there in each country?",
    "Which artist has the most tracks?"
]

def set_question(q: str):
    st.session_state.current_question = q

with example_col1:
    if st.button(f"📊 {example_questions[0]}", use_container_width=True):
        set_question(example_questions[0])
with example_col2:
    if st.button(f"🌍 {example_questions[1]}", use_container_width=True):
        set_question(example_questions[1])
with example_col3:
    if st.button(f"🎵 {example_questions[2]}", use_container_width=True):
        set_question(example_questions[2])
with example_col4:
    if st.button(f"👥 {example_questions[3]}", use_container_width=True):
        set_question(example_questions[3])
with example_col5:
    if st.button(f"🎸 {example_questions[4]}", use_container_width=True):
        set_question(example_questions[4])

# 7. Main Input Area
st.markdown("---")
input_col, btn_col = st.columns([5, 1])

with input_col:
    user_query = st.text_input(
        "Enter your question about the Chinook database:",
        value=st.session_state.current_question,
        placeholder="e.g. Which country generated the highest total revenue?",
        label_visibility="collapsed"
    )

with btn_col:
    run_btn = st.button("🚀 Run Agent", use_container_width=True, type="primary")

# 8. Execution Flow
if run_btn:
    if not user_query.strip():
        st.error("Please enter a question to analyze.")
    elif not api_key_present:
        st.error("Cannot proceed: GROQ_API_KEY is not configured.")
    else:
        st.session_state.current_question = user_query.strip()
        
        # Observable Loading Status
        with st.status("🧠 Autonomous Agent Executing...", expanded=True) as status_box:
            def update_step(msg: str):
                status_box.write(f"• {msg}")
                
            update_step(f"Selected model: {active_model_id}")
            result = run_agent_workflow(
                question=user_query.strip(),
                model_id=active_model_id,
                db_path=DEFAULT_DB_PATH,
                progress_callback=update_step
            )
            
            # Rate limit check
            if result.get("error") and "RATE_LIMIT" in str(result["error"]).upper():
                status_box.update(label="⚠️ Rate Limit Reached", state="error", expanded=False)
            elif result.get("success"):
                status_box.update(label="✅ Query Execution Complete!", state="complete", expanded=False)
            else:
                status_box.update(label="⚠️ Agent Completed with Alerts", state="error", expanded=False)
                
        st.session_state.agent_result = result

# 9. Results Display
res = st.session_state.agent_result

if res is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 9a. Rate Limit Alert
    if res.get("error") and "RATE_LIMIT" in str(res["error"]).upper():
        st.warning("""
        ⚠️ **Rate limit reached for this model.**
        Please choose another model from the sidebar (e.g. *Model 1 - Fast* or *Model 3 - Powerful*) and try again.
        """)
        
    # 9b. Self-Correction Banner & Expander
    correction_details = res.get("correction_details")
    if correction_details and len(correction_details.get("corrections", [])) > 0:
        st.markdown("""
        <div class="correction-banner">
            <strong>🔁 Self-Correction Activated!</strong>
            The agent autonomously caught an initial SQL execution error, analyzed the SQLite diagnostic message, and regenerated a valid query.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔁 View Autonomous Self-Correction Trace", expanded=True):
            corr_list = correction_details.get("corrections", [])
            for c_idx, c in enumerate(corr_list, 1):
                st.markdown(f"**Attempt {c['attempt']} Correction:**")
                st.markdown(f"- **Failed SQL:** `{c['failed_sql']}`")
                st.markdown(f"- **SQLite Error Caught:** `{c['error']}`")
                st.markdown(f"- **Corrected SQL:**")
                st.code(c["corrected_sql"], language="sql")
                st.markdown("---")
                
    # 9c. Results Tabs
    tab_answer, tab_sql, tab_data, tab_viz, tab_trace = st.tabs([
        "💬 Answer", 
        "📝 Generated SQL", 
        "📊 Data Table", 
        "📈 Visualization", 
        "🔍 Agent Trace"
    ])
    
    # Tab 1: Answer
    with tab_answer:
        st.markdown("#### 💬 Natural Language Summary")
        if res.get("explanation"):
            st.info(res["explanation"])
        else:
            st.write("No summary generated.")
            if res.get("error"):
                st.error(f"Reason: {res.get('error')}")
                
    # Tab 2: Generated SQL
    with tab_sql:
        st.markdown("#### 📝 Validated SQLite Query")
        if res.get("sql_query"):
            st.code(res["sql_query"], language="sql")
            st.caption("🔒 Verified by Security Gate: Only single SELECT queries are permitted.")
        else:
            st.write("No query generated.")
            
    # Tab 3: Data Table
    with tab_data:
        st.markdown("#### 📊 Query Results")
        df = res.get("result_df")
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
            st.caption(f"Retrieved {len(df)} rows.")
        elif df is not None and df.empty:
            st.warning("The query returned 0 rows / empty result set.")
        else:
            st.write("No data available.")
            
    # Tab 4: Visualization
    with tab_viz:
        st.markdown("#### 📈 Visual Analytics")
        df = res.get("result_df")
        is_suitable, rec_type, x_c, y_c = detect_chart_suitability(df)
        
        if is_suitable and df is not None:
            chart_col1, chart_col2 = st.columns([1, 4])
            with chart_col1:
                chart_type = st.selectbox(
                    "Chart Type:",
                    options=["Auto", "Bar", "Line", "Pie"],
                    index=0,
                    key="chart_type_selector"
                )
            fig = create_chart(df, chart_type=chart_type, title=st.session_state.current_question)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Chart type not suitable for this data format.")
        else:
            st.info("💡 Results are not suitable for automatic charting (requires at least one categorical/date column and one numeric column).")
            
    # Tab 5: Agent Trace (Observable events only)
    with tab_trace:
        st.markdown("#### 🔍 Observable Agent Workflow Events")
        st.caption("Shows real-time state machine transitions and audit events (no hidden chain-of-thought).")
        
        trace_steps = res.get("history", [])
        if trace_steps:
            for step in trace_steps:
                if "failed" in step.lower() or "error" in step.lower():
                    st.markdown(f"❌ `{step}`")
                elif "succeeded" in step.lower() or "passed" in step.lower() or "loaded" in step.lower():
                    st.markdown(f"✅ `{step}`")
                elif "correction" in step.lower():
                    st.markdown(f"🔁 `{step}`")
                else:
                    st.markdown(f"• `{step}`")
        else:
            st.write("No trace records found.")
            
        st.markdown(f"""
        <div style="margin-top: 16px; font-size: 0.85rem; color: #94a3b8;">
            <strong>Total Attempts:</strong> {res.get('attempt_count', 0)} / {MAX_RETRIES} &nbsp;|&nbsp;
            <strong>Model Used:</strong> {res.get('model_id', active_model_id)} &nbsp;|&nbsp;
            <strong>Status:</strong> {'SUCCESS' if res.get('success') else 'FAILED'}
        </div>
        """, unsafe_allow_html=True)

# 10. Sleek Footer with Social Links
st.markdown("""
<div class="footer-container">
    <div style="margin-bottom: 12px; font-weight: 500; color: #94a3b8;">
        Developed by <strong style="color: #f1f5f9;">Ahmed Wael</strong> &bull; Built for the <strong>Elevvo AI Internship</strong>
    </div>
    <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
        <a href="https://github.com/Abo0wael" target="_blank" class="social-btn social-btn-github" style="font-size: 0.8rem; padding: 6px 14px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
            GitHub: @Abo0wael
        </a>
        <a href="https://www.linkedin.com/in/ahmed-wael-9a6a5938a" target="_blank" class="social-btn social-btn-linkedin" style="font-size: 0.8rem; padding: 6px 14px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
            LinkedIn: Ahmed Wael
        </a>
    </div>
    <div style="font-size: 0.78rem; color: #475569;">
        Autonomous Text-to-SQL Agent &bull; Powered by LangGraph, Groq & SQLite
    </div>
</div>
""", unsafe_allow_html=True)
