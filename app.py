import os
import sys
import ctypes
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
            for dll_name in ["libiomp5md.dll", "c10.dll", "torch_cpu.dll"]:
                dll_path = os.path.join(torch_lib, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception:
                        pass
    except Exception:
        pass

import streamlit as st
import pandas as pd
import io
import hashlib
import uuid
from datetime import datetime
import sqlite3
import re
import json

from bge_m3_engine import init_embedding_db, load_bge_m3_model, MODEL_NAME_PRIMARY
from jd_parser import parse_job_description
from resume_parser import extract_text_from_pdf, parse_full_resume
from scoring_engine import evaluate_candidate

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="HireSense AI – Resume Screening Portal",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Setup ---
def init_db():
    """Initialize SQLite database with users, ranking_history, and embeddings tables."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        name TEXT,
        job_title TEXT,
        company TEXT,
        date_joined TEXT,
        last_login TEXT
    )
    ''')
    
    # Create ranking history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS ranking_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        job_title TEXT,
        description TEXT,
        results TEXT,
        FOREIGN KEY (email) REFERENCES users (email)
    )
    ''')
    
    # Create interview results and proctoring table
    c.execute('''
    CREATE TABLE IF NOT EXISTS interview_results (
        id TEXT PRIMARY KEY,
        candidate_id TEXT,
        recruiter_email TEXT,
        candidate_name TEXT,
        candidate_email TEXT,
        candidate_phone TEXT,
        role TEXT,
        overall_score INTEGER,
        percentage INTEGER,
        verdict TEXT,
        highest_level INTEGER,
        violations_count INTEGER DEFAULT 0,
        violations_log TEXT,
        category_scores TEXT,
        turns_transcript TEXT,
        summary_report TEXT,
        recruiter_decision TEXT DEFAULT 'Pending Review',
        recruiter_notes TEXT,
        started_at TEXT,
        finished_at TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    
    # Initialize embedding vector store
    init_embedding_db('Resume.db')

# --- Initialize Session State ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.session_state["user_name"] = None
    st.session_state["profile_tab"] = "profile"
    st.session_state["current_page"] = "login"

# --- Security Functions ---
def hash_password(password, salt=None):
    """Hash a password for storing."""
    if salt is None:
        salt = uuid.uuid4().hex
    hashed = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    return f"{salt}${hashed}"

def verify_password(stored_password, provided_password):
    """Verify stored password against provided input."""
    try:
        salt, hashed = stored_password.split('$')
        return hashed == hashlib.sha256(salt.encode() + provided_password.encode()).hexdigest()
    except Exception:
        return False

# --- User Management Functions ---
def save_user(email, password, name=""):
    """Registers a new user in database."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return False
    
    hashed_password = hash_password(password)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
        (email, hashed_password, name, "", "", current_date, current_date)
    )
    conn.commit()
    conn.close()
    return True

def authenticate_user(email, password):
    """Authenticate user with email and password."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    if not result:
        conn.close()
        return False
    
    stored_password = result[0]
    if verify_password(stored_password, password):
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("UPDATE users SET last_login = ? WHERE email = ?", (current_date, email))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def update_profile(email, name, job_title, company):
    """Update user profile information."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    c.execute(
        "UPDATE users SET name = ?, job_title = ?, company = ? WHERE email = ?",
        (name, job_title, company, email)
    )
    conn.commit()
    conn.close()
    return True

def get_user_profile(email):
    """Get user profile data."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    c.execute(
        "SELECT email, name, job_title, company, date_joined, last_login FROM users WHERE email = ?",
        (email,)
    )
    result = c.fetchone()
    conn.close()
    if not result:
        return None
    return {
        "email": result[0],
        "name": result[1],
        "job_title": result[2],
        "company": result[3],
        "date_joined": result[4],
        "last_login": result[5]
    }

def change_password(email, current_password, new_password):
    """Change user password."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    if not result:
        conn.close()
        return False, "User not found"
    
    stored_password = result[0]
    if not verify_password(stored_password, current_password):
        conn.close()
        return False, "Current password is incorrect"
    
    hashed_password = hash_password(new_password)
    c.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
    conn.commit()
    conn.close()
    return True, "Password changed successfully"

def save_ranking_history(email, job_title, description, results):
    """Save resume ranking history for user."""
    try:
        conn = sqlite3.connect('Resume.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO ranking_history (email, timestamp, job_title, description, results) VALUES (?, ?, ?, ?, ?)",
            (
                email,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                job_title,
                description,
                results.to_json()
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving history: {e}")

def get_user_history(email):
    """Get resume ranking history for user."""
    conn = sqlite3.connect('Resume.db')
    query = "SELECT id, timestamp, job_title, description, results FROM ranking_history WHERE email = ? ORDER BY timestamp DESC"
    history_df = pd.read_sql_query(query, conn, params=(email,))
    conn.close()
    return history_df

# --- Interview Engine & Database Helpers ---
def get_interview_results(recruiter_email=None):
    """Fetch completed interview results from database."""
    conn = sqlite3.connect('Resume.db')
    try:
        if recruiter_email and recruiter_email != "all":
            query = "SELECT * FROM interview_results WHERE recruiter_email = ? ORDER BY finished_at DESC"
            df = pd.read_sql_query(query, conn, params=(recruiter_email,))
        else:
            query = "SELECT * FROM interview_results ORDER BY finished_at DESC"
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def get_interview_details(interview_id):
    """Fetch complete details for a single interview."""
    conn = sqlite3.connect('Resume.db')
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM interview_results WHERE id = ?", (interview_id,))
        row = c.fetchone()
        if not row:
            return None
        col_names = [description[0] for description in c.description]
        return dict(zip(col_names, row))
    except Exception:
        return None
    finally:
        conn.close()

def update_recruiter_decision(interview_id, decision, notes=""):
    """Update recruiter decision on an interview."""
    try:
        conn = sqlite3.connect('Resume.db')
        c = conn.cursor()
        c.execute(
            "UPDATE interview_results SET recruiter_decision = ?, recruiter_notes = ? WHERE id = ?",
            (decision, notes, interview_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating decision: {e}")
        return False

def check_interview_engine_health():
    """Check if the interview server is responding."""
    import urllib.request
    try:
        req = urllib.request.urlopen("http://localhost:3000/api/health", timeout=1.5)
        return req.getcode() == 200
    except Exception:
        return False

def ensure_interview_engine_running():
    """Starts the Node.js interview server if not already running."""
    import subprocess
    import threading
    if check_interview_engine_health():
        return True
    
    server_dir = os.path.join(os.path.dirname(__file__), "Final_Interview_Module", "interview-platform")
    if not os.path.exists(server_dir):
        return False

    def _start():
        try:
            cmd = ["node", "server.js"]
            creation_flags = 0x08000000 if sys.platform == "win32" else 0
            subprocess.Popen(cmd, cwd=server_dir, creationflags=creation_flags)
        except Exception as e:
            print(f"Failed to start interview server: {e}")

    threading.Thread(target=_start, daemon=True).start()
    return True

# --- Cached BGE-M3 Model Loader ---
@st.cache_resource
def get_bge_model():
    """Cache and load BAAI/bge-m3 embedding model."""
    model, model_name = load_bge_m3_model(MODEL_NAME_PRIMARY)
    return model, model_name

# --- Custom Styling ---
st.markdown("""
    <style>
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            font-size: 16px;
            border-radius: 6px;
            padding: 10px 16px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0066cc;
            box-shadow: 0 4px 12px rgba(30,144,255,0.3);
        }
        .card-box {
            background: #ffffff;
            border-radius: 10px;
            padding: 18px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            margin-bottom: 12px;
        }
        .badge-strong {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 13px;
        }
        .badge-moderate {
            background-color: #fef9c3;
            color: #854d0e;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 13px;
        }
        .badge-weak {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Navigation Pages ---
def show_login_page():
    st.sidebar.title("📝 User Login")
    st.sidebar.markdown("### Access HireSense AI Portal")
    login_email = st.sidebar.text_input("📧 Email", key="login_email", placeholder="Enter your email")
    login_password = st.sidebar.text_input("🔑 Password", type="password", key="login_password", placeholder="Enter your password")
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔐 Login", use_container_width=True):
            if authenticate_user(login_email, login_password):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = login_email
                profile = get_user_profile(login_email)
                st.session_state["user_name"] = profile["name"] if profile else login_email
                st.session_state["current_page"] = "dashboard"
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid email or password")
    with col2:
        if st.button("📝 Register", use_container_width=True):
            st.session_state["current_page"] = "register"
            st.rerun()

def show_register_page():
    st.sidebar.title("📝 Register Account")
    reg_email = st.sidebar.text_input("📧 Email*", key="reg_email", placeholder="Enter your email")
    reg_name = st.sidebar.text_input("👤 Full Name", key="reg_name", placeholder="Enter your full name")
    reg_password = st.sidebar.text_input("🔑 Password*", type="password", key="reg_password", placeholder="Enter your password")
    reg_confirm_password = st.sidebar.text_input("🔑 Confirm Password*", type="password", key="reg_confirm_password", placeholder="Confirm your password")
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Register", use_container_width=True):
            if not reg_email or not reg_password:
                st.sidebar.error("❌ Email and password required")
            elif reg_password != reg_confirm_password:
                st.sidebar.error("❌ Passwords do not match")
            else:
                if save_user(reg_email, reg_password, reg_name):
                    st.sidebar.success("✅ Account created! Please login.")
                    st.session_state["current_page"] = "login"
                    st.rerun()
                else:
                    st.sidebar.warning("⚠ Email already registered.")
    with col2:
        if st.button("↩️ Login", use_container_width=True):
            st.session_state["current_page"] = "login"
            st.rerun()

def show_profile_page():
    st.title("👤 Recruiter Profile")
    profile = get_user_profile(st.session_state["user_email"])
    if not profile:
        st.error("❌ Error loading profile")
        return
    
    profile_tab, password_tab, history_tab = st.tabs(["✏️ Edit Profile", "🔐 Password", "📊 Screening History"])
    
    with profile_tab:
        name = st.text_input("Full Name", value=profile["name"] if profile["name"] else "")
        job_title = st.text_input("Job Title", value=profile["job_title"] if profile["job_title"] else "")
        company = st.text_input("Company", value=profile["company"] if profile["company"] else "")
        if st.button("💾 Save Changes"):
            if update_profile(profile["email"], name, job_title, company):
                st.session_state["user_name"] = name
                st.success("✅ Profile updated successfully!")
                st.rerun()

    with password_tab:
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")
        if st.button("🔄 Update Password"):
            if new_password != confirm_new_password:
                st.error("❌ Passwords do not match")
            else:
                ok, msg = change_password(profile["email"], current_password, new_password)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

    with history_tab:
        history = get_user_history(profile["email"])
        if history.empty:
            st.info("No prior screening history found.")
        else:
            for idx, row in history.iterrows():
                with st.expander(f"📌 {row['job_title']} — {row['timestamp']}"):
                    st.write(row['description'])

def show_dashboard():
    welcome_name = st.session_state["user_name"] or st.session_state["user_email"] or "Recruiter"

    st.markdown("""
        <h2 style="
            background: -webkit-linear-gradient(45deg, #1FA2FF, #12D8FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            text-align: center;
            font-size: 2.3rem;">
            🤖 HireSense AI — Resume Screening Portal
        </h2>
    """, unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; font-size:16px; margin-bottom:15px;'>Welcome back, <b>{welcome_name}</b> 👋</div>", unsafe_allow_html=True)

    bge_model, model_name = get_bge_model()
    st.caption("⚡ Powered by advanced semantic embeddings for intelligent candidate matching")

    tab_screening, tab_interview, tab_results = st.tabs([
        "🚀 Candidate Screening & Ranking",
        "🎙️ AI Technical Interview Room",
        "📊 Interview Assessment Results"
    ])

    with tab_screening:
        # --- Job Information & Inputs ---
        st.subheader("📋 1. Job Description & Upload Resumes")
        col_title, col_exp = st.columns([2, 1])
        with col_title:
            job_title = st.text_input("Job Title", placeholder="e.g. Senior Python Backend Engineer")
        with col_exp:
            min_exp = st.number_input("Min Required Experience (Years)", min_value=0, max_value=30, value=0, step=1)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("#### 📄 Job Description")
            # --- JD File Upload ---
            jd_file = st.file_uploader(
                "Upload JD File (PDF or TXT) — optional",
                type=["pdf", "txt"],
                key="jd_file_upload",
                help="Upload a PDF or TXT file to auto-fill the Job Description below."
            )
            jd_prefill = ""
            if jd_file is not None:
                try:
                    if jd_file.type == "application/pdf" or jd_file.name.lower().endswith(".pdf"):
                        from pypdf import PdfReader
                        reader = PdfReader(jd_file)
                        jd_prefill = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        ).strip()
                    else:
                        jd_prefill = jd_file.read().decode("utf-8", errors="ignore").strip()
                    if jd_prefill:
                        st.success(f"✅ JD extracted from '{jd_file.name}' ({len(jd_prefill)} characters)")
                except Exception as e:
                    st.warning(f"⚠️ Could not parse JD file: {e}")

            job_description = st.text_area(
                "Job Description",
                value=jd_prefill,
                placeholder="Paste the full Job Description (JD) here, or upload a file above...",
                height=220,
                key="job_desc"
            )
        with col2:
            st.markdown("#### 📂 Upload Candidate Resumes")
            uploaded_files = st.file_uploader(
                "Select PDF Resumes",
                type=["pdf"],
                accept_multiple_files=True,
                key="resume_files"
            )
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully.")
                for f in uploaded_files:
                    st.caption(f"📎 {f.name}")

        st.markdown("---")

        # --- Screening Trigger ---
        if st.button("🔍 Screen & Rank Candidates", disabled=not (uploaded_files and job_description)):
            with st.spinner("🧠 Analyzing Job Description, generating semantic embeddings & evaluating candidates..."):
                # 1. Parse Job Description
                jd_profile = parse_job_description(job_description)
                if min_exp > 0:
                    jd_profile["min_years_experience"] = float(min_exp)

                # 2. Process each resume
                candidate_evaluations = []
                error_files = []

                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    if "Error extracting text" in text:
                        error_files.append(file.name)
                    else:
                        cand_profile = parse_full_resume(text)
                        # Ensure candidate name falls back to file name if generic
                        if cand_profile["candidate_name"] == "Candidate Profile":
                            file_base = os.path.splitext(file.name)[0].replace("_", " ").title()
                            cand_profile["candidate_name"] = file_base

                        eval_res = evaluate_candidate(jd_profile, cand_profile, bge_model=bge_model)
                        eval_res["file_name"] = file.name
                        candidate_evaluations.append(eval_res)

                if error_files:
                    st.warning(f"⚠ Could not parse text from: {', '.join(error_files)}")

                if candidate_evaluations:
                    # Sort candidates by overall score descending
                    ranked_candidates = sorted(candidate_evaluations, key=lambda x: x["overall_score"], reverse=True)
                    st.session_state["ranked_candidates"] = ranked_candidates
                    st.session_state["jd_profile"] = jd_profile
                    st.session_state["job_title"] = job_title or "Job Requirement"
                    st.session_state["job_description"] = job_description

                    # Save ranking history
                    res_df = pd.DataFrame([{
                        "Rank": idx + 1,
                        "Candidate": c["candidate_profile"]["candidate_name"],
                        "File": c["file_name"],
                        "Match Score": f"{c['overall_score']}%",
                        "Status": c["status"]
                    } for idx, c in enumerate(ranked_candidates)])

                    save_ranking_history(
                        st.session_state.get("user_email") or "guest@hiresense.ai",
                        job_title or "Job Requirement",
                        job_description,
                        res_df
                    )

        # --- Display Screening Dashboard Results ---
        if "ranked_candidates" in st.session_state and st.session_state["ranked_candidates"]:
            ranked = st.session_state["ranked_candidates"]
            jd_prof = st.session_state["jd_profile"]

            st.markdown("## 📊 Recruiter Dashboard & Shortlist Leaderboard")

            # --- Top Summary KPI Metrics ---
            total_cand = len(ranked)
            top_score = ranked[0]["overall_score"]
            avg_score = round(sum(c["overall_score"] for c in ranked) / total_cand, 1)
            strong_count = sum(1 for c in ranked if c["overall_score"] >= 80.0)
            mod_count = sum(1 for c in ranked if 60.0 <= c["overall_score"] < 80.0)
            weak_count = sum(1 for c in ranked if c["overall_score"] < 60.0)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("📁 Total Resumes", total_cand)
            m2.metric("🏆 Top Match Score", f"{top_score}%")
            m3.metric("📈 Avg Match Score", f"{avg_score}%")
            m4.metric("🌟 Strong Matches", strong_count)
            m5.metric("⚠️ Weak Matches", weak_count)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- JD Structured Extraction Summary ---
            with st.expander("📌 View Extracted Job Requirements Breakdown"):
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown("##### 🟢 Required Skills (Must-Have)")
                sc1.write(", ".join(jd_prof.get("required_skills", [])) if jd_prof.get("required_skills") else "None specified")

                sc2.markdown("##### 🔵 Preferred Skills (Nice-to-Have)")
                sc2.write(", ".join(jd_prof.get("preferred_skills", [])) if jd_prof.get("preferred_skills") else "None specified")

                sc3.markdown("##### 💼 Experience & Education")
                sc3.write(f"**Min Exp:** {jd_prof.get('min_years_experience')} years")
                sc3.write(f"**Education:** {', '.join(jd_prof.get('education_required', [])) if jd_prof.get('education_required') else 'Any'}")

            # --- Candidate Leaderboard Table ---
            st.subheader("🏆 Candidate Shortlist Ranking")

            table_data = []
            for idx, c in enumerate(ranked, start=1):
                p = c["candidate_profile"]
                table_data.append({
                    "Rank": idx,
                    "Candidate Name": p["candidate_name"],
                    "Overall Match": f"{c['overall_score']}%",
                    "Required Skills": f"{c['scores']['req_skill']}%",
                    "Semantic Relevance": f"{c['scores']['semantic']}%",
                    "Experience": p["experience_display"],
                    "Status": c["status"],
                    "File Name": c["file_name"]
                })

            df_leaderboard = pd.DataFrame(table_data)
            st.dataframe(df_leaderboard, hide_index=True, use_container_width=True)

            # --- Candidate Detail Inspection Section ---
            st.markdown("---")
            st.subheader("👤 Detailed Candidate Analysis & Match Explanation")
            st.caption("Click on any candidate below to view their explainable score breakdown, skill matches, resume evidence, and gaps.")

            cand_names = [f"Rank {idx+1}: {c['candidate_profile']['candidate_name']} ({c['overall_score']}%)" for idx, c in enumerate(ranked)]
            selected_cand_idx = st.selectbox("Select Candidate to Inspect:", range(len(cand_names)), format_func=lambda i: cand_names[i])

            selected_cand = ranked[selected_cand_idx]
            cp = selected_cand["candidate_profile"]

            # --- Candidate Detail View Card ---
            st.markdown(f"### 📄 Candidate Analysis: **{cp['candidate_name']}**")

            # Overview bar
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Overall Score", f"{selected_cand['overall_score']}%")
            oc2.metric("Rank", f"#{selected_cand_idx + 1} of {total_cand}")
            oc3.metric("Match Status", selected_cand["status"])
            oc4.metric("Experience Detected", cp["experience_display"])

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Multi-Factor Explainability Table ---
            st.markdown("#### 🧢 Score Calculation Breakdown")
            st.caption("Every component of the final match score is fully explainable and weighted consistently to 100%.")

            breakdown_df = pd.DataFrame(selected_cand["breakdown"])
            breakdown_df.columns = ["Category", "Category Score", "Weighted Contribution", "Explanation"]
            st.table(breakdown_df)

            # --- Required Skill Match Analysis (✓ / ◐ / ✗) ---
            st.markdown("#### 🛠️ Required Skills Match")
            col_req, col_part, col_miss = st.columns(3)

            with col_req:
                st.markdown("##### 🟢 Matched (✓)")
                if selected_cand["req_matched"]:
                    for item in selected_cand["req_matched"]:
                        st.markdown(f"<span class='badge-strong'>✓ {item}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No exact required skills matched.")

            with col_part:
                st.markdown("##### 🟡 Partial / Related (◐)")
                if selected_cand["req_partial"]:
                    for item in selected_cand["req_partial"]:
                        st.markdown(f"<span class='badge-moderate'>◐ {item['skill']}</span>", unsafe_allow_html=True)
                        st.caption(item['note'])
                else:
                    st.caption("No partial/related skills.")

            with col_miss:
                st.markdown("##### 🔴 Missing (✗)")
                if selected_cand["req_missing"]:
                    for item in selected_cand["req_missing"]:
                        st.markdown(f"<span class='badge-weak'>✗ {item}</span>", unsafe_allow_html=True)
                else:
                    st.success("All required skills satisfied!")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Preferred Skill Match Visualization ---
            st.markdown("#### ⭐ Preferred Skills Match")
            pref_skills_all = jd_prof.get("preferred_skills", [])
            if not pref_skills_all:
                st.caption("No preferred skills were specified in the Job Description.")
            else:
                pref_matched   = selected_cand.get("pref_matched", [])
                pref_partial   = selected_cand.get("pref_partial", [])
                pref_missing   = selected_cand.get("pref_missing", [])

                # Summary pill bar
                p1, p2, p3 = st.columns(3)
                p1.metric("✅ Matched",  len(pref_matched))
                p2.metric("◐ Partial",   len(pref_partial))
                p3.metric("❌ Missing",   len(pref_missing))

                st.markdown("<br>", unsafe_allow_html=True)

                # Visual badge grid
                col_pm, col_pp, col_pms = st.columns(3)

                with col_pm:
                    st.markdown("##### 🟢 Matched Preferred (✓)")
                    if pref_matched:
                        for item in pref_matched:
                            st.markdown(
                                f"<span class='badge-strong'>✓ {item}</span>&nbsp;",
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("None matched.")

                with col_pp:
                    st.markdown("##### 🟡 Partial / Related (◐)")
                    if pref_partial:
                        for item in pref_partial:
                            st.markdown(
                                f"<span class='badge-moderate'>◐ {item['skill']}</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(item['note'])
                    else:
                        st.caption("No partial matches.")

                with col_pms:
                    st.markdown("##### 🔴 Missing Preferred (✗)")
                    if pref_missing:
                        for item in pref_missing:
                            st.markdown(
                                f"<span class='badge-weak'>✗ {item}</span>&nbsp;",
                                unsafe_allow_html=True
                            )
                    else:
                        st.success("All preferred skills present!")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Verbatim Resume Evidence ---
            st.markdown("#### 💬 Evidence Grounding From Resume")
            st.caption("Actual sentence quotes extracted directly from candidate's uploaded resume supporting their match score:")

            if selected_cand["evidence_snippets"]:
                for ev in selected_cand["evidence_snippets"]:
                    st.info(f"**Skill / Requirement:** `{ev['target']}`\n\n> *\"{ev['quote']}\"*")
            else:
                st.caption("No specific verbatim quote extracted.")

            # Download buttons
            st.markdown("---")
            d1, d2 = st.columns(2)
            with d1:
                csv_data = df_leaderboard.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Shortlist CSV", csv_data, "candidate_shortlist.csv", "text/csv")
            with d2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df_leaderboard.to_excel(writer, index=False)
                buf.seek(0)
                st.download_button("📥 Download Shortlist Excel", buf, "candidate_shortlist.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Fast-track selected candidate to Interview tab
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🎙️ Invite {cp['candidate_name']} to Proctored AI Interview", key=f"invite_{selected_cand_idx}", use_container_width=True):
                st.session_state["preselected_candidate"] = selected_cand
                st.success(f"Candidate **{cp['candidate_name']}** has been pre-filled for interview! Click on the '🎙️ AI Technical Interview Room' tab above to review & start.")

    # =========================================================================
    # TAB 2: AI TECHNICAL INTERVIEW ROOM (PROCTORED)
    # =========================================================================
    with tab_interview:
        st.subheader("🎙️ AI Technical Interview & Proctoring Room")
        
        # Engine status check & auto-runner
        is_engine_ready = check_interview_engine_health()
        if not is_engine_ready:
            ensure_interview_engine_running()
            is_engine_ready = check_interview_engine_health()

        col_stat1, col_stat2 = st.columns([3, 1])
        with col_stat1:
            st.caption("Adaptive, voice-enabled technical interview engine with live webcam verification, fullscreen proctoring, and anti-cheat tracking.")
        with col_stat2:
            if is_engine_ready:
                st.success("🟢 Interview Engine: Online")
            else:
                st.warning("🟡 Starting Engine on :3000...")

        # Step 1: Candidate & Resume Intake
        st.markdown("#### 📄 1. Candidate Resume & Evaluation Setup")
        
        intake_source = st.radio(
            "Candidate Source:",
            ["📤 Upload New Candidate Resume", "🏆 Select From Shortlisted Candidates"],
            horizontal=True,
            key="intake_source_radio"
        )

        cand_name = "Candidate"
        cand_email = ""
        cand_phone = ""
        cand_role = st.session_state.get("job_title", "Software Engineer")
        cand_exp = "1-3 Years"
        cand_skills = ["Data Structures", "Algorithms", "Databases", "System Design", "Python"]
        cand_resume_text = ""

        # Pre-fill if recruiter clicked "Invite" in Tab 1
        if "preselected_candidate" in st.session_state and st.session_state["preselected_candidate"]:
            pre_cand = st.session_state["preselected_candidate"]
            cp_pre = pre_cand["candidate_profile"]
            cand_name = cp_pre["candidate_name"]
            cand_skills = cp_pre.get("skills", cand_skills)[:10]
            cand_exp = cp_pre.get("experience_display", "Technical Candidate")
            cand_resume_text = cp_pre.get("raw_text", "")
            st.info(f"📌 Loaded pre-selected candidate: **{cand_name}** ({len(cand_skills)} verified skills from screening)")

        if intake_source == "📤 Upload New Candidate Resume":
            uploaded_cand_resume = st.file_uploader(
                "Upload Candidate Resume (PDF or TXT)",
                type=["pdf", "txt"],
                key="interview_resume_upload",
                help="Upload a candidate's resume file to automatically extract their profile, skills, and experience."
            )
            if uploaded_cand_resume is not None:
                extracted_text = ""
                try:
                    if uploaded_cand_resume.name.lower().endswith(".pdf"):
                        extracted_text = extract_text_from_pdf(uploaded_cand_resume)
                    else:
                        extracted_text = uploaded_cand_resume.read().decode("utf-8", errors="ignore")
                    
                    if extracted_text and "Error extracting text" not in extracted_text:
                        cand_resume_text = extracted_text
                        parsed = parse_full_resume(extracted_text)
                        if parsed["candidate_name"] and parsed["candidate_name"] != "Candidate Profile":
                            cand_name = parsed["candidate_name"]
                        else:
                            cand_name = os.path.splitext(uploaded_cand_resume.name)[0].replace("_", " ").title()
                        
                        em = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', extracted_text)
                        if em:
                            cand_email = em.group(0)
                        ph = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', extracted_text)
                        if ph:
                            cand_phone = ph.group(0)
                        
                        if parsed.get("job_titles"):
                            cand_role = parsed["job_titles"][0]
                        elif st.session_state.get("job_title"):
                            cand_role = st.session_state["job_title"]
                        
                        cand_exp = parsed.get("experience_display", "Technical Candidate")
                        if parsed.get("skills"):
                            cand_skills = parsed["skills"][:10]
                        
                        st.success(f"✅ Successfully extracted profile for **{cand_name}** with {len(cand_skills)} key technical skills!")
                except Exception as e:
                    st.warning(f"⚠️ Error parsing uploaded resume: {e}")
        else:
            if "ranked_candidates" in st.session_state and st.session_state["ranked_candidates"]:
                ranked = st.session_state["ranked_candidates"]
                cand_options = [f"Rank {idx+1}: {c['candidate_profile']['candidate_name']} ({c['overall_score']}%)" for idx, c in enumerate(ranked)]
                chosen_idx = st.selectbox("Select Candidate from Screening Shortlist:", range(len(cand_options)), format_func=lambda i: cand_options[i])
                chosen_c = ranked[chosen_idx]
                cp = chosen_c["candidate_profile"]
                cand_name = cp["candidate_name"]
                cand_role = st.session_state.get("job_title", "Software Engineer")
                cand_exp = cp.get("experience_display", "Technical Candidate")
                cand_skills = cp.get("skills", ["Data Structures", "Algorithms", "System Design"])[:10]
                cand_resume_text = cp.get("raw_text", "")
            else:
                st.info("No candidates screened yet in this session. You can upload a resume above, or run screening in Tab 1.")

        # Editable details review card
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("✏️ Confirm & Customize Candidate Details", expanded=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                f_name = st.text_input("Candidate Full Name", value=cand_name, key="f_cand_name")
                f_email = st.text_input("Candidate Email", value=cand_email or f"{cand_name.lower().replace(' ', '.')}@hiresense.ai", key="f_cand_email")
                f_phone = st.text_input("Phone Number (Optional)", value=cand_phone, key="f_cand_phone")
            with f_col2:
                f_role = st.text_input("Target Role / Position", value=cand_role, key="f_cand_role")
                f_skills_str = st.text_area("Technical Topics & Skills to Probe (comma-separated)", value=", ".join(cand_skills), key="f_cand_skills", height=70)
                f_questions = st.number_input("Number of Primary Interview Questions", min_value=3, max_value=12, value=6, step=1, key="f_cand_num_q")

        # Proctoring & Security Pre-Flight Checklist
        st.markdown("#### 🛡️ 2. Proctoring & Assessment Security Controls")
        
        sec_col1, sec_col2 = st.columns(2)
        with sec_col1:
            st.markdown("""
            - 🎥 **Live Webcam Monitoring:** Picture-in-picture video stream verification.
            - 🎙️ **Microphone Speech-to-Text:** Live voice responses with waveform visualization.
            """)
        with sec_col2:
            st.markdown("""
            - 🖥️ **Full-Screen Enforcement:** Monitored full-screen view; window exits are logged.
            - 🚫 **Anti-Tab-Switch & Clipboard Protection:** Window blur, tab changes, and copy-paste are blocked and recorded.
            """)

        sec_agree = st.checkbox(
            "Enforce full-screen proctoring and integrity logging for this assessment session.",
            value=True,
            key="sec_proctor_agreement"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_start_col, _ = st.columns([1, 2])
        with btn_start_col:
            start_launch = st.button("🚀 Launch Proctored Technical Interview", type="primary", use_container_width=True, disabled=not sec_agree)

        if start_launch:
            with st.spinner("Initializing adaptive AI interview session..."):
                skills_clean = [s.strip() for s in f_skills_str.split(",") if s.strip()]
                candidate_payload = {
                    "name": f_name,
                    "email": f_email,
                    "phone": f_phone,
                    "role": f_role,
                    "skills": skills_clean,
                    "resumeText": cand_resume_text or f"Candidate: {f_name}\nRole: {f_role}\nSkills: {', '.join(skills_clean)}",
                    "jdText": st.session_state.get("job_description", f"Target role is {f_role} requiring core computer science principles and software engineering."),
                    "consent": "true",
                    "recruiter": st.session_state.get("user_email") or "guest@hiresense.ai"
                }
                try:
                    import urllib.request
                    import json
                    req_data = json.dumps(candidate_payload).encode('utf-8')
                    req = urllib.request.Request(
                        "http://localhost:3000/api/candidates",
                        data=req_data,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp_data = json.loads(resp.read().decode('utf-8'))
                        st.session_state["active_interview_data"] = resp_data
                        st.success(f"✅ Proctored Interview Room Ready for **{f_name}**! (ID: `{resp_data['candidateId']}`)")
                except Exception as ex:
                    st.error(f"❌ Failed to reach interview server: {ex}. Ensure the Node service is running.")

        # Display Active Interview Room
        if "active_interview_data" in st.session_state and st.session_state["active_interview_data"]:
            act = st.session_state["active_interview_data"]
            cid = act["candidateId"]
            recruiter_em = st.session_state.get("user_email") or "guest@hiresense.ai"
            live_url = f"http://localhost:3000/index.html?candidate={cid}&recruiter={recruiter_em}"

            st.markdown("---")
            st.markdown(f"### 🎧 Active Proctored Session: **{act['profile']['name']}**")

            open_col1, open_col2 = st.columns([1, 1])
            with open_col1:
                st.markdown(f"""
                <a href="{live_url}" target="_blank" style="text-decoration:none;">
                    <div style="background:#2563eb;color:#ffffff;text-align:center;padding:12px;border-radius:8px;font-weight:700;font-size:16px;box-shadow:0 4px 12px rgba(37,99,235,0.3);">
                        ↗️ Open Proctored Interview in Dedicated Full-screen Window
                    </div>
                </a>
                """, unsafe_allow_html=True)
                st.caption("Recommended for candidate testing: Opens a standalone tab for full webcam and microphone hardware permissions.")
            with open_col2:
                st.info(f"**Candidate ID:** `{cid}` · **Role:** `{act['profile']['role']}` · **Recruiter:** `{recruiter_em}`")

            st.markdown("#### 🖥️ Embedded Assessment Room")
            st.markdown(f"""
            <iframe 
                src="{live_url}" 
                width="100%" 
                height="880" 
                style="border:1px solid #cbd5e1; border-radius:12px; box-shadow:0 4px 16px rgba(0,0,0,0.06);"
                allow="camera; microphone; display-capture; fullscreen"
                allowfullscreen
            ></iframe>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: INTERVIEW ASSESSMENT RESULTS & AUDIT
    # =========================================================================
    with tab_results:
        st.subheader("📊 AI Interview Assessment Results & Integrity Audit")
        st.caption("Review candidate performance metrics, Q&A transcripts, and proctoring integrity records stored in Resume.db.")

        df_results = get_interview_results()
        
        if df_results.empty:
            st.info("No interview results recorded in database yet. Launch and complete a technical interview in the 'AI Technical Interview Room' tab to view results here.")
        else:
            # Top KPI Summary Cards
            tot_interviews = len(df_results)
            avg_pct = round(df_results["percentage"].mean(), 1) if "percentage" in df_results else 0
            sel_cnt = sum(df_results["verdict"] == "Selected") if "verdict" in df_results else 0
            tot_viols = int(df_results["violations_count"].sum()) if "violations_count" in df_results else 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("👥 Interviews Conducted", tot_interviews)
            k2.metric("📈 Average Score", f"{avg_pct}%")
            k3.metric("🌟 Recommended (Selected)", sel_cnt)
            k4.metric("🛡️ Flagged Incidents", tot_viols, delta="Clean" if tot_viols == 0 else f"{tot_viols} Flags", delta_color="normal" if tot_viols == 0 else "inverse")

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Leaderboard Table
            st.markdown("#### 🏆 Assessment Leaderboard")
            
            show_cols = ["id", "candidate_name", "role", "percentage", "verdict", "highest_level", "violations_count", "recruiter_decision", "finished_at"]
            avail_cols = [c for c in show_cols if c in df_results.columns]
            df_disp = df_results[avail_cols].copy()
            
            col_rename = {
                "id": "Interview ID",
                "candidate_name": "Candidate Name",
                "role": "Role",
                "percentage": "Match Score %",
                "verdict": "AI Recommendation",
                "highest_level": "Peak Level (1-5)",
                "violations_count": "Violations",
                "recruiter_decision": "Recruiter Decision",
                "finished_at": "Completed At"
            }
            df_disp = df_disp.rename(columns={k: v for k, v in col_rename.items() if k in df_disp.columns})
            st.dataframe(df_disp, hide_index=True, use_container_width=True)

            # Detailed Candidate Assessment Inspector
            st.markdown("---")
            st.markdown("#### 👤 Detailed Candidate Assessment Inspector")
            
            cand_selector = [
                f"{row['candidate_name']} ({row.get('role', 'Candidate')}) — {row.get('percentage', 0)}% [{row.get('verdict', 'N/A')}]"
                for _, row in df_results.iterrows()
            ]
            chosen_cand_idx = st.selectbox("Select Candidate to Inspect:", range(len(cand_selector)), format_func=lambda i: cand_selector[i])
            
            chosen_row = df_results.iloc[chosen_cand_idx]
            chosen_id = chosen_row["id"]

            # Candidate Score & Status Bar
            m_s1, m_s2, m_s3, m_s4 = st.columns(4)
            m_s1.metric("Overall Score", f"{chosen_row.get('percentage', 0)}%")
            m_s2.metric("AI Recommendation", chosen_row.get("verdict", "Pending"))
            m_s3.metric("Peak Difficulty Reached", f"Level {chosen_row.get('highest_level', 1)}/5")
            c_viols = int(chosen_row.get("violations_count", 0))
            m_s4.metric("Proctoring Violations", f"{c_viols} Logged", delta="Clean" if c_viols == 0 else "Flagged", delta_color="normal" if c_viols == 0 else "inverse")

            # Proctoring & Assessment Integrity Audit Breakdown Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🛡️ Proctoring & Integrity Audit Breakdown")
            
            v_log_str = chosen_row.get("violations_log")
            v_list = []
            if v_log_str:
                try:
                    v_list = json.loads(v_log_str) if isinstance(v_log_str, str) else v_log_str
                except Exception:
                    v_list = []
            
            aud1, aud2 = st.columns(2)
            with aud1:
                st.markdown("""
                - **Webcam Monitoring:** 🟢 Verified Live
                - **Fullscreen Compliance:** """ + ("🟢 Fullscreen Compliant" if not any(v.get('type') == 'Fullscreen Exited' for v in v_list) else f"🔴 Exited Fullscreen {sum(1 for v in v_list if v.get('type') == 'Fullscreen Exited')} time(s)"))
            with aud2:
                tab_switches_cnt = sum(1 for v in v_list if v.get('type') == 'Tab Switch')
                focus_blur_cnt = sum(1 for v in v_list if v.get('type') == 'Window Focus Lost')
                st.markdown(f"""
                - **Tab Switching:** {'🟢 0 Tab Switches' if tab_switches_cnt == 0 else f'🔴 {tab_switches_cnt} Tab Switch(es)'}
                - **Window Focus:** {'🟢 Continuous Focus' if focus_blur_cnt == 0 else f'🟡 {focus_blur_cnt} Focus Loss Event(s)'}
                """)
            
            if v_list:
                with st.expander(f"⚠️ View Chronological Audit Incidents ({len(v_list)} events)", expanded=True):
                    for v_item in v_list:
                        st.warning(f"**[{v_item.get('time', '00:00')}] {v_item.get('type', 'Violation')}:** {v_item.get('desc', '')} *(Logged at {v_item.get('at', '')})*")
            else:
                st.success("✅ Assessment Integrity Verified: Zero proctoring irregularities were logged during this session.")

            # Full Q&A Transcript
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 💬 Detailed Q&A Transcript & AI Evaluation")
            turns_str = chosen_row.get("turns_transcript")
            turns_data = []
            if turns_str:
                try:
                    turns_data = json.loads(turns_str) if isinstance(turns_str, str) else turns_str
                except Exception:
                    turns_data = []

            if turns_data:
                for q_idx, turn in enumerate(turns_data, start=1):
                    turn_score = turn.get('score', 0)
                    pill = "🟢 Strong" if turn_score >= 7 else "🟡 Moderate" if turn_score >= 5 else "🔴 Weak"
                    with st.expander(f"Q{q_idx} [{turn.get('topic', 'Topic')} · Difficulty L{turn.get('level', 1)}] — Score: {turn_score}/10 ({pill})", expanded=(q_idx <= 2)):
                        st.markdown(f"**Question:** {turn.get('question', '')}")
                        st.markdown(f"**Candidate Response:**\n> *\"{turn.get('answer', '')}\"*")
                        st.markdown(f"**AI Evaluator Critique:** {turn.get('summary', 'Evaluated.')}")
                        
                        col_hit, col_miss = st.columns(2)
                        with col_hit:
                            hits = turn.get('correct', [])
                            st.markdown(f"**Key Concepts Satisfied:** {', '.join(hits) if hits else 'None'}")
                        with col_miss:
                            misses = turn.get('missing', [])
                            st.markdown(f"**Missing Depth / Gaps:** {', '.join(misses) if misses else 'None'}")
            else:
                st.caption("No transcript items found for this record.")

            # Recruiter Decision & Notes
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📝 Recruiter Audit Decision & Notes")
            dec_col1, dec_col2 = st.columns([1, 2])
            with dec_col1:
                cur_dec = chosen_row.get("recruiter_decision", "Pending Review")
                options_dec = ["Pending Review", "✅ Advance Candidate", "❌ Reject Candidate"]
                sel_dec_idx = options_dec.index(cur_dec) if cur_dec in options_dec else 0
                chosen_new_dec = st.selectbox("Recruiter Decision:", options_dec, index=sel_dec_idx, key=f"rec_dec_{chosen_id}")
            with dec_col2:
                cur_note = chosen_row.get("recruiter_notes") or ""
                chosen_new_note = st.text_input("Reviewer Feedback & Notes:", value=cur_note, key=f"rec_note_{chosen_id}")

            if st.button("💾 Save Recruiter Decision", key=f"btn_save_dec_{chosen_id}"):
                if update_recruiter_decision(chosen_id, chosen_new_dec, chosen_new_note):
                    st.success("✅ Recruiter decision and notes saved to Resume.db!")
                    st.rerun()

            # Export Buttons
            st.markdown("---")
            st.markdown("##### 📥 Export Assessment Report")
            exp1, exp2, exp3 = st.columns(3)
            with exp1:
                st.markdown(f'<a href="http://localhost:3000/api/interview/{chosen_id}/download?format=txt" target="_blank" style="text-decoration:none;"><div style="background:#f1f5f9;border:1px solid #cbd5e1;color:#1e293b;padding:8px 12px;border-radius:6px;text-align:center;font-weight:600;">📝 Download Text Summary</div></a>', unsafe_allow_html=True)
            with exp2:
                st.markdown(f'<a href="http://localhost:3000/api/interview/{chosen_id}/download?format=json" target="_blank" style="text-decoration:none;"><div style="background:#f1f5f9;border:1px solid #cbd5e1;color:#1e293b;padding:8px 12px;border-radius:6px;text-align:center;font-weight:600;">⚙ Download Full JSON Report</div></a>', unsafe_allow_html=True)
            with exp3:
                st.markdown(f'<a href="http://localhost:3000/index.html?candidate={chosen_row.get("candidate_id", "")}" target="_blank" style="text-decoration:none;"><div style="background:#2563eb;color:white;padding:8px 12px;border-radius:6px;text-align:center;font-weight:600;">🖨️ View & Print Official PDF</div></a>', unsafe_allow_html=True)


# --- Sidebar ---
def render_sidebar():
    st.sidebar.markdown("""
<h2 style="
    text-align: center;
    font-weight: bold;
    font-size: 32px;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
    HireSense AI
</h2>
    """, unsafe_allow_html=True)
    
    if st.session_state["authenticated"]:
        st.sidebar.subheader(f"👤 {st.session_state['user_email']}")
        
        # Engine indicator
        if check_interview_engine_health():
            st.sidebar.caption("🟢 AI Interview Engine: Active")
        else:
            st.sidebar.caption("🟡 AI Interview Engine: Standby")

        st.sidebar.markdown("---")
        if st.sidebar.button("🏠 Recruiter Dashboard", use_container_width=True):
            st.session_state["current_page"] = "dashboard"
            st.rerun()
        if st.sidebar.button("👤 My Profile", use_container_width=True):
            st.session_state["current_page"] = "profile"
            st.rerun()
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user_email"] = None
            st.session_state["current_page"] = "login"
            st.rerun()

def render_footer():
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #666;
            text-align: center;
            padding: 8px 0;
            font-size: 13px;
            border-top: 1px solid #e2e8f0;
        }
        </style>
        <div class="footer">
            HireSense AI — Resume Screening Portal · Intelligent Candidate Shortlisting
        </div>
    """, unsafe_allow_html=True)

# --- Main Routine ---
def main():
    init_db()
    render_sidebar()
    
    curr_page = st.session_state.get("current_page", "dashboard")
    if not st.session_state["authenticated"]:
        if curr_page == "register":
            show_register_page()
        else:
            show_login_page()
    else:
        if curr_page == "profile":
            show_profile_page()
        else:
            show_dashboard()
            
    render_footer()

if __name__ == "__main__":
    main()