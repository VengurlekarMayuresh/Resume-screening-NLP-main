# 🤖 HireSense AI — Intelligent Resume Screening Portal
### Comprehensive Project Report

---

> **Document Purpose:** This report explains what HireSense AI is, how it works internally,
> and the end-to-end flow from a recruiter uploading files to ranked candidate results.
> It is written to be fully self-contained — no prior technical knowledge is required.

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [How It Works — Step by Step](#5-how-it-works--step-by-step)
6. [The Scoring Engine Explained](#6-the-scoring-engine-explained)
7. [Skill Matching Logic](#7-skill-matching-logic)
8. [Semantic Embedding Engine](#8-semantic-embedding-engine)
9. [Candidate Name Extraction](#9-candidate-name-extraction)
10. [Data Storage & Caching](#10-data-storage--caching)
11. [User Interface & Features](#11-user-interface--features)
12. [File Structure](#12-file-structure)
13. [Limitations & Future Scope](#13-limitations--future-scope)

---

## 1. Project Overview

**HireSense AI** is an AI-powered recruitment assistant that automatically screens, evaluates,
and ranks candidate resumes against a given Job Description (JD). Instead of a recruiter manually
reading hundreds of resumes, HireSense AI reads and scores each one in seconds, then presents a
ranked shortlist with full explainability — showing exactly *why* each candidate received their score.

The system is built as a **web application** using Python and Streamlit, making it accessible
directly from a browser with no additional installation needed for end users.

---

## 2. The Problem It Solves

Traditional resume screening is:

| Problem | Impact |
|---|---|
| **Time-consuming** | A recruiter can read ~6 resumes/hour; 200 applications = 33+ hours |
| **Inconsistent** | Human bias and fatigue lead to inconsistent evaluation criteria |
| **Keyword-only tools** | Simple matchers miss context and semantically related skills |
| **No explainability** | Automated tools often give a score with no reasoning |

HireSense AI addresses all four by combining **rule-based skill matching**,
**semantic AI understanding**, and a **fully transparent scoring breakdown**.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HireSense AI                             │
│                     (Streamlit Web App)                         │
├──────────────┬──────────────────────────┬───────────────────────┤
│  Frontend    │     Processing Engine    │    Data Layer         │
│  (app.py)    │                          │                       │
│              │  ┌─────────────────┐     │  ┌────────────────┐  │
│  Login /     │  │  jd_parser.py   │     │  │  Resume.db     │  │
│  Register    │  │  (JD Analysis)  │     │  │  (SQLite)      │  │
│              │  └────────┬────────┘     │  │                │  │
│  JD Input    │           │              │  │  - users       │  │
│  (text/PDF)  │  ┌────────▼────────┐     │  │  - history     │  │
│              │  │ resume_parser   │     │  │  - embeddings  │  │
│  Resume      │  │ .py             │     │  └────────────────┘  │
│  Upload      │  │ (Resume Parse)  │     │                       │
│              │  └────────┬────────┘     │  ┌────────────────┐  │
│  Results     │           │              │  │  HuggingFace   │  │
│  Dashboard   │  ┌────────▼────────┐     │  │  Model Cache   │  │
│              │  │ scoring_engine  │◄────┤  └────────────────┘  │
│              │  │ .py             │     │                       │
│              │  │ (7-factor score)│     │                       │
│              │  └────────┬────────┘     │                       │
│              │           │              │                       │
│              │  ┌────────▼────────┐     │                       │
│              │  │ bge_m3_engine   │     │                       │
│              │  │ .py             │     │                       │
│              │  │ (AI Embeddings) │     │                       │
│              │  └─────────────────┘     │                       │
└──────────────┴──────────────────────────┴───────────────────────┘
```

---

## 4. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Frontend / UI** | Streamlit (Python) | Browser-based web interface |
| **PDF Parsing** | PyPDF | Extracts text from PDF resumes and JDs |
| **AI Embeddings** | sentence-transformers | Converts text to numerical vectors for semantic comparison |
| **Primary Model** | `all-MiniLM-L6-v2` | 384-dimensional semantic embeddings (fast, reliable) |
| **NLP / Skills** | Custom regex + dictionary | Skill extraction and matching |
| **Database** | SQLite (`Resume.db`) | User accounts, ranking history, embedding cache |
| **Data Processing** | pandas, numpy | Data manipulation and vector math |
| **Auth / Security** | SHA-256 + salt hashing | Secure password storage |

---

## 5. How It Works — Step by Step

### Step 1: User Authentication 🔐

When a recruiter opens the app, they see a **Login / Register** form in the sidebar.

- **Registration**: The recruiter enters their email, name, and password. The password is hashed
  using **SHA-256 with a random salt** (making it impossible to reverse) and stored in SQLite.
- **Login**: The entered password is hashed and compared against the stored hash. On success,
  the session is authenticated and the recruiter reaches the dashboard.

> All screening history is tied to individual recruiter accounts and is completely private.

---

### Step 2: Providing the Job Description (JD) 📄

The recruiter has **two options** to provide the Job Description:

**Option A — Upload a File**
- Upload a PDF or TXT file containing the JD
- The system extracts all text from the file automatically
- The extracted text is auto-filled into the text area for review/editing

**Option B — Paste Text**
- Directly paste the JD text into the text area

The recruiter also provides:
- **Job Title** (e.g., "Senior Python Backend Engineer")
- **Minimum Required Experience** in years

---

### Step 3: JD Parsing — Extracting What the Job Needs 🧠

The `jd_parser.py` module analyzes the JD text to extract structured requirements:

**a) Section Detection**
The parser scans each line for keywords to categorize content:
- Lines under **"Required / Must Have / Essential"** → Required Skills
- Lines under **"Preferred / Nice to Have / Bonus"** → Preferred Skills
- Lines under **"Responsibilities / Duties"** → Job Responsibilities

**b) Skill Extraction**
The parser searches the JD for any of **90+ known technical skills** using exact
boundary-matching regex. Common aliases are all mapped to canonical names:
- "javascript", "java script", "JS" → `JavaScript`
- "ml", "machine learning" → `Machine Learning`
- "k8s", "kubernetes" → `Kubernetes`

The skill database covers:
- Programming languages: Python, Java, JavaScript, TypeScript, C++, Go, Rust...
- Frameworks: React, Node.js, Django, FastAPI, Spring Boot, Angular...
- AI/ML: Machine Learning, TensorFlow, PyTorch, Scikit-Learn, NLP...
- Databases: PostgreSQL, MongoDB, MySQL, Redis, Snowflake...
- Cloud & DevOps: AWS, Azure, GCP, Docker, Kubernetes, Terraform, CI/CD...

**c) Experience Extraction**
Regex patterns find phrases like "3+ years of experience" and extract the number.

**d) Education Requirements**
Degree requirements (B.Tech, M.Tech, Ph.D., MCA, MBA, etc.) are detected.

**Output — Structured JD Profile:**
```json
{
  "required_skills":  ["Python", "React", "PostgreSQL"],
  "preferred_skills": ["Docker", "AWS"],
  "min_years_experience": 3.0,
  "responsibilities": ["Build REST APIs", "Lead code reviews"],
  "education_required": ["B.Tech / B.E."],
  "raw_text": "..."
}
```

---

### Step 4: Uploading Candidate Resumes 📂

The recruiter uploads one or more **PDF resumes** via the multi-file uploader.
Each uploaded file name is listed as confirmation, and the Screen button
activates only when both a JD and at least one resume are present.

---

### Step 5: Resume Parsing — Reading Each Candidate 📖

For every uploaded PDF, `resume_parser.py` runs the following pipeline:

**a) Text Extraction**
PyPDF reads each page and extracts raw text character by character.

**b) Text Sanitization — The Critical Step**
Raw PDF text is messy. The sanitizer fixes four classes of problems:

| Problem | Example | Fix |
|---|---|---|
| Kerning artifact | `J a v a S c r i p t` | Merged → `JavaScript` |
| CamelCase splitting | `JavaScript` → `java script` | Protected before splitting |
| Newline-split skills | `Machine\nLearning` | Collapsed → `Machine Learning` |
| Multi-space | `Python   Django` | Normalized → `Python Django` |

Protected CamelCase terms (JavaScript, TypeScript, TensorFlow, MongoDB, etc.)
are temporarily replaced with safe placeholders before any processing,
then restored — preventing them from being incorrectly split.

**c) Section Segmentation**
The resume is divided into: Summary, Work Experience, Projects, Skills, Education, Certifications.

**d) Candidate Name Extraction** *(See Section 9 for full detail)*

**e) Skills Extraction**
All 90+ skills from the database are searched across the sanitized resume text.

**f) Experience Calculation**
Date ranges (e.g., "Jan 2020 – Present") are extracted and summed.
Explicit mentions ("3 years of experience") are also detected as fallback.

**g) Education & University Extraction**
Degree names and institution names are identified from the education section.

**Output — Structured Candidate Profile:**
```json
{
  "candidate_name":     "Neha Sharma",
  "skills":             ["Python", "JavaScript", "Machine Learning"],
  "years_of_experience": 2.0,
  "experience_display": "2 yrs",
  "degrees":            ["B.Tech / B.E. (Computer Science)"],
  "universities":       ["Mumbai University"],
  "raw_text":           "..."
}
```

---

### Step 6: Scoring Each Candidate ⚖️

`scoring_engine.py` evaluates each candidate against the JD across **7 weighted factors**.
*(Detailed in Section 6)*

---

### Step 7: Ranking & Displaying Results 🏆

After all candidates are scored:

1. **Sorted** by Overall Score — highest first
2. **KPI row**: 5 metric cards — Total, Top Score, Avg Score, Strong Matches, Weak Matches
3. **JD Breakdown expander**: extracted required skills, preferred skills, education, experience
4. **Candidate Leaderboard**: table of all candidates with score, status, experience, file name
5. **Deep Candidate Inspector** — select any candidate to see:
   - Full 7-factor score breakdown table with weighted points
   - Required Skills: 🟢 Matched / 🟡 Partial / 🔴 Missing — color-coded badges
   - Preferred Skills: same 3-column visualization with summary counts
   - Evidence quotes: verbatim sentences from the resume proving each skill match
6. **Download**: Export full shortlist as CSV or Excel

---

### Step 8: History Auto-Save 💾

Every screening session is automatically saved to the database under the recruiter's account.
Past sessions are viewable in the "📊 Screening History" tab on the Profile page.

---

## 6. The Scoring Engine Explained

Each candidate is evaluated across **7 independent factors**:

| # | Factor | Weight | What It Measures |
|---|---|---|---|
| 1 | Semantic Relevance | **20%** | Conceptual similarity between JD and resume (AI-powered) |
| 2 | Required Skills Match | **25%** | Coverage of must-have skills |
| 3 | Preferred Skills Match | **10%** | Coverage of nice-to-have skills |
| 4 | Experience Match | **15%** | Meets minimum years of experience requirement |
| 5 | Responsibility Match | **15%** | Work history aligns with JD duties |
| 6 | Project Relevance | **10%** | Projects relate to the role's technical domain |
| 7 | Education & Certs | **5%** | Holds required degrees |

### Final Score Formula

```
Final Score =
    (Semantic        × 0.20) +
    (Required Skills × 0.25) +
    (Preferred Skills × 0.10) +
    (Experience      × 0.15) +
    (Responsibilities × 0.15) +
    (Projects        × 0.10) +
    (Education       × 0.05)
```

### Match Status

| Score Range | Badge |
|---|---|
| ≥ 80% | 🌟 **Strong Match** |
| 60 – 79% | 🟡 **Moderate Match** |
| < 60% | 🔴 **Weak Match** |

---

## 7. Skill Matching Logic

HireSense AI uses a **3-tier anti-false-positive system** to avoid marking wrong skills:

### Tier 1 — Exact Match ✅
The skill is found verbatim in the resume text using word-boundary regex
(preventing "Java" from matching inside "JavaScript").

> JD requires **PostgreSQL** → Resume has **PostgreSQL** → ✅ Exact Match (+1.0 pt)

### Tier 2 — Related Technology Match ◐
The exact skill is absent, but a closely related technology from the same family is found.

> JD requires **PostgreSQL** → Resume has **MySQL** → ◐ Partial Match (+0.5 pt)
> JD requires **AWS** → Resume has **Azure** → ◐ Partial Match (+0.5 pt)

**Technology Family Map:**
```
PostgreSQL  ↔  MySQL, SQLite, Oracle DB, SQL Server
AWS         ↔  Azure, GCP
React       ↔  Vue.js, Angular
Docker      ↔  Kubernetes
Django      ↔  Flask, FastAPI
Java        ↔  Kotlin, C#, Scala
JavaScript  ↔  TypeScript
```

### Tier 3 — Missing ✗
Neither the skill nor any related skill is found. → 0 points

### Score Formula
```
Skill Score = (Exact + 0.5 × Partial) / Total_required_skills × 100
```

---

## 8. Semantic Embedding Engine

This is the AI layer that understands **meaning**, not just exact words.

### What Is a Semantic Embedding?

Every piece of text is converted into a long list of numbers called a **vector** (384 numbers
in this system). The key property: texts with *similar meaning* produce vectors that point
in nearly the same direction, even if they use completely different words.

**Example:**
- *"Built machine learning models for fraud detection"*
- *"Developed AI algorithms to identify fraudulent transactions"*

These use different words but mean the same thing. A semantic model produces very similar
vectors for both — and the system correctly scores a candidate higher even when their
resume uses different wording than the JD.

### How It Works

```
Step 1: Load Model (all-MiniLM-L6-v2)
        └─ Loaded once at startup, cached in memory

Step 2: Encode JD Text
        └─ JD text → [0.12, -0.45, 0.78, ... ] (384 numbers)

Step 3: Encode Resume Text
        └─ Resume → [0.11, -0.43, 0.76, ... ] (384 numbers)

Step 4: Cosine Similarity
        └─ Angle between the two vectors → score 0.0 to 1.0

Step 5: Score
        └─ Score × 100 = Semantic Relevance %
```

**Cosine Similarity formula:**
```
similarity = (A · B) / (|A| × |B|)
```
- Result of **1.0** = identical meaning
- Result of **0.0** = completely unrelated

### Embedding Cache
To avoid recomputing vectors for the same text, every embedding is stored in SQLite.
The text is hashed with SHA-256 as the key. On subsequent runs, the stored vector
is returned instantly (no model computation needed).

---

## 9. Candidate Name Extraction

Resume formats vary wildly. HireSense AI uses **3 strategies in sequence**:

**Strategy 1 — Top-12 Line Scan**
Scans the first 12 lines (where names always appear). For each line:
- ❌ Skip: contains email, phone, URL, LinkedIn, GitHub
- ❌ Skip: is a section header (Skills, Education, Experience…)
- ✂️ Strip: honorific prefix (Mr., Ms., Dr., Prof.)
- ✅ Accept: 2–4 words, all alphabetic (hyphens and apostrophes allowed)

**Strategy 2 — ALL-CAPS Detection**
Many Indian and formal resume templates write names in all-caps:
`NEHA SHARMA` → detected → converted to → `Neha Sharma`

**Strategy 3 — Pre-Header Detection**
If still no name found, checks the line immediately *before* the first
known section header. The name often appears directly above "Skills:" or "Experience:".

**Fallback**: Uses the PDF filename (without extension) as the display name.

---

## 10. Data Storage & Caching

All data lives in a single SQLite file: `Resume.db`

### Table: `users`
```
email (PK) | password_hash | name | job_title | company | date_joined | last_login
```
Passwords are stored as `salt$hash` — the original password is never stored.

### Table: `ranking_history`
```
id | email (FK) | timestamp | job_title | description | results (JSON)
```
Stores every screening session for history tracking.

### Table: `resume_embeddings`
```
text_hash (SHA-256, PK) | model_name | vector_blob (binary) | dim | created_at
```
Caches AI-computed vectors to avoid redundant computation. The text hash ensures
that the same resume text always returns the same cached vector regardless of file name.

---

## 11. User Interface & Features

### Sidebar
- Gradient "HireSense AI" logo
- Navigation buttons: Recruiter Dashboard | My Profile | Logout
- Login / Register forms (for unauthenticated access)

### Candidate Screening Section
| Element | Function |
|---|---|
| JD File Uploader | Upload PDF or TXT — text auto-fills below |
| JD Text Area | Paste/edit the job description manually |
| Job Title field | Free-text input |
| Min Experience | Number input (0–30 years) |
| Resume Uploader | Multi-PDF upload with filename list |
| Screen & Rank button | Disabled until JD + resumes are ready |

### Results Dashboard
| Element | Function |
|---|---|
| KPI Metrics Row | 5 cards: Total / Top Score / Avg Score / Strong / Weak |
| JD Requirements Expander | Required skills, preferred skills, education, experience |
| Candidate Leaderboard | Full table: rank, name, scores, experience, status, file |
| Candidate Inspector | 7-factor breakdown + skill badges + evidence quotes |
| Download Buttons | CSV export + Excel export of shortlist |

### Profile Page
- Edit: name, job title, company
- Change password (with verification of current password)
- Screening history: all past sessions with expandable details

---

## 12. File Structure

```
Resume-screening-NLP-main/
│
├── app.py              ← Main web app: UI, routing, authentication, results display
├── jd_parser.py        ← JD section detection, skill database (90+ skills), structured extraction
├── resume_parser.py    ← PDF text extraction, sanitization, name/skill/exp/edu parsing
├── scoring_engine.py   ← 7-factor weighted scoring, 3-tier skill classification, evidence extraction
├── bge_m3_engine.py    ← Embedding model loader, SQLite vector cache, cosine similarity
├── evaluator.py        ← Benchmark evaluation utilities (not part of main screening flow)
├── requirements.txt    ← Python package dependencies
├── Resume.db           ← SQLite database (auto-created on first run)
│
├── Samples/
│   ├── resumes/        ← Sample candidate PDF resumes for testing
│   └── JD/             ← Sample job description PDFs for testing
│
└── .venv/              ← Python virtual environment
```

---

## 13. Limitations & Future Scope

### Current Limitations

| Limitation | Details |
|---|---|
| PDF-only resumes | DOCX, images, and scanned PDFs are not supported |
| English-only | Skill database and NLP logic are English-specific |
| Static skill database | New technologies require manual addition |
| No ATS integration | Results cannot push to HR platforms automatically |
| Local model execution | Large-scale deployments may need GPU acceleration |

### Future Enhancements

- 🔗 **ATS Integration** — Push ranked results to Greenhouse, Lever, Workday
- 📄 **DOCX / Image Resume Support** — Word documents and scanned PDFs
- 🌐 **Multi-language NLP** — Hindi, French, German JDs and resumes
- 🤖 **LLM Candidate Summaries** — Gemini/GPT-generated one-line candidate summaries
- 📊 **Recruiter Analytics** — Cross-campaign hiring analytics and trends
- ⚡ **REST API** — Headless batch-processing API for enterprise volume
- 🔔 **Email Reports** — Auto-send shortlist to hiring managers after screening

---

## Summary

HireSense AI is a **7-factor AI-powered resume screening system** built on three pillars:

| Pillar | What It Does |
|---|---|
| **Rule-based Parsing** | Structures unformatted JDs and resumes into clean data |
| **Lexical Matching** | Finds exact and related skills with false-positive prevention |
| **Semantic AI** | Understands meaning beyond keywords using vector embeddings |

Every score is **100% explainable** — broken down by factor, with verbatim evidence
from the resume backing every match claim.

> **The bottom line:** A recruiter can go from 200 unread PDF resumes to a fully
> ranked, evidence-backed shortlist in **under 2 minutes**.

---

*Report generated: October 2026 | HireSense AI — Resume Screening Portal*
