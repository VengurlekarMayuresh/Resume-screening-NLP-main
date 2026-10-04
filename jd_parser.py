import re

# Comprehensive dictionary of technical and domain skills
SKILL_DATABASE = {
    # Languages
    "python": "Python", "java": "Java", "c++": "C++", "c#": "C#",
    "javascript": "JavaScript", "java script": "JavaScript",   # split-form alias
    "typescript": "TypeScript",  "type script": "TypeScript",   # split-form alias
    "html": "HTML", "css": "CSS", "sql": "SQL", "r": "R",
    "golang": "Go", "go": "Go", "rust": "Rust", "php": "PHP",
    "ruby": "Ruby", "swift": "Swift", "kotlin": "Kotlin", "scala": "Scala",
    "bash": "Bash", "powershell": "PowerShell", "shell": "Shell Scripting",
    "c language": "C", "c programming": "C",

    # Web & API Frameworks
    "react": "React", "react.js": "React", "reactjs": "React",
    "angular": "Angular", "angularjs": "Angular",
    "vue": "Vue.js", "vue.js": "Vue.js", "vuejs": "Vue.js",
    "node.js": "Node.js", "nodejs": "Node.js", "node js": "Node.js",
    "express": "Express.js", "expressjs": "Express.js",
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "fast api": "FastAPI",
    "spring": "Spring Boot", "spring boot": "Spring Boot",
    "asp.net": "ASP.NET", "aspnet": "ASP.NET",
    "next.js": "Next.js", "nextjs": "Next.js", "next js": "Next.js",
    "bootstrap": "Bootstrap", "tailwind": "Tailwind CSS",
    "rest api": "REST APIs", "restful api": "REST APIs", "rest apis": "REST APIs",
    "graphql": "GraphQL", "graph ql": "GraphQL",
    "microservices": "Microservices", "micro services": "Microservices",

    # AI / ML / Data
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "artificial intelligence": "AI", "ai": "AI",
    "nlp": "NLP", "natural language processing": "NLP",
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "pytorch": "PyTorch", "py torch": "PyTorch",
    "tensorflow": "TensorFlow", "tensor flow": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn", "scikit learn": "Scikit-Learn",
    "pandas": "Pandas", "numpy": "NumPy", "num py": "NumPy",
    "opencv": "OpenCV", "open cv": "OpenCV",
    "spark": "Apache Spark", "apache spark": "Apache Spark",
    "hadoop": "Hadoop",
    "power bi": "Power BI", "powerbi": "Power BI",
    "tableau": "Tableau",
    "data science": "Data Science", "data analysis": "Data Analysis",
    "data engineering": "Data Engineering",

    # Databases & Storage
    "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mysql": "MySQL", "my sql": "MySQL",
    "mongodb": "MongoDB", "mongo db": "MongoDB", "mongo": "MongoDB",
    "redis": "Redis",
    "sqlite": "SQLite", "sql lite": "SQLite",
    "oracle": "Oracle DB",
    "snowflake": "Snowflake", "bigquery": "BigQuery", "big query": "BigQuery",
    "sql server": "SQL Server", "mssql": "SQL Server",
    "dynamodb": "DynamoDB", "dynamo db": "DynamoDB",
    "cassandra": "Cassandra", "elasticsearch": "Elasticsearch",

    # Cloud & DevOps
    "aws": "AWS", "amazon web services": "AWS",
    "azure": "Azure", "microsoft azure": "Azure",
    "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
    "docker": "Docker",
    "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "terraform": "Terraform",
    "ci/cd": "CI/CD", "cicd": "CI/CD", "ci cd": "CI/CD",
    "git": "Git", "github": "GitHub", "git hub": "GitHub",
    "gitlab": "GitLab", "git lab": "GitLab",
    "jenkins": "Jenkins", "linux": "Linux", "ansible": "Ansible",
}

CERTIFICATION_PATTERNS = [
    r"\b(aws\s+certified[^\n,.]*)\b",
    r"\b(azure\s+certified[^\n,.]*)\b",
    r"\b(google\s+cloud\s+certified[^\n,.]*)\b",
    r"\b(pmp|prince2)\b",
    r"\b(cissp|cism|ceh)\b",
    r"\b(cka|ckad)\b",
    r"\b(certified\s+[a-zA-Z0-9\s]+)\b"
]

DEGREE_PATTERNS = [
    (r"\b(ph\.?d|doctorate|doctor\s+of\s+philosophy)\b", "Ph.D."),
    (r"\b(m\.?tech|master\s+of\s+technology|m\.?e\.?|master\s+of\s+engineering)\b", "M.Tech / M.E."),
    (r"\b(b\.?tech|bachelor\s+of\s+technology|b\.?e\.?|bachelor\s+of\s+engineering)\b", "B.Tech / B.E."),
    (r"\b(m\.?s\.?|master\s+of\s+science|m\.?sc\.?)\b", "M.Sc / M.S."),
    (r"\b(b\.?s\.?|bachelor\s+of\s+science|b\.?sc\.?)\b", "B.Sc / B.S."),
    (r"\b(m\.?c\.?a\.?|master\s+of\s+computer\s+applications)\b", "MCA"),
    (r"\b(b\.?c\.?a\.?|bachelor\s+of\s+computer\s+applications)\b", "BCA"),
    (r"\b(m\.?b\.?a\.?|master\s+of\s+business\s+administration)\b", "MBA"),
]

def extract_all_skills_from_text(text):
    """Extract known skills from text using boundary-exact regex matching."""
    if not text:
        return []
    text_lower = text.lower()
    extracted = set()
    for key, display in SKILL_DATABASE.items():
        pattern = r'(?<![a-zA-Z0-9#+])' + re.escape(key) + r'(?![a-zA-Z0-9#+])'
        if re.search(pattern, text_lower):
            extracted.add(display)
    return sorted(list(extracted))

def parse_job_description(jd_text):
    """
    Parses Job Description text into structured categories:
    - required_skills (Must-have)
    - preferred_skills (Nice-to-have)
    - min_years_experience
    - target_roles
    - responsibilities
    - education_required
    - certifications_required
    """
    if not jd_text or not jd_text.strip():
        return {
            "required_skills": [],
            "preferred_skills": [],
            "min_years_experience": 0.0,
            "target_roles": [],
            "responsibilities": [],
            "education_required": [],
            "certifications_required": [],
            "raw_text": ""
        }

    lines = [l.strip() for l in jd_text.split("\n") if l.strip()]

    # 1. Split sections into Required vs Preferred vs Responsibilities
    req_lines = []
    pref_lines = []
    resp_lines = []
    general_lines = []

    current_mode = "general"
    for line in lines:
        line_lower = line.lower()
        if re.search(r'\b(must\s+have|required|essential|requirements|minimum\s+qualifications)\b', line_lower):
            current_mode = "required"
            continue
        elif re.search(r'\b(preferred|nice\s+to\s+have|bonus|plus|desired|good\s+to\s+have)\b', line_lower):
            current_mode = "preferred"
            continue
        elif re.search(r'\b(responsibilities|duties|what\s+you\s+will\s+do|key\s+deliverables|role\s+overview)\b', line_lower):
            current_mode = "responsibilities"
            continue

        if current_mode == "required":
            req_lines.append(line)
        elif current_mode == "preferred":
            pref_lines.append(line)
        elif current_mode == "responsibilities":
            resp_lines.append(line)
        else:
            general_lines.append(line)

    req_text = " ".join(req_lines)
    pref_text = " ".join(pref_lines)
    all_text = jd_text

    req_skills = extract_all_skills_from_text(req_text)
    pref_skills = extract_all_skills_from_text(pref_text)
    all_skills = extract_all_skills_from_text(all_text)

    # If no explicit "Required" section was partitioned, default skills found to Required
    if not req_skills and not pref_skills:
        req_skills = all_skills
        pref_skills = []
    else:
        # Avoid duplicate overlap between required & preferred
        pref_skills = [s for s in pref_skills if s not in req_skills]
        # Any leftover skills from general text defaulted to required
        leftover = [s for s in all_skills if s not in req_skills and s not in pref_skills]
        req_skills.extend(leftover)

    # 2. Extract Min Years Experience required
    exp_matches = re.findall(
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|relevant\s+experience)',
        all_text,
        re.IGNORECASE
    )
    min_exp = 0.0
    if exp_matches:
        try:
            exp_vals = [float(x) for x in exp_matches if 0 <= float(x) <= 30]
            if exp_vals:
                min_exp = max(exp_vals)
        except Exception:
            pass

    # 3. Target Job Roles
    target_roles = []
    role_pats = [
        r'\b(Backend\s+Developer|Backend\s+Engineer|Frontend\s+Developer|Full\s+Stack\s+Developer)\b',
        r'\b(Software\s+Engineer|Data\s+Scientist|Machine\s+Learning\s+Engineer|DevOps\s+Engineer)\b',
        r'\b(Cloud\s+Architect|Data\s+Engineer|QA\s+Engineer|Security\s+Engineer)\b'
    ]
    for pat in role_pats:
        m = re.search(pat, all_text, re.IGNORECASE)
        if m and m.group(0).title() not in target_roles:
            target_roles.append(m.group(0).title())

    # 4. Responsibilities Extraction
    responsibilities = []
    if resp_lines:
        responsibilities = [r.lstrip("•-*123456789. ") for r in resp_lines if len(r) > 10]
    else:
        # Extract bullet points from full text
        bullets = [l.lstrip("•-*123456789. ") for l in lines if l.startswith(("•", "-", "*", "1.", "2.", "3."))]
        responsibilities = [b for b in bullets if len(b) > 15][:6]

    # 5. Education Requirements
    education_req = []
    for pat, name in DEGREE_PATTERNS:
        if re.search(pat, all_text, re.IGNORECASE):
            if name not in education_req:
                education_req.append(name)

    # 6. Certifications Required
    certs_req = []
    for pat in CERTIFICATION_PATTERNS:
        matches = re.findall(pat, all_text, re.IGNORECASE)
        for m in matches:
            c_clean = m.strip().title()
            if c_clean not in certs_req:
                certs_req.append(c_clean)

    return {
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "min_years_experience": min_exp,
        "target_roles": target_roles,
        "responsibilities": responsibilities,
        "education_required": education_req,
        "certifications_required": certs_req,
        "raw_text": jd_text
    }
