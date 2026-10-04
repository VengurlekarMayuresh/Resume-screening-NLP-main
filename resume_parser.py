import re
from datetime import datetime
from pypdf import PdfReader
from jd_parser import extract_all_skills_from_text, DEGREE_PATTERNS


# Tech terms whose CamelCase must NOT be split (e.g. JavaScript → java script)
_PROTECTED_TERMS = [
    "JavaScript", "TypeScript", "TensorFlow", "PyTorch", "NumPy", "SciPy",
    "OpenCV", "FastAPI", "MongoDB", "DynamoDB", "BigQuery", "GraphQL",
    "PostgreSQL", "SQLite", "MySQL", "GitHub", "GitLab", "LinkedIn",
    "PowerShell", "PowerBI", "NextJS", "ReactJS", "VueJS", "AngularJS",
    "NodeJS", "ExpressJS", "Scikit", "DevOps", "CI/CD",
]

def sanitize_resume_text(raw_text):
    """
    Cleans PDF extraction artifacts:
    1. Protects known CamelCase tech brand names from being split
    2. Despaces kerning/letter-spaced text ('B t e c h' -> 'Btech')
    3. Separates squished camel-case column boundaries
    4. Normalizes whitespace and non-standard bullets
    """
    if not raw_text:
        return ""
    text = raw_text

    # Step 1: Temporarily replace protected CamelCase terms with placeholders
    placeholders = {}
    for term in _PROTECTED_TERMS:
        placeholder = f"__PROT_{term.upper().replace('/', '_')}__"
        placeholders[placeholder] = term
        # Case-insensitive replace of the original term
        text = re.sub(re.escape(term), placeholder, text, flags=re.IGNORECASE)

    # Step 2: De-space kerning artifacts ('J a v a' -> 'Java')
    for _ in range(3):
        text = re.sub(r'(?<=[a-zA-Z0-9])\s(?=[a-zA-Z0-9]\b)', '', text)

    # Step 3: Collapse newlines between word characters into a space
    # Fixes: 'Machine\nLearning' -> 'Machine Learning' (PDF column layout split)
    text = re.sub(r'(?<=[a-zA-Z])\r?\n(?=[a-zA-Z])', ' ', text)

    # Step 4: Split genuinely squished camelCase words (not protected terms)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Step 5: Normalize whitespace (spaces/tabs only — keep intentional newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # Step 6: Restore protected terms
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)

    return text.strip()


def extract_text_from_pdf(file):
    """Extracts and cleans text from PDF resume file."""
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        cleaned = sanitize_resume_text(text)
        return cleaned if cleaned else "No readable text found."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_candidate_name(text):
    """Extract candidate name from header lines of resume."""
    if not text:
        return "Candidate"

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Common section headers to skip
    SKIP_HEADERS = {
        "resume", "curriculum vitae", "cv", "profile", "summary",
        "objective", "career objective", "professional summary",
        "skills", "experience", "education", "projects", "contact",
        "references", "certifications", "achievements", "declaration"
    }

    # Honorifics to strip from the front
    HONORIFIC_RE = re.compile(
        r"^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?)\s+", re.IGNORECASE
    )

    def is_valid_name(line):
        """Return cleaned name if line looks like a name, else None."""
        # Skip lines with contact info
        if re.search(
            r"(@|\.com|\.in|\.net|\.org|\d{5,}|mob|phone|tel:|linkedin|github|http|www\.|curriculum|resume)",
            line, re.IGNORECASE
        ):
            return None
        # Skip section headers
        if line.lower().strip(":-") in SKIP_HEADERS:
            return None
        # Strip honorific prefix if present
        cleaned = HONORIFIC_RE.sub("", line).strip()
        # Handle ALL-CAPS names (very common in Indian resumes): "NEHA SHARMA"
        if cleaned.isupper() and 2 <= len(cleaned.split()) <= 4:
            words = cleaned.split()
            if all(re.match(r"^[A-Z]+$", w) for w in words) and all(len(w) >= 2 for w in words):
                return cleaned.title()
        # Normal title-case / mixed-case name: 2–4 words, only alphabetic + allowed punct
        words = cleaned.split()
        if 2 <= len(words) <= 4:
            if all(re.match(r"^[A-Za-z][A-Za-z\'\-\.]*$", w) for w in words):
                # Reject if any word is a common English keyword (not a name)
                low_words = {w.lower() for w in words}
                if not low_words.intersection(SKIP_HEADERS):
                    return cleaned.title()
        return None

    # First pass: first 12 lines
    for line in lines[:12]:
        name = is_valid_name(line)
        if name:
            return name

    # Second pass: look for an isolated short line before a known section header
    for i, line in enumerate(lines[:20]):
        if line.lower().strip(":-") in SKIP_HEADERS and i > 0:
            name = is_valid_name(lines[i - 1])
            if name:
                return name

    return "Candidate Profile"

def segment_resume_sections(text):
    """Segment resume into summary, experience, projects, skills, education, certifications."""
    sections = {
        "header": "",
        "summary": "",
        "experience": "",
        "projects": "",
        "skills": "",
        "education": "",
        "certifications": "",
        "other": ""
    }
    if not text:
        return sections

    lines = text.split("\n")
    current_sec = "header"

    header_patterns = {
        "summary": r"^(?:summary|professional\s+summary|profile|about\s+me|overview)\b",
        "experience": r"^(?:work\s+experience|professional\s+experience|employment\s+history|experience|work\s+history)\b",
        "projects": r"^(?:projects|key\s+projects|academic\s+projects|personal\s+projects)\b",
        "skills": r"^(?:skills|technical\s+skills|core\s+competencies|technologies|technical\s+proficiency)\b",
        "education": r"^(?:education|academic\s+background|academics|qualifications|academic\s+history)\b",
        "certifications": r"^(?:certifications|certificates|licenses|courses)\b",
    }

    for idx, line in enumerate(lines):
        line_clean = line.strip().lower()
        if idx >= 1 and 2 <= len(line_clean) <= 35:
            matched = False
            for sec, pat in header_patterns.items():
                if re.match(pat, line_clean, re.IGNORECASE):
                    current_sec = sec
                    matched = True
                    break
            if matched:
                continue
        sections[current_sec] += "\n" + line

    return sections

def parse_full_resume(resume_text):
    """
    Parses resume text into a rich structured candidate profile:
    - candidate_name
    - skills (Technical & domain)
    - years_of_experience
    - experience_display
    - job_titles
    - degrees
    - degree_display
    - universities
    - university_display
    - sections (dict of segmented text)
    - projects_text
    - experience_text
    - raw_text
    """
    clean_text = sanitize_resume_text(resume_text)
    current_year = datetime.now().year
    candidate_name = extract_candidate_name(clean_text)
    sections = segment_resume_sections(clean_text)

    skills = extract_all_skills_from_text(clean_text)

    # Experience Years Calculation
    is_student = bool(re.search(r'\b(student|undergraduate|fresher|pursuing|enrolled|b\.?tech\s+student)\b', clean_text, re.IGNORECASE))
    future_grad = re.search(r'(?:–|-|to)\s*(?:[A-Za-z]+\s*)?(202[5-9]|203\d)', clean_text, re.IGNORECASE)

    date_ranges = re.findall(
        r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(20\d{2}|19\d{2})\s*(?:–|-|to)\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(20\d{2}|present|current)",
        clean_text,
        re.IGNORECASE
    )
    work_years = []
    for s_yr, e_yr in date_ranges:
        s = int(s_yr)
        e = current_year if e_yr.lower() in ["present", "current"] else int(e_yr)
        diff = e - s
        if 0 < diff <= 30 and e <= current_year:
            work_years.append(diff)

    explicit = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|work)", clean_text, re.IGNORECASE)
    exp_years = None
    for m in explicit:
        try:
            v = float(m)
            if 0.5 <= v <= 35:
                exp_years = v
                break
        except Exception:
            pass

    exp_display = "Not Detected"
    if exp_years is not None:
        years_of_exp = exp_years
        exp_display = f"{exp_years} yrs"
    elif work_years and not (is_student and len(work_years) <= 1):
        years_of_exp = float(sum(work_years))
        exp_display = f"{int(years_of_exp)} yrs"
    elif future_grad:
        grad_yr = future_grad.group(1)
        years_of_exp = 0.0
        exp_display = f"Fresher (Class of {grad_yr})"
    elif is_student:
        years_of_exp = 0.0
        exp_display = "Fresher / Student (0 yrs)"
    else:
        years_of_exp = 0.0
        exp_display = "0 yrs"

    # Degrees
    found_degrees = []
    for pat, name in DEGREE_PATTERNS:
        if re.search(pat, clean_text, re.IGNORECASE):
            spec = ""
            if name == "B.Tech / B.E." and re.search(r'\b(computer\s+engineering|computer\s+science|it|information\s+technology|ai|data\s+science)\b', clean_text, re.IGNORECASE):
                spec_m = re.search(r'\b(computer\s+engineering|computer\s+science|information\s+technology|artificial\s+intelligence)\b', clean_text, re.IGNORECASE)
                if spec_m:
                    spec = f" ({spec_m.group(0).title()})"
            disp = f"{name}{spec}"
            if disp not in found_degrees and name not in [d.split(" (")[0] for d in found_degrees]:
                found_degrees.append(disp)

    # Institutions / Universities
    INSTITUTE_PATTERNS = [
        r'\b(?:Indian\s+Institute\s+of\s+Technology(?:\s+[A-Za-z]+)?|National\s+Institute\s+of\s+Technology(?:\s+[A-Za-z]+)?|Birla\s+Institute\s+of\s+Technology(?:\s+[A-Za-z]+)?)\b',
        r'\b(?:IIT|NIT|BITS|IIIT|IIM|MIT|Stanford|Harvard|Oxford|Cambridge)\s+[A-Za-z]+\b',
        r'\b([A-Z][a-zA-Z\s]{2,30}(?:Institute\s+of\s+Technology|Engineering\s+College|University|College|Vidyalaya|Jr\s+College|Academy|Polytechnic)(?:,\s*[A-Z][a-zA-Z]+)?)\b',
    ]
    found_unis = []
    for line in clean_text.split("\n"):
        line = line.strip()
        for pat in INSTITUTE_PATTERNS:
            for m in re.finditer(pat, line, re.IGNORECASE):
                name = m.group(0).strip()
                name = re.sub(r'^[,\-\s•]+', '', name)
                if len(name) > 6 and name not in found_unis:
                    found_unis.append(name)
    sorted_unis = sorted(list(set(found_unis)), key=lambda x: -len(x))
    cleaned_unis = []
    for u in sorted_unis:
        if not any(u.lower() in ex.lower() for ex in cleaned_unis):
            cleaned_unis.append(u)

    # Job Titles / Roles
    found_titles = []
    TECH_ROLES = [
        r'\b(Software\s+Engineer|Full[-\s]?Stack\s+Developer|Frontend\s+Developer|Backend\s+Developer)\b',
        r'\b(Data\s+Scientist|Machine\s+Learning\s+Engineer|AI\s+Engineer|Data\s+Analyst)\b',
        r'\b(Computer\s+Engineering\s+Student|Engineering\s+Student)\b',
        r'\b(DevOps\s+Engineer|Cloud\s+Architect|Cybersecurity\s+Analyst|QA\s+Engineer)\b',
    ]
    for pat in TECH_ROLES:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            t = m.group(0).strip().title()
            if t not in found_titles:
                found_titles.append(t)

    return {
        "candidate_name": candidate_name,
        "skills": skills,
        "years_of_experience": years_of_exp,
        "experience_display": exp_display,
        "job_titles": found_titles,
        "degrees": found_degrees,
        "degree_display": found_degrees[0] if found_degrees else "Not Specified",
        "universities": cleaned_unis,
        "university_display": cleaned_unis[0] if cleaned_unis else "Not Specified",
        "sections": sections,
        "experience_text": sections.get("experience") or sections.get("summary") or clean_text,
        "projects_text": sections.get("projects") or clean_text,
        "raw_text": clean_text
    }
