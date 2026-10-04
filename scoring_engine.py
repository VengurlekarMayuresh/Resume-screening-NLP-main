import re
import numpy as np
from bge_m3_engine import get_embedding, compute_cosine_similarity

# Related Technology Families for Anti-False-Positive Handling
RELATED_TECH_FAMILY = {
    "PostgreSQL": ["MySQL", "SQLite", "Oracle DB", "SQL Server", "SQL", "MongoDB"],
    "MySQL": ["PostgreSQL", "SQLite", "Oracle DB", "SQL Server", "SQL"],
    "AWS": ["Azure", "GCP", "Google Cloud"],
    "Azure": ["AWS", "GCP"],
    "GCP": ["AWS", "Azure"],
    "React": ["Vue.js", "Angular"],
    "Angular": ["React", "Vue.js"],
    "Vue.js": ["React", "Angular"],
    "Java": ["Kotlin", "C#", "Scala"],
    "JavaScript": ["TypeScript"],
    "TypeScript": ["JavaScript"],
    "Python": ["R"],
    "Deep Learning": ["Machine Learning", "AI"],
    "Machine Learning": ["Data Science", "Deep Learning"],
    "Docker": ["Kubernetes", "Containerization"],
    "Kubernetes": ["Docker"],
    "FastAPI": ["Flask", "Django"],
    "Django": ["Flask", "FastAPI"],
    "Flask": ["FastAPI", "Django"]
}

def extract_evidence_snippets(resume_text, targets, max_snippets=4):
    """
    Extract verbatim sentence evidence from resume text matching target keywords/skills.
    Ensures explainability is strictly grounded in actual resume text.
    """
    if not resume_text or not targets:
        return []

    # Split resume text into sentences or clean bullet lines
    lines = [s.strip() for s in re.split(r'[\n.•;]', resume_text) if len(s.strip()) > 15]

    evidence_map = []
    seen = set()

    for target in targets:
        target_lower = target.lower()
        for line in lines:
            line_lower = line.lower()
            if target_lower in line_lower and line not in seen:
                evidence_map.append({
                    "target": target,
                    "quote": line.strip()
                })
                seen.add(line)
                break
        if len(evidence_map) >= max_snippets:
            break

    return evidence_map

def classify_skill_matches(required_skills, candidate_skills, resume_text):
    """
    Anti-False-Positive Skill Classifier:
    - Exact Match (✓): Skill explicitly found in candidate skills.
    - Related / Partial Match (◐): Related tech found (e.g. MySQL when PostgreSQL required).
    - Missing (✗): Neither exact nor related skill found.
    """
    matched = []
    partial = []
    missing = []

    cand_skills_lower = [s.lower() for s in candidate_skills]
    # Normalize newlines → spaces so multi-word skills split across PDF lines are found
    resume_text_lower = re.sub(r'\r?\n', ' ', resume_text).lower()
    resume_text_lower = re.sub(r'[ \t]+', ' ', resume_text_lower)

    for skill in required_skills:
        skill_lower = skill.lower()
        # 1. Check exact match
        pattern = r'(?<![a-zA-Z0-9#+])' + re.escape(skill_lower) + r'(?![a-zA-Z0-9#+])'
        if skill_lower in cand_skills_lower or re.search(pattern, resume_text_lower):
            matched.append(skill)
        else:
            # 2. Check related technology family
            related_found = []
            family = RELATED_TECH_FAMILY.get(skill, [])
            for rel in family:
                rel_pattern = r'(?<![a-zA-Z0-9#+])' + re.escape(rel.lower()) + r'(?![a-zA-Z0-9#+])'
                if rel.lower() in cand_skills_lower or re.search(rel_pattern, resume_text_lower):
                    related_found.append(rel)

            if related_found:
                partial.append({
                    "skill": skill,
                    "found_related": related_found[0],
                    "note": f"{related_found[0]} detected — {skill} not explicitly confirmed."
                })
            else:
                missing.append(skill)

    return matched, partial, missing

def evaluate_candidate(jd_profile, candidate_profile, bge_model=None):
    """
    Evaluates candidate against JD profile across 7 explainable signals:
    1. Semantic Relevance (BGE-M3): 20%
    2. Required Skill Match: 25%
    3. Preferred Skill Match: 10%
    4. Experience Match: 15%
    5. Responsibility Match: 15%
    6. Project Relevance: 10%
    7. Education/Cert Match: 5%
    Total = 100%
    """
    raw_resume = candidate_profile["raw_text"]
    raw_jd = jd_profile["raw_text"]

    # 1. BGE-M3 Semantic Relevance (20%)
    jd_emb = get_embedding(raw_jd, model=bge_model)
    res_emb = get_embedding(raw_resume, model=bge_model)
    semantic_sim = compute_cosine_similarity(jd_emb, res_emb)
    semantic_score = float(np.round(semantic_sim * 100, 1))

    # 2. Required Skills Match (25%)
    req_skills = jd_profile.get("required_skills", [])
    cand_skills = candidate_profile.get("skills", [])
    req_matched, req_partial, req_missing = classify_skill_matches(req_skills, cand_skills, raw_resume)

    if req_skills:
        # Full points for exact match, half points for related/partial match
        req_score_val = (len(req_matched) + (0.5 * len(req_partial))) / len(req_skills)
        req_score = float(np.round(min(1.0, req_score_val) * 100, 1))
    else:
        req_score = semantic_score

    # 3. Preferred Skills Match (10%)
    pref_skills = jd_profile.get("preferred_skills", [])
    pref_matched, pref_partial, pref_missing = classify_skill_matches(pref_skills, cand_skills, raw_resume)

    if pref_skills:
        pref_score_val = (len(pref_matched) + (0.5 * len(pref_partial))) / len(pref_skills)
        pref_score = float(np.round(min(1.0, pref_score_val) * 100, 1))
    else:
        pref_score = 100.0 if not pref_skills else semantic_score

    # 4. Experience Match (15%)
    jd_min_exp = jd_profile.get("min_years_experience", 0.0)
    cand_exp = candidate_profile.get("years_of_experience", 0.0)

    if jd_min_exp == 0.0:
        exp_score = 100.0
        exp_explanation = f"No minimum experience required (Candidate has {cand_exp} yrs)."
    elif cand_exp >= jd_min_exp:
        exp_score = 100.0
        exp_explanation = f"Meets required experience ({cand_exp} yrs vs {jd_min_exp} yrs required)."
    elif cand_exp > 0:
        ratio = cand_exp / jd_min_exp
        exp_score = float(np.round(ratio * 100, 1))
        exp_explanation = f"Partial experience ({cand_exp} yrs vs {jd_min_exp} yrs required)."
    else:
        exp_score = 40.0 if "fresher" in candidate_profile.get("experience_display", "").lower() else 20.0
        exp_explanation = f"Below required experience ({candidate_profile.get('experience_display')} vs {jd_min_exp} yrs required)."

    # 5. Responsibility Match (15%)
    resp_list = jd_profile.get("responsibilities", [])
    exp_text = candidate_profile.get("experience_text", "")
    if resp_list and exp_text:
        resp_text = " ".join(resp_list)
        resp_jd_emb = get_embedding(resp_text, model=bge_model)
        exp_res_emb = get_embedding(exp_text, model=bge_model)
        resp_sim = compute_cosine_similarity(resp_jd_emb, exp_res_emb)
        resp_score = float(np.round(resp_sim * 100, 1))
    else:
        resp_score = semantic_score

    # 6. Project Relevance (10%)
    proj_text = candidate_profile.get("projects_text", "")
    if proj_text:
        proj_jd_emb = get_embedding(raw_jd, model=bge_model)
        proj_res_emb = get_embedding(proj_text, model=bge_model)
        proj_sim = compute_cosine_similarity(proj_jd_emb, proj_res_emb)
        proj_score = float(np.round(proj_sim * 100, 1))
    else:
        proj_score = semantic_score

    # 7. Education & Certification Match (5%)
    edu_req = jd_profile.get("education_required", [])
    cand_degrees = candidate_profile.get("degrees", [])

    if not edu_req:
        edu_score = 100.0
    else:
        matched_edu = [d for d in edu_req if any(d.lower() in cd.lower() for cd in cand_degrees)]
        edu_score = 100.0 if matched_edu else 50.0

    # Calculate Final Explainable Weighted Score (0 to 100%)
    w_semantic = 0.20
    w_req_skill = 0.25
    w_pref_skill = 0.10
    w_exp = 0.15
    w_resp = 0.15
    w_proj = 0.10
    w_edu = 0.05

    final_score = (
        (semantic_score * w_semantic) +
        (req_score * w_req_skill) +
        (pref_score * w_pref_skill) +
        (exp_score * w_exp) +
        (resp_score * w_resp) +
        (proj_score * w_proj) +
        (edu_score * w_edu)
    )

    final_score_pct = float(np.round(final_score, 1))

    # Weight contributions breakdown in points (out of 100)
    contrib_semantic = round(semantic_score * w_semantic, 1)
    contrib_req_skill = round(req_score * w_req_skill, 1)
    contrib_pref_skill = round(pref_score * w_pref_skill, 1)
    contrib_exp = round(exp_score * w_exp, 1)
    contrib_resp = round(resp_score * w_resp, 1)
    contrib_proj = round(proj_score * w_proj, 1)
    contrib_edu = round(edu_score * w_edu, 1)

    # Status classification
    if final_score_pct >= 80.0:
        status = "Strong Match"
    elif final_score_pct >= 60.0:
        status = "Moderate Match"
    else:
        status = "Weak Match"

    # Evidence Grounding Snippets
    all_matched_targets = req_matched + [p["skill"] for p in req_partial] + pref_matched
    evidence_snippets = extract_evidence_snippets(raw_resume, all_matched_targets)

    # Breakdown Table Data
    breakdown = [
        {"category": "Semantic Relevance", "score": f"{semantic_score}%", "pts": f"+{contrib_semantic}", "explanation": "BGE-M3 embedding semantic alignment between JD and Resume text."},
        {"category": "Required Skills", "score": f"{req_score}%", "pts": f"+{contrib_req_skill}", "explanation": f"Matched {len(req_matched)}/{len(req_skills)} required skills ({len(req_partial)} related)."},
        {"category": "Preferred Skills", "score": f"{pref_score}%", "pts": f"+{contrib_pref_skill}", "explanation": f"Matched {len(pref_matched)}/{len(pref_skills)} preferred skills."},
        {"category": "Experience", "score": f"{exp_score}%", "pts": f"+{contrib_exp}", "explanation": exp_explanation},
        {"category": "Responsibilities", "score": f"{resp_score}%", "pts": f"+{contrib_resp}", "explanation": "Responsibility alignment with candidate's professional work history."},
        {"category": "Projects", "score": f"{proj_score}%", "pts": f"+{contrib_proj}", "explanation": "Project technical alignment with role requirements."},
        {"category": "Education & Certs", "score": f"{edu_score}%", "pts": f"+{contrib_edu}", "explanation": f"Degree match ({candidate_profile.get('degree_display')})."}
    ]

    return {
        "overall_score": final_score_pct,
        "status": status,
        "breakdown": breakdown,
        "scores": {
            "semantic": semantic_score,
            "req_skill": req_score,
            "pref_skill": pref_score,
            "experience": exp_score,
            "responsibilities": resp_score,
            "projects": proj_score,
            "education": edu_score
        },
        "req_matched": req_matched,
        "req_partial": req_partial,
        "req_missing": req_missing,
        "pref_matched": pref_matched,
        "pref_partial": pref_partial,
        "pref_missing": pref_missing,
        "evidence_snippets": evidence_snippets,
        "candidate_profile": candidate_profile
    }
