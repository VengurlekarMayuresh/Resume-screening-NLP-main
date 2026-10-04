import json
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jd_parser import parse_job_description
from resume_parser import parse_full_resume
from scoring_engine import evaluate_candidate, classify_skill_matches

def compute_dcg_at_k(relevance_scores, k):
    """Compute Discounted Cumulative Gain at K."""
    relevance_scores = relevance_scores[:k]
    dcg = 0.0
    for idx, rel in enumerate(relevance_scores, start=1):
        dcg += (2**rel - 1) / math.log2(idx + 1)
    return dcg

def compute_ndcg_at_k(retrieved_rel, ideal_rel, k):
    """Compute Normalized Discounted Cumulative Gain at K."""
    dcg = compute_dcg_at_k(retrieved_rel, k)
    idcg = compute_dcg_at_k(ideal_rel, k)
    return dcg / idcg if idcg > 0 else 0.0

def compute_mrr(retrieved_rel):
    """Compute Mean Reciprocal Rank (first rank with relevance grade >= 2)."""
    for idx, rel in enumerate(retrieved_rel, start=1):
        if rel >= 2:
            return 1.0 / idx
    return 0.0

def legacy_tfidf_ranker(jd_text, resumes):
    """Legacy 60% Skill + 40% TF-IDF Cosine similarity scoring mechanism."""
    docs = [jd_text] + resumes
    vec = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vec.fit_transform(docs)
    job_v = tfidf_matrix[0]
    res_vs = tfidf_matrix[1:]
    sims = cosine_similarity(job_v, res_vs).flatten()

    jd_skills = parse_job_description(jd_text)["required_skills"]
    scores = []
    for i, res_text in enumerate(resumes):
        cand_skills = parse_full_resume(res_text)["skills"]
        if jd_skills:
            matched, _, _ = classify_skill_matches(jd_skills, cand_skills, res_text)
            skill_score = len(matched) / len(jd_skills)
        else:
            skill_score = float(sims[i])
        comp = (0.60 * skill_score) + (0.40 * float(sims[i]))
        scores.append(comp * 100)
    return scores

def run_evaluation(dataset_path="evaluation_dataset.json", bge_model=None, k=2):
    """
    Runs comprehensive evaluation comparing:
    1. Legacy TF-IDF Scoring
    2. Upgraded BGE-M3 Hybrid Scoring
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    legacy_metrics = {"precision": [], "recall": [], "f1": [], "ndcg": [], "mrr": []}
    bgem3_metrics = {"precision": [], "recall": [], "f1": [], "ndcg": [], "mrr": []}

    for case in eval_cases:
        jd_text = case["job_description"]
        candidates = case["candidates"]
        resumes = [c["resume_text"] for c in candidates]
        grades = [c["relevance_grade"] for c in candidates]

        total_relevant = sum(1 for g in grades if g >= 2)

        # --- 1. Evaluate Legacy TF-IDF ---
        legacy_scores = legacy_tfidf_ranker(jd_text, resumes)
        legacy_ranked = [g for _, g in sorted(zip(legacy_scores, grades), reverse=True)]

        legacy_rel_at_k = [1 if g >= 2 else 0 for g in legacy_ranked[:k]]
        p_leg = sum(legacy_rel_at_k) / k
        r_leg = sum(legacy_rel_at_k) / total_relevant if total_relevant > 0 else 0
        f1_leg = (2 * p_leg * r_leg) / (p_leg + r_leg) if (p_leg + r_leg) > 0 else 0
        ndcg_leg = compute_ndcg_at_k(legacy_ranked, sorted(grades, reverse=True), k)
        mrr_leg = compute_mrr(legacy_ranked)

        legacy_metrics["precision"].append(p_leg)
        legacy_metrics["recall"].append(r_leg)
        legacy_metrics["f1"].append(f1_leg)
        legacy_metrics["ndcg"].append(ndcg_leg)
        legacy_metrics["mrr"].append(mrr_leg)

        # --- 2. Evaluate Upgraded BGE-M3 Hybrid ---
        jd_profile = parse_job_description(jd_text)
        bgem3_scores = []
        for res_text in resumes:
            cand_profile = parse_full_resume(res_text)
            res = evaluate_candidate(jd_profile, cand_profile, bge_model=bge_model)
            bgem3_scores.append(res["overall_score"])

        bgem3_ranked = [g for _, g in sorted(zip(bgem3_scores, grades), reverse=True)]

        bgem3_rel_at_k = [1 if g >= 2 else 0 for g in bgem3_ranked[:k]]
        p_bge = sum(bgem3_rel_at_k) / k
        r_bge = sum(bgem3_rel_at_k) / total_relevant if total_relevant > 0 else 0
        f1_bge = (2 * p_bge * r_bge) / (p_bge + r_bge) if (p_bge + r_bge) > 0 else 0
        ndcg_bge = compute_ndcg_at_k(bgem3_ranked, sorted(grades, reverse=True), k)
        mrr_bge = compute_mrr(bgem3_ranked)

        bgem3_metrics["precision"].append(p_bge)
        bgem3_metrics["recall"].append(r_bge)
        bgem3_metrics["f1"].append(f1_bge)
        bgem3_metrics["ndcg"].append(ndcg_bge)
        bgem3_metrics["mrr"].append(mrr_bge)

    comparison = pd.DataFrame({
        "Evaluation Metric": [f"Precision@{k}", f"Recall@{k}", f"F1-Score@{k}", f"NDCG@{k}", "MRR (Mean Reciprocal Rank)"],
        "Legacy TF-IDF Score": [
            f"{np.mean(legacy_metrics['precision'])*100:.1f}%",
            f"{np.mean(legacy_metrics['recall'])*100:.1f}%",
            f"{np.mean(legacy_metrics['f1'])*100:.1f}%",
            f"{np.mean(legacy_metrics['ndcg'])*100:.1f}%",
            f"{np.mean(legacy_metrics['mrr'])*100:.1f}%"
        ],
        "Upgraded BGE-M3 Hybrid Score": [
            f"{np.mean(bgem3_metrics['precision'])*100:.1f}%",
            f"{np.mean(bgem3_metrics['recall'])*100:.1f}%",
            f"{np.mean(bgem3_metrics['f1'])*100:.1f}%",
            f"{np.mean(bgem3_metrics['ndcg'])*100:.1f}%",
            f"{np.mean(bgem3_metrics['mrr'])*100:.1f}%"
        ]
    })

    return comparison

if __name__ == "__main__":
    print("Running system benchmark evaluation...")
    df = run_evaluation()
    print(df.to_string(index=False))
