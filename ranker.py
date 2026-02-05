import csv
import os
from typing import List, Dict

# Import existing modules
from parser import parse_resume
from embedder import embed_text
from faiss_index import load_index, search

def load_jobs(csv_path: str) -> List[Dict]:
    """
    Reads job metadata from a CSV file.
    Expected columns: id, title, description
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Job metadata CSV not found at: {csv_path}")
        
    jobs = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(row)
    return jobs

def recommend_jobs(
    resume_path: str, 
    csv_path: str = "jobs.csv", 
    index_path: str = "jobs.index", 
    k: int = 5
) -> List[Dict]:
    """
    Orchestrates the matching pipeline:
    1. Parse resume to clean text.
    2. Embed the cleaned text.
    3. Search the FAISS index.
    4. Match indices to job metadata.
    """
    # 1. Parse resume
    clean_text = parse_resume(resume_path)
    if clean_text.startswith("Error:"):
        print(f"Parsing failed: {clean_text}")
        return []

    # 2. Embed text
    query_vector = embed_text(clean_text)

    # 3. Load index and search
    index = load_index(index_path)
    scores, ids = search(index, query_vector, k=k)

    # 4. Load metadata and filter
    all_jobs = load_jobs(csv_path)
    
    recommendations = []
    for score, idx in zip(scores[0], ids[0]):
        if idx != -1 and idx < len(all_jobs):
            job = all_jobs[idx].copy()
            job['similarity_score'] = float(score)
            recommendations.append(job)
            
    return recommendations

if __name__ == "__main__":
    # Test block
    RESUME_PATH = r"E:\resume matcher\resume (4).pdf"
    CSV_PATH = "jobs.csv"
    INDEX_PATH = "jobs.index"

    if os.path.exists(CSV_PATH) and os.path.exists(INDEX_PATH):
        print(f"Searching for top matches for: {RESUME_PATH}\n")
        results = recommend_jobs(RESUME_PATH, CSV_PATH, INDEX_PATH, k=3)
        
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['title']} (Score: {res['similarity_score']:.4f})")
            print(f"   Desc: {res['description'][:100]}...\n")
    else:
        print("Required test files (jobs.csv, jobs.index) missing. Please run setup_test_data.py first.")
