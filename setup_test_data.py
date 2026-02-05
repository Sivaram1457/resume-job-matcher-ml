import csv
import numpy as np
import faiss
from embedder import embed_batch
from faiss_index import build_index, save_index

def setup():
    # 1. Create sample jobs
    jobs = [
        {"id": "0", "title": "Python Developer", "description": "Experienced in FastAPI, PostgreSQL, and building scalable backends."},
        {"id": "1", "title": "Machine Learning Engineer", "description": "Expert in PyTorch, transformers, and NLP embedding systems."},
        {"id": "2", "title": "DevOps Engineer", "description": "Cloud fundamentals, AWS, Docker, and CI/CD pipelines."},
        {"id": "3", "title": "Frontend Developer", "description": "React, Tailwind CSS, and modern UI/UX design."},
        {"id": "4", "title": "Data Scientist", "description": "Statistical modeling, data visualization, and predictive analytics."}
    ]

    csv_path = "jobs.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "description"])
        writer.writeheader()
        writer.writerows(jobs)
    
    print(f"Created {csv_path}")

    # 2. Generate embeddings and FAISS index
    descriptions = [j['description'] for j in jobs]
    vectors = embed_batch(descriptions)
    
    index = build_index(vectors)
    save_index(index, "jobs.index")
    print("Created jobs.index")

if __name__ == "__main__":
    setup()
