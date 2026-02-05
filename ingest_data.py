import csv
import os
import numpy as np
import faiss
from embedder import embed_batch, get_device
from faiss_index import build_index, save_index

def ingest():
    input_csv = "job_title_des.csv"
    output_csv = "jobs.csv"
    output_index = "jobs.index"

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in current directory.")
        return

    print(f"Starting ingestion of {input_csv} using device: {get_device()}...")

    # 1. Read and map CSV
    jobs = []
    descriptions = []
    
    with open(input_csv, mode='r', encoding='utf-8') as f:
        # The CSV has headers: ,Job Title,Job Description
        # The first column is an index.
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            job = {
                "id": str(i),
                "title": row["Job Title"].strip(),
                "description": row["Job Description"].strip()
            }
            if job["description"]:
                jobs.append(job)
                descriptions.append(job["description"])

    print(f"Loaded {len(jobs)} jobs. Generating embeddings...")

    # 2. Batch embed descriptions
    # Using 64 as default batch size
    vectors = embed_batch(descriptions, batch_size=64)
    print(f"Generated embeddings for {len(vectors)} jobs. Vector shape: {vectors.shape}")

    # 3. Build and save FAISS index
    index = build_index(vectors)
    save_index(index, output_index)
    print(f"Saved FAISS index to {output_index}")

    # 4. Save mapped metadata to jobs.csv
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "description"])
        writer.writeheader()
        writer.writerows(jobs)
    
    print(f"Saved metadata to {output_csv}")
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest()
