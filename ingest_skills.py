import csv
import os
import re
import hashlib
import numpy as np
import faiss
import tempfile
from embedder import embed_batch, get_device
from faiss_index import build_index, save_index, load_index
from typing import List, Dict, Generator

# --- STORAGE PROTECTION ---
# Force Hugging Face and Transformers to use E drive for models
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"e:\resume matcher\.cache\huggingface"
os.environ["HF_HOME"] = r"e:\resume matcher\.cache\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"e:\resume matcher\.cache\huggingface"
# Force Python temp files to E drive
os.environ["TEMP"] = r"e:\resume matcher\.tmp"
os.environ["TMP"] = r"e:\resume matcher\.tmp"

# Create directories if they don't exist
os.makedirs(r"e:\resume matcher\.cache\huggingface", exist_ok=True)
os.makedirs(r"e:\resume matcher\.tmp", exist_ok=True)

def extract_title_from_url(url: str) -> str:
    """
    Extracts job title from LinkedIn view URL.
    Example: https://www.linkedin.com/jobs/view/housekeeper-i-pt-at-jacksonville...
    -> Housekeeper I Pt
    """
    try:
        # Regex to find the part between 'view/' and '-at-' or the end of the slug
        match = re.search(r'view/([^/?#]+)', url)
        if match:
            slug = match.group(1)
            # Remove trailing "-at-..." if present
            title_part = re.split(r'-at-', slug)[0]
            # Replace hyphens with spaces and capitalize
            title = title_part.replace('-', ' ').title()
            return title
    except Exception:
        pass
    return "Unknown Job Title"

def stream_csv(path: str) -> Generator[Dict, None, None]:
    """
    Streams CSV rows one by one to save memory.
    """
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def ingest_large_dataset(additional_limit: int = 50000):
    input_csv = "job_skills.csv"
    output_metadata = "jobs.csv"
    output_index = "jobs.index"
    
    # QUALITY Filters
    MIN_SKILLS_LENGTH = 30 
    BATCH_SIZE = 1000 

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Starting INCREMENTAL ingestion with DEDUPLICATION and QUALITY FILTERS.")
    print(f"Using device: {get_device()}. Adding {additional_limit} NEW unique jobs.")

    index = None
    seen_hashes = set()
    existing_jobs = []
    
    # 1. Load existing data if available
    if os.path.exists(output_metadata) and os.path.exists(output_index):
        print("Loading existing metadata and index...")
        index = load_index(output_index)
        with open(output_metadata, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_jobs.append(row)
                # Rebuild hashes for global deduplication
                h = hashlib.md5(f"{row['title']}|{row['description']}".encode('utf-8')).hexdigest()
                seen_hashes.add(h)
        print(f"Resuming with {len(existing_jobs)} existing unique entries.")
    else:
        print("No existing index found. Starting fresh.")

    total_processed = len(existing_jobs)
    new_jobs_processed = 0
    total_scanned = 0
    
    batch_texts = []
    batch_metadata = []
    
    # We'll use 'a' (append) mode if we are resuming
    file_mode = 'a' if existing_jobs else 'w'
    
    with open(output_metadata, mode=file_mode, newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["id", "title", "description"])
        if file_mode == 'w':
            writer.writeheader()

        for i, row in enumerate(stream_csv(input_csv)):
            total_scanned += 1
            url = row.get("job_link", "")
            skills = row.get("job_skills", "").strip()
            
            # Quality Filter
            if not skills or skills == 'None' or len(skills) < MIN_SKILLS_LENGTH:
                continue
                
            title = extract_title_from_url(url)
            if title == "Unknown Job Title":
                continue
            
            # Deduplication
            data_hash = hashlib.md5(f"{title}|{skills}".encode('utf-8')).hexdigest()
            if data_hash in seen_hashes:
                continue
            seen_hashes.add(data_hash)
            
            # New Job metadata
            job_meta = {
                "id": str(total_processed + new_jobs_processed),
                "title": title,
                "description": skills 
            }
            
            batch_texts.append(skills)
            batch_metadata.append(job_meta)
            
            # When batch is full, embed and add to index
            if len(batch_texts) >= BATCH_SIZE:
                print(f"Embedding batch of {BATCH_SIZE} (Total New: {new_jobs_processed + BATCH_SIZE})...")
                vectors = embed_batch(batch_texts, batch_size=64) 
                
                if index is None:
                    dimension = vectors.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                
                index.add(vectors.astype("float32"))
                writer.writerows(batch_metadata)
                
                new_jobs_processed += len(batch_texts)
                batch_texts = []
                batch_metadata = []
                
            if new_jobs_processed >= additional_limit:
                print(f"Reached additional limit of {additional_limit}. Total: {total_processed + new_jobs_processed}")
                break

            if total_scanned % 10000 == 0:
                 print(f"Status: Scanned {total_scanned} rows, New processing {new_jobs_processed} unique...")

        # Process final partial batch
        if batch_texts:
            print(f"Embedding last batch of {len(batch_texts)}...")
            vectors = embed_batch(batch_texts, batch_size=64)
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors.astype("float32"))
            writer.writerows(batch_metadata)
            new_jobs_processed += len(batch_texts)

    if index:
        save_index(index, output_index)
        print(f"Deduplication summary: Scanned {total_scanned} rows -> {new_jobs_processed} NEW unique jobs.")
        print(f"Total Database Size: {total_processed + new_jobs_processed} jobs.")
        print(f"FAISS index updated at {output_index}")
        print(f"Metadata updated at {output_metadata}")
    else:
        print("No valid data found to index.")

if __name__ == "__main__":
    ingest_large_dataset(additional_limit=100000)
