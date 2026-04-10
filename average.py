import sqlite3
import numpy as np
from collections import defaultdict
import os
import time

DB_PATH = "data/posts.db"
OUTPUT_DIR = "data"
SPLIT_DATE = "2022-02-15"
CHUNK_SIZE = 100000

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM posts WHERE embedded = 1").fetchone()[0]
    print(f"Total embedded posts: {total:,}")
    print("Reading embeddings in chunks and accumulating by hour...\n")

    # store running sum and count per hour to avoid holding all vectors in memory
    hour_sums = defaultdict(lambda: np.zeros(1536, dtype=np.float64))
    hour_counts = defaultdict(int)

    processed = 0
    offset = 0
    start_time = time.time()

    while True:
        rows = conn.execute("""
            SELECT substr(created_at, 1, 13) || ':00:00', embedding
            FROM posts
            WHERE embedded = 1
            LIMIT ? OFFSET ?
        """, (CHUNK_SIZE, offset)).fetchall()

        if not rows:
            break

        for hour, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
            hour_sums[hour] += vector
            hour_counts[hour] += 1

        processed += len(rows)
        offset += CHUNK_SIZE

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = total - processed
        eta_mins = (remaining / rate / 60) if rate > 0 else 0
        pct = processed / total * 100

        print(
            f"\r  Progress: {pct:.1f}% | "
            f"Posts: {processed:,}/{total:,} | "
            f"Hours found: {len(hour_sums):,} | "
            f"Rate: {rate:.0f}/s | "
            f"ETA: {eta_mins:.1f}m",
            end="", flush=True
        )

    conn.close()
    print(f"\n\nComputing averages for {len(hour_sums):,} hourly windows...")

    train_vectors = []
    train_timestamps = []
    test_vectors = []
    test_timestamps = []

    for i, hour in enumerate(sorted(hour_sums.keys())):
        avg_vector = (hour_sums[hour] / hour_counts[hour]).astype(np.float32)

        if hour < SPLIT_DATE:
            train_vectors.append(avg_vector)
            train_timestamps.append(hour)
        else:
            test_vectors.append(avg_vector)
            test_timestamps.append(hour)

        if i % 500 == 0:
            print(f"  Averaged {i:,} / {len(hour_sums):,} hours...")

    print("\nSaving files...")
    np.save(f"{OUTPUT_DIR}/train_embeddings.npy", np.array(train_vectors))
    np.save(f"{OUTPUT_DIR}/train_timestamps.npy", np.array(train_timestamps))
    np.save(f"{OUTPUT_DIR}/test_embeddings.npy", np.array(test_vectors))
    np.save(f"{OUTPUT_DIR}/test_timestamps.npy", np.array(test_timestamps))

    total_hours = len(train_vectors) + len(test_vectors)
    size_mb = total_hours * 1536 * 4 / 1e6

    print(f"\nDone.")
    print(f"Train: {len(train_vectors):,} hours → data/train_embeddings.npy")
    print(f"Test:  {len(test_vectors):,} hours → data/test_embeddings.npy")
    print(f"Total size: ~{size_mb:.1f} MB")

if __name__ == "__main__":
    run()
