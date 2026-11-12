import sqlite3
import os
import json
import numpy as np
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "face_recognition.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Celebrities table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS celebrities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        notes TEXT,
        image_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # Celebrity Images table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS celebrity_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        celebrity_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        face_detected INTEGER DEFAULT 1,
        bounding_box TEXT,
        confidence REAL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (celebrity_id) REFERENCES celebrities(id) ON DELETE CASCADE
    )
    """)

    # Face Embeddings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        celebrity_id INTEGER NOT NULL,
        image_id INTEGER,
        embedding BLOB NOT NULL,
        dimension INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (celebrity_id) REFERENCES celebrities(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES celebrity_images(id) ON DELETE CASCADE
    )
    """)

    # Benchmark experiments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS benchmark_experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        workload_type TEXT NOT NULL,
        workload_size INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        cpu_device_name TEXT NOT NULL,
        gpu_device_name TEXT NOT NULL,
        gpu_backend_used TEXT NOT NULL,
        cpu_inference_time_ms REAL NOT NULL,
        gpu_inference_time_ms REAL,
        cpu_throughput REAL NOT NULL,
        gpu_throughput REAL,
        speedup REAL,
        faster_device TEXT,
        cpu_memory_mb REAL,
        gpu_memory_mb REAL,
        cpu_utilization_pct REAL,
        gpu_utilization_pct REAL,
        status TEXT NOT NULL,
        details_json TEXT
    )
    """)

    conn.commit()
    conn.close()

def register_celebrity(name: str, notes: str = "") -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO celebrities (name, notes, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (name.strip(), notes.strip(), now, now)
    )
    celeb_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return celeb_id

def get_celebrities():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*, COUNT(e.id) as embedding_count 
        FROM celebrities c
        LEFT JOIN face_embeddings e ON c.id = e.celebrity_id
        GROUP BY c.id
        ORDER BY c.name ASC
    """)
    rows = cursor.fetchall()
    celebrities = [dict(row) for row in rows]
    conn.close()
    return celebrities

def get_celebrity(celebrity_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM celebrities WHERE id = ?", (celebrity_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_celebrity_by_name(name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM celebrities WHERE LOWER(name) = LOWER(?)", (name.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_celebrity(celebrity_id: int, name: str, notes: str = ""):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "UPDATE celebrities SET name = ?, notes = ?, updated_at = ? WHERE id = ?",
        (name.strip(), notes.strip(), now, celebrity_id)
    )
    conn.commit()
    conn.close()

def delete_celebrity(celebrity_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM face_embeddings WHERE celebrity_id = ?", (celebrity_id,))
    cursor.execute("DELETE FROM celebrity_images WHERE celebrity_id = ?", (celebrity_id,))
    cursor.execute("DELETE FROM celebrities WHERE id = ?", (celebrity_id,))
    conn.commit()
    conn.close()

def add_celebrity_image(celebrity_id: int, file_path: str, face_detected: int = 1, bbox=None, confidence: float = 1.0) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    bbox_str = json.dumps(bbox) if bbox is not None else None
    cursor.execute(
        "INSERT INTO celebrity_images (celebrity_id, file_path, face_detected, bounding_box, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (celebrity_id, file_path, face_detected, bbox_str, confidence, now)
    )
    image_id = cursor.lastrowid
    cursor.execute("UPDATE celebrities SET image_count = image_count + 1, updated_at = ? WHERE id = ?", (now, celebrity_id))
    conn.commit()
    conn.close()
    return image_id

def save_embedding(celebrity_id: int, image_id: int, embedding_vector: np.ndarray) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    norm_emb = embedding_vector.astype(np.float32)
    # Ensure L2 normalized
    norm = np.linalg.norm(norm_emb)
    if norm > 1e-6:
        norm_emb = norm_emb / norm
    emb_blob = norm_emb.tobytes()
    dim = len(norm_emb)

    cursor.execute(
        "INSERT INTO face_embeddings (celebrity_id, image_id, embedding, dimension, created_at) VALUES (?, ?, ?, ?, ?)",
        (celebrity_id, image_id, emb_blob, dim, now)
    )
    emb_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return emb_id

def get_all_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.id, e.celebrity_id, e.dimension, e.embedding, c.name as celebrity_name, i.file_path
        FROM face_embeddings e
        JOIN celebrities c ON e.celebrity_id = c.id
        LEFT JOIN celebrity_images i ON e.image_id = i.id
    """)
    rows = cursor.fetchall()
    results = []
    for row in rows:
        emb_arr = np.frombuffer(row["embedding"], dtype=np.float32)
        results.append({
            "id": row["id"],
            "celebrity_id": row["celebrity_id"],
            "celebrity_name": row["celebrity_name"],
            "dimension": row["dimension"],
            "embedding": emb_arr,
            "image_path": row["file_path"]
        })
    conn.close()
    return results

def save_benchmark_result(data: dict) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO benchmark_experiments (
            timestamp, workload_type, workload_size, batch_size, model_name,
            cpu_device_name, gpu_device_name, gpu_backend_used,
            cpu_inference_time_ms, gpu_inference_time_ms,
            cpu_throughput, gpu_throughput, speedup, faster_device,
            cpu_memory_mb, gpu_memory_mb, cpu_utilization_pct, gpu_utilization_pct,
            status, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        data.get("workload_type", "single"),
        data.get("workload_size", 1),
        data.get("batch_size", 1),
        data.get("model_name", "ArcFace-ResNet"),
        data.get("cpu_device_name", "Unknown CPU"),
        data.get("gpu_device_name", "Unknown GPU"),
        data.get("gpu_backend_used", "None"),
        data.get("cpu_inference_time_ms", 0.0),
        data.get("gpu_inference_time_ms"),
        data.get("cpu_throughput", 0.0),
        data.get("gpu_throughput"),
        data.get("speedup"),
        data.get("faster_device", "CPU"),
        data.get("cpu_memory_mb", 0.0),
        data.get("gpu_memory_mb"),
        data.get("cpu_utilization_pct", 0.0),
        data.get("gpu_utilization_pct"),
        data.get("status", "SUCCESS"),
        json.dumps(data.get("details", {}))
    ))
    res_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return res_id

def get_benchmark_history(limit: int = 50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM benchmark_experiments
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

def delete_benchmark_result(result_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM benchmark_experiments WHERE id = ?", (result_id,))
    conn.commit()
    conn.close()
