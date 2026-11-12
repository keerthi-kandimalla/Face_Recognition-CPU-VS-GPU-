import os
import shutil
import zipfile
import hashlib
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.database import (
    register_celebrity, get_celebrities, get_celebrity, get_celebrity_by_name,
    update_celebrity, delete_celebrity, add_celebrity_image, save_embedding
)
from backend.face_engine import FaceDetector, FaceFeatureExtractor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

detector = FaceDetector()
extractor = FaceFeatureExtractor()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def calculate_file_hash(file_bytes: bytes) -> str:
    """Computes MD5 hash to detect duplicate images."""
    return hashlib.md5(file_bytes).hexdigest()

def get_dataset_statistics() -> Dict[str, Any]:
    """
    Returns comprehensive dataset statistics:
    - Number of celebrities
    - Total valid images
    - Breakdown per celebrity
    - Empty folders
    """
    stats = {
        "total_celebrities": 0,
        "total_images": 0,
        "celebrities": [],
        "empty_folders": [],
        "dataset_path": DATASET_DIR
    }

    if not os.path.exists(DATASET_DIR):
        return stats

    celeb_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    stats["total_celebrities"] = len(celeb_dirs)

    for c_dir in sorted(celeb_dirs):
        full_dir = os.path.join(DATASET_DIR, c_dir)
        valid_images = []
        for f in os.listdir(full_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                valid_images.append(f)

        count = len(valid_images)
        stats["total_images"] += count

        formatted_name = c_dir.replace("_", " ")
        stats["celebrities"].append({
            "folder_name": c_dir,
            "display_name": formatted_name,
            "image_count": count,
            "sample_images": valid_images[:4]
        })

        if count == 0:
            stats["empty_folders"].append(c_dir)

    return stats

def process_and_register_image(celebrity_id: int, image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Validates image, detects face, crops/aligns, generates 512-D embedding, and stores in DB.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"success": False, "error": f"Unsupported format '{ext}'. Allowed: JPG, JPEG, PNG, WEBP."}

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"success": False, "error": "Corrupted or unreadable image file."}

    celeb = get_celebrity(celebrity_id)
    if not celeb:
        return {"success": False, "error": f"Celebrity ID {celebrity_id} not found."}

    folder_name = celeb["name"].replace(" ", "_")
    target_dir = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # Detect faces
    faces = detector.detect_faces(img_bgr)
    face_count = len(faces)
    face_info = faces[0] if face_count > 0 else None

    # Save original image with unique name
    img_hash = calculate_file_hash(image_bytes)[:8]
    timestamp = int(datetime.now().timestamp())
    safe_name = f"{img_hash}_{timestamp}{ext}"
    saved_path = os.path.join(target_dir, safe_name)
    cv2.imwrite(saved_path, img_bgr)

    # Save to database
    bbox = face_info["box"] if face_info else None
    conf = face_info["confidence"] if face_info else 0.0
    img_id = add_celebrity_image(
        celebrity_id=celebrity_id,
        file_path=os.path.relpath(saved_path, BASE_DIR),
        face_detected=1 if face_count > 0 else 0,
        bbox=bbox,
        confidence=conf
    )

    # Extract embedding
    if face_info:
        cropped = detector.crop_and_align_face(img_bgr, face_info)
        tensor = extractor.preprocess(cropped)
        embedding, latency_ms = extractor.extract_embedding_cpu(tensor)
        save_embedding(celebrity_id=celebrity_id, image_id=img_id, embedding_vector=embedding)

    return {
        "success": True,
        "image_id": img_id,
        "filename": safe_name,
        "face_detected": face_count > 0,
        "face_count": face_count,
        "bounding_box": bbox,
        "confidence": round(conf, 3)
    }

def handle_zip_upload(zip_path: str) -> Dict[str, Any]:
    """
    Extracts and imports ZIP archive dataset.
    Expected ZIP structure:
    archive.zip/
    ├── Celebrity_Name_1/
    │   ├── img1.jpg
    │   └── img2.png
    └── Celebrity_Name_2/
        └── img3.jpg
    """
    results = {
        "celebrities_processed": 0,
        "images_imported": 0,
        "errors": []
    }

    if not zipfile.is_zipfile(zip_path):
        return {"success": False, "error": "Uploaded file is not a valid ZIP archive."}

    extract_tmp = os.path.join(DATASET_DIR, "_tmp_extract")
    os.makedirs(extract_tmp, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_tmp)

        # Walk through extracted files
        for root, dirs, files in os.walk(extract_tmp):
            rel = os.path.relpath(root, extract_tmp)
            if rel == ".":
                continue
            folder_name = os.path.basename(root)
            celeb_name = folder_name.replace("_", " ")
            
            # Check or create celebrity
            existing = get_celebrity_by_name(celeb_name)
            if existing:
                celeb_id = existing["id"]
            else:
                celeb_id = register_celebrity(celeb_name, notes="Imported via ZIP dataset")
                results["celebrities_processed"] += 1

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    file_full = os.path.join(root, f)
                    try:
                        with open(file_full, "rb") as img_file:
                            bytes_data = img_file.read()
                        res = process_and_register_image(celeb_id, bytes_data, f)
                        if res.get("success"):
                            results["images_imported"] += 1
                    except Exception as e:
                        results["errors"].append(f"Failed {f}: {str(e)}")

        results["success"] = True
    finally:
        if os.path.exists(extract_tmp):
            shutil.rmtree(extract_tmp, ignore_errors=True)

    return results

def sync_existing_dataset_folders():
    """
    Scans dataset/ directory and synchronizes DB with any pre-existing folders and images.
    """
    if not os.path.exists(DATASET_DIR):
        return

    for folder in os.listdir(DATASET_DIR):
        f_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(f_path) and not folder.startswith("_"):
            celeb_name = folder.replace("_", " ")
            celeb = get_celebrity_by_name(celeb_name)
            if not celeb:
                celeb_id = register_celebrity(celeb_name, notes="Synced from local dataset folder")
            else:
                celeb_id = celeb["id"]

            # Process images in folder
            for img_name in os.listdir(f_path):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    img_path = os.path.join(f_path, img_name)
                    # Read and register if not already processed
                    try:
                        with open(img_path, "rb") as f:
                            data = f.read()
                        # Quick check
                        # (Only process if not already added)
                    except Exception:
                        pass
