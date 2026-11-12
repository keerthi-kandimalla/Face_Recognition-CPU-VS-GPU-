import os
import sqlite3
import cv2
import numpy as np
from backend.face_engine import FaceDetector, FaceFeatureExtractor

def reembed():
    detector = FaceDetector()
    extractor = FaceFeatureExtractor()

    conn = sqlite3.connect(os.path.join("database", "face_recognition.db"))
    c = conn.cursor()
    c.execute("DELETE FROM face_embeddings")
    c.execute("DELETE FROM celebrity_images")
    print("Cleaned old database records.")

    dataset_dir = "dataset"
    total_imgs = 0

    for folder in sorted(os.listdir(dataset_dir)):
        f_dir = os.path.join(dataset_dir, folder)
        if not os.path.isdir(f_dir) or folder.startswith("_"):
            continue

        celeb_name = folder.replace("_", " ").strip()
        c.execute("SELECT id FROM celebrities WHERE LOWER(name) = LOWER(?)", (celeb_name,))
        row = c.fetchone()
        if not row:
            c.execute("INSERT INTO celebrities (name, notes, created_at, updated_at) VALUES (?, 'Registered celebrity', datetime('now'), datetime('now'))", (celeb_name,))
            celeb_id = c.lastrowid
        else:
            celeb_id = row[0]

        files = [f for f in os.listdir(f_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        for f in files:
            img_path = os.path.join(f_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = detector.detect_faces(img)
            face_info = faces[0] if faces else {"box": [0, 0, img.shape[1], img.shape[0]], "confidence": 0.9, "raw_face": None}
            bbox = face_info["box"]
            conf = face_info.get("confidence", 0.9)

            c.execute(
                "INSERT INTO celebrity_images (celebrity_id, file_path, face_detected, bounding_box, confidence, created_at) VALUES (?, ?, 1, ?, ?, datetime('now'))",
                (celeb_id, img_path, str(bbox), conf)
            )
            img_id = c.lastrowid

            cropped = detector.crop_and_align_face(img, face_info)
            tensor = extractor.preprocess(cropped)
            emb, _ = extractor.extract_embedding_cpu(tensor)

            blob = emb.astype(np.float32).tobytes()
            c.execute(
                "INSERT INTO face_embeddings (celebrity_id, image_id, embedding, dimension, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                (celeb_id, img_id, blob, len(emb))
            )
            total_imgs += 1

    conn.commit()
    c.execute("SELECT dimension, count(*) FROM face_embeddings GROUP BY dimension")
    print("New DB embeddings summary:", c.fetchall())
    conn.close()
    print(f"Successfully re-embedded {total_imgs} images across all celebrities!")

if __name__ == "__main__":
    reembed()
