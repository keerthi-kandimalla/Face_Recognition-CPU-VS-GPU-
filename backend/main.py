import os
import io
import csv
import base64
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List

from backend.database import (
    init_db, get_celebrities, get_celebrity, register_celebrity,
    update_celebrity, delete_celebrity, get_all_embeddings,
    get_benchmark_history, delete_benchmark_result, save_embedding, add_celebrity_image
)
from backend.hardware import get_system_hardware_info, get_live_metrics
from backend.face_engine import FaceDetector, FaceFeatureExtractor, match_face
from backend.benchmark import BenchmarkEngine
from backend.dataset_service import (
    get_dataset_statistics, process_and_register_image, handle_zip_upload
)

# Initialize database
init_db()

app = FastAPI(
    title="CPU vs GPU Acceleration of CNN Face Recognition",
    description="Computer Architecture Performance Analysis Platform",
    version="1.0.0"
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
detector = FaceDetector()
extractor = FaceFeatureExtractor()
benchmark_engine = BenchmarkEngine(
    cpu_compiled=extractor.cpu_compiled,
    gpu_compiled=extractor.gpu_compiled
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Mount dataset images for preview
if os.path.exists(DATASET_DIR):
    app.mount("/dataset-files", StaticFiles(directory=DATASET_DIR), name="dataset-files")

# 1. HARDWARE ENDPOINTS
@app.get("/api/hardware")
def api_get_hardware():
    """Returns detected CPU, GPU, and acceleration backend capabilities."""
    info = get_system_hardware_info()
    return info

@app.get("/api/metrics/live")
def api_get_live_metrics():
    """Returns real-time CPU and Memory utilization."""
    return get_live_metrics()

# 2. CELEBRITY MANAGEMENT ENDPOINTS
@app.get("/api/celebrities")
def api_get_celebrities():
    """List all registered celebrities and image counts."""
    return get_celebrities()

@app.post("/api/celebrities")
def api_register_celebrity(name: str = Form(...), notes: str = Form("")):
    """Register a new celebrity."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="Celebrity name cannot be empty.")
    try:
        celeb_id = register_celebrity(name.strip(), notes.strip())
        return {"success": True, "id": celeb_id, "name": name.strip()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

@app.put("/api/celebrities/{celeb_id}")
def api_update_celebrity(celeb_id: int, name: str = Form(...), notes: str = Form("")):
    """Update celebrity information."""
    update_celebrity(celeb_id, name, notes)
    return {"success": True, "message": "Updated successfully."}

@app.delete("/api/celebrities/{celeb_id}")
def api_delete_celebrity(celeb_id: int):
    """Delete a celebrity and all their stored embeddings/images."""
    delete_celebrity(celeb_id)
    return {"success": True, "message": "Deleted successfully."}

@app.post("/api/celebrities/{celeb_id}/images")
async def api_upload_celebrity_image(celeb_id: int, file: UploadFile = File(...)):
    """Upload a new image for an existing celebrity."""
    contents = await file.read()
    res = process_and_register_image(celeb_id, contents, file.filename)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res

# 3. DATASET MANAGEMENT ENDPOINTS
@app.get("/api/dataset/stats")
def api_get_dataset_stats():
    """Returns dataset summary statistics and folder counts."""
    return get_dataset_statistics()

@app.post("/api/dataset/upload-zip")
async def api_upload_zip(file: UploadFile = File(...)):
    """Upload a zip file containing celebrity folders."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted.")
    
    tmp_path = os.path.join(DATASET_DIR, f"upload_{int(time.time())}.zip")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
        
    try:
        result = handle_zip_upload(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return result

# 4. FACE RECOGNITION (EXPERIMENT A: SINGLE IMAGE)
@app.post("/api/recognize")
async def api_recognize_face(file: UploadFile = File(...), threshold: float = Form(0.38)):
    """
    Executes real face recognition and measures CPU vs GPU inference latency.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    h, w = img_bgr.shape[:2]
    t0_detection = time.perf_counter()
    faces = detector.detect_faces(img_bgr)
    t1_detection = time.perf_counter()
    detection_time_ms = round((t1_detection - t0_detection) * 1000.0, 2)

    if not faces:
        return {
            "face_detected": False,
            "message": "No face detected in the uploaded image.",
            "detection_time_ms": detection_time_ms
        }

    primary_face = faces[0]
    bbox = primary_face["box"]

    # Crop and align face
    t0_preprocess = time.perf_counter()
    cropped_face = detector.crop_and_align_face(img_bgr, primary_face)
    tensor = extractor.preprocess(cropped_face)
    t1_preprocess = time.perf_counter()
    preprocess_time_ms = round((t1_preprocess - t0_preprocess) * 1000.0, 2)

    # 1. CPU Inference
    cpu_embedding, cpu_inf_time_ms = extractor.extract_embedding_cpu(tensor)

    # 2. GPU Inference (if available)
    gpu_embedding = None
    gpu_inf_time_ms = None
    gpu_error = None
    t0_gpu = time.perf_counter()
    try:
        gpu_embedding, gpu_inf_time_ms = extractor.extract_embedding_gpu(tensor)
    except Exception as e:
        gpu_error = str(e)
    t1_gpu = time.perf_counter()
    gpu_total_time_ms = round((t1_gpu - t0_gpu) * 1000.0, 2) if gpu_inf_time_ms is not None else None

    # Retrieve all stored embeddings from DB
    stored_embeddings = get_all_embeddings()
    match_result = match_face(cpu_embedding, stored_embeddings, threshold=threshold)

    # Determine faster device & speedup
    faster_device = "CPU"
    speedup = None
    if gpu_inf_time_ms is not None and gpu_inf_time_ms > 0:
        speedup = round(cpu_inf_time_ms / gpu_inf_time_ms, 2)
        if gpu_inf_time_ms < cpu_inf_time_ms:
            faster_device = "GPU"
        else:
            faster_device = "CPU"

    # Encode cropped face to base64 for frontend preview
    _, buffer = cv2.imencode(".jpg", cropped_face)
    cropped_b64 = base64.b64encode(buffer).decode("utf-8")

    # Draw bounding box on original image for preview
    annotated_img = img_bgr.copy()
    x, y, bw, bh = bbox
    cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (0, 220, 100), 2)
    label = f"{match_result['celebrity_name']} ({int(match_result['similarity_score']*100)}%)"
    cv2.putText(annotated_img, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)
    _, ann_buffer = cv2.imencode(".jpg", annotated_img)
    annotated_b64 = base64.b64encode(ann_buffer).decode("utf-8")

    hw = get_system_hardware_info()

    return {
        "face_detected": True,
        "face_count": len(faces),
        "bounding_box": bbox,
        "confidence": round(primary_face["confidence"], 3),
        "cropped_face_b64": f"data:image/jpeg;base64,{cropped_b64}",
        "annotated_image_b64": f"data:image/jpeg;base64,{annotated_b64}",
        "recognition": match_result,
        "performance": {
            "detection_time_ms": detection_time_ms,
            "preprocess_time_ms": preprocess_time_ms,
            "cpu_inference_time_ms": round(cpu_inf_time_ms, 2),
            "gpu_inference_time_ms": round(gpu_inf_time_ms, 2) if gpu_inf_time_ms else None,
            "faster_device": faster_device,
            "speedup": speedup,
            "cpu_device": hw["cpu"]["name"],
            "gpu_device": hw["gpu"]["name"],
            "gpu_backend": hw["recommended_backend"],
            "gpu_available": gpu_inf_time_ms is not None,
            "gpu_error": gpu_error,
            "architecture_note": (
                "For single-image inference (batch size 1), CPU frequently displays competitive or faster latency "
                "because GPU involves kernel launch latency, driver synchronization, and host-to-device memory transfer overhead. "
                "Run Experiment B (Batch Benchmark) to observe GPU parallel compute scaling."
            )
        }
    }

# 5. BENCHMARKING (EXPERIMENT B: LARGE-SCALE WORKLOAD)
@app.post("/api/benchmark/run")
def api_run_benchmark(
    workload_size: int = Form(50),
    batch_size: int = Form(8),
    repetitions: int = Form(1)
):
    """
    Executes large-scale CNN benchmark on CPU and GPU across workload and batch sizes.
    """
    if workload_size < 1 or workload_size > 5000:
        raise HTTPException(status_code=400, detail="Workload size must be between 1 and 5000.")
    if batch_size not in [1, 8, 16, 32, 64]:
        raise HTTPException(status_code=400, detail="Batch size must be one of [1, 8, 16, 32, 64].")

    result = benchmark_engine.execute_benchmark(
        workload_type="custom",
        workload_size=workload_size,
        batch_size=batch_size
    )
    return result

@app.get("/api/benchmark/history")
def api_get_benchmark_history(limit: int = 50):
    """Returns historical benchmark experiments."""
    return get_benchmark_history(limit=limit)

@app.delete("/api/benchmark/history/{result_id}")
def api_delete_benchmark_history(result_id: int):
    """Delete a benchmark history record."""
    delete_benchmark_result(result_id)
    return {"success": True}

@app.get("/api/benchmark/export-csv")
def api_export_benchmark_csv():
    """Exports all benchmark experiments to CSV."""
    records = get_benchmark_history(limit=500)
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        "ID", "Timestamp", "Workload Type", "Workload Size", "Batch Size",
        "Model Name", "CPU Device", "GPU Device", "GPU Backend",
        "CPU Inference Time (ms)", "GPU Inference Time (ms)",
        "CPU Throughput (FPS)", "GPU Throughput (FPS)",
        "Speedup", "Faster Device", "CPU Memory (MB)", "GPU Memory (MB)",
        "Status"
    ])
    
    for r in records:
        writer.writerow([
            r.get("id"), r.get("timestamp"), r.get("workload_type"), r.get("workload_size"),
            r.get("batch_size"), r.get("model_name"), r.get("cpu_device_name"),
            r.get("gpu_device_name"), r.get("gpu_backend_used"),
            r.get("cpu_inference_time_ms"), r.get("gpu_inference_time_ms"),
            r.get("cpu_throughput"), r.get("gpu_throughput"),
            r.get("speedup"), r.get("faster_device"),
            r.get("cpu_memory_mb"), r.get("gpu_memory_mb"),
            r.get("status")
        ])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=benchmark_results_{int(time.time())}.csv"}
    )

# Mount frontend static directory if exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
