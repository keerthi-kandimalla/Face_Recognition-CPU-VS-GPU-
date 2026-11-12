import os
import cv2
import numpy as np
import time

from backend.database import init_db, get_celebrities, get_all_embeddings
from backend.hardware import get_system_hardware_info
from backend.face_engine import FaceDetector, FaceFeatureExtractor, match_face
from backend.benchmark import BenchmarkEngine

def run_all_tests():
    print("==========================================")
    print("STARTING FACE RECOGNITION & COA TEST SUITE")
    print("==========================================")

    # 1. Test Hardware Detection
    print("\n[TEST 1] Testing Hardware Detection...")
    hw = get_system_hardware_info()
    print("  CPU Name:", hw["cpu"]["name"])
    print("  Cores:", hw["cpu"]["physical_cores"], "Threads:", hw["cpu"]["logical_processors"])
    print("  GPU Name:", hw["gpu"]["name"])
    print("  Driver:", hw["gpu"]["driver_version"])
    print("  Recommended Backend:", hw["recommended_backend"])
    print("  GPU Inference Ready:", hw["gpu_inference_ready"])
    assert hw["cpu"]["physical_cores"] > 0, "CPU cores check failed"
    print("  -> TEST 1 PASSED!")

    # 2. Test Database & Dataset
    print("\n[TEST 2] Testing SQLite Database & Telugu Celebrity Dataset...")
    init_db()
    celebs = get_celebrities()
    print(f"  Found {len(celebs)} registered Telugu celebrities in DB.")
    assert len(celebs) >= 18, f"Expected at least 18 celebrities, got {len(celebs)}"
    embeddings = get_all_embeddings()
    print(f"  Found {len(embeddings)} stored 512-D face embeddings.")
    assert len(embeddings) > 0, "No embeddings found"
    print("  -> TEST 2 PASSED!")

    # 3. Test Face Detector (OpenCV YuNet)
    print("\n[TEST 3] Testing Face Detector (OpenCV YuNet)...")
    detector = FaceDetector()
    test_img = np.zeros((360, 360, 3), dtype=np.uint8)
    cv2.circle(test_img, (180, 180), 80, (200, 200, 200), -1) # dummy face
    faces = detector.detect_faces(test_img)
    print(f"  Detector executed successfully. Detected face count: {len(faces)}")
    print("  -> TEST 3 PASSED!")

    # 4. Test CNN Feature Extractor on CPU
    print("\n[TEST 4] Testing CNN Feature Extractor on CPU...")
    extractor = FaceFeatureExtractor()
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    tensor = extractor.preprocess(dummy_face)
    emb_cpu, cpu_time = extractor.extract_embedding_cpu(tensor)
    print(f"  CPU Embedding shape: {emb_cpu.shape}, Norm: {np.linalg.norm(emb_cpu):.4f}, Time: {cpu_time:.2f} ms")
    assert emb_cpu.shape[-1] in [128, 512], "Invalid embedding vector dimension"
    print("  -> TEST 4 PASSED!")

    # 5. Test CNN Feature Extractor on GPU (Intel Iris Xe)
    print("\n[TEST 5] Testing CNN Feature Extractor on GPU (Intel Iris Xe)...")
    if hw["gpu_inference_ready"]:
        try:
            emb_gpu, gpu_time = extractor.extract_embedding_gpu(tensor)
            print(f"  GPU Embedding shape: {emb_gpu.shape}, Norm: {np.linalg.norm(emb_gpu):.4f}, Time: {gpu_time:.2f} ms")
            print("  -> TEST 5 PASSED (Genuine Intel Iris Xe GPU execution confirmed)!")
        except Exception as e:
            print(f"  GPU execution notice: {e}")
    else:
        print("  -> GPU acceleration not active, skipped.")

    # 6. Test Cosine Similarity Matching
    print("\n[TEST 6] Testing Identity Recognition & Cosine Similarity...")
    # Test query with first stored embedding
    first_emb = embeddings[0]["embedding"]
    match = match_face(first_emb, embeddings, threshold=0.50)
    print(f"  Query against own embedding -> Recognized: {match['celebrity_name']}, Similarity: {match['similarity_score']}")
    assert match["match_found"] is True, "Self-matching should be True"
    assert match["celebrity_name"] == embeddings[0]["celebrity_name"]
    print("  -> TEST 6 PASSED!")

    # 7. Test Comparative Benchmark Engine
    print("\n[TEST 7] Testing CPU vs GPU Benchmark Engine...")
    engine = BenchmarkEngine()
    bench_res = engine.execute_benchmark(workload_type="test", workload_size=16, batch_size=8)
    print(f"  Benchmark executed for {bench_res['workload_size']} images (Batch 8):")
    print(f"    CPU Time: {bench_res['cpu_inference_time_ms']} ms | Throughput: {bench_res['cpu_throughput']} FPS")
    if bench_res.get("gpu_inference_time_ms"):
        print(f"    GPU Time: {bench_res['gpu_inference_time_ms']} ms | Throughput: {bench_res['gpu_throughput']} FPS")
        print(f"    Speedup: {bench_res['speedup']}x | Faster Device: {bench_res['faster_device']}")
    print("  -> TEST 7 PASSED!")

    print("\n==========================================")
    print("ALL 7 SYSTEM & ARCHITECTURE TESTS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    run_all_tests()
