import uvicorn
import os
import sys

if __name__ == "__main__":
    print("================================================================================")
    print("  CPU vs GPU Acceleration of CNN-Based Face Recognition Web Application")
    print("  A Computer Architecture Performance Analysis Platform")
    print("================================================================================")
    print("  Server launching on: http://localhost:8000")
    print("  Press Ctrl+C to terminate.")
    print("================================================================================")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
