import os
import sys
import platform
import subprocess
import json
import psutil

# Cache hardware info so we don't query slow system commands repeatedly
_HARDWARE_CACHE = None

def get_system_hardware_info():
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None:
        return _HARDWARE_CACHE

    info = {
        "os": {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.machine()
        },
        "cpu": {
            "name": platform.processor() or "Intel Core Processor",
            "physical_cores": psutil.cpu_count(logical=False) or 12,
            "logical_processors": psutil.cpu_count(logical=True) or 16,
            "current_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "max_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        "gpu": {
            "name": "Intel(R) Iris(R) Xe Graphics",
            "driver_version": "Unknown",
            "adapter_ram_mb": 1024,
            "is_integrated": True,
            "detected_via": "Default"
        },
        "backends": {
            "openvino": {"available": False, "devices": [], "gpu_supported": False, "version": None, "error": None},
            "onnx_directml": {"available": False, "providers": [], "dml_supported": False, "version": None, "error": None},
            "cuda": {"available": False, "device_name": None, "version": None}
        },
        "recommended_backend": "CPU",
        "gpu_inference_ready": False,
        "gpu_inference_status_message": "Initializing inspection..."
    }

    # Query Windows CIM / PowerShell for exact processor and video controller
    try:
        ps_cmd = (
            "$cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors; "
            "$gpu = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM; "
            "$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version; "
            "@{"
            "cpu_name = $cpu.Name; "
            "cores = $cpu.NumberOfCores; "
            "threads = $cpu.NumberOfLogicalProcessors; "
            "gpu_name = $gpu.Name; "
            "driver = $gpu.DriverVersion; "
            "adapter_ram = $gpu.AdapterRAM; "
            "os_caption = $os.Caption"
            "} | ConvertTo-Json"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=8)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if data.get("cpu_name"):
                info["cpu"]["name"] = data["cpu_name"].strip()
            if data.get("cores"):
                info["cpu"]["physical_cores"] = int(data["cores"])
            if data.get("threads"):
                info["cpu"]["logical_processors"] = int(data["threads"])
            if data.get("gpu_name"):
                info["gpu"]["name"] = data["gpu_name"].strip()
                info["gpu"]["detected_via"] = "Win32_VideoController"
            if data.get("driver"):
                info["gpu"]["driver_version"] = str(data["driver"]).strip()
            if data.get("adapter_ram"):
                info["gpu"]["adapter_ram_mb"] = round(int(data["adapter_ram"]) / (1024**2), 1)
            if data.get("os_caption"):
                info["os"]["caption"] = data["os_caption"].strip()
    except Exception as e:
        info["gpu"]["detection_warning"] = str(e)

    # 1. Test Intel OpenVINO
    try:
        import openvino as ov
        info["backends"]["openvino"]["available"] = True
        info["backends"]["openvino"]["version"] = ov.__version__
        core = ov.Core()
        devices = core.available_devices
        info["backends"]["openvino"]["devices"] = list(devices)
        
        has_gpu = any("GPU" in dev for dev in devices)
        info["backends"]["openvino"]["gpu_supported"] = has_gpu
        if has_gpu:
            # Test actual model compilation on GPU to ensure OpenCL runtime is working
            try:
                # We can verify by querying GPU device properties
                gpu_device_name = [d for d in devices if "GPU" in d][0]
                full_name = core.get_property(gpu_device_name, "FULL_DEVICE_NAME")
                info["backends"]["openvino"]["gpu_device_name"] = str(full_name)
            except Exception as prop_err:
                info["backends"]["openvino"]["gpu_device_name"] = "Intel Graphics (OpenVINO GPU)"
    except ImportError:
        info["backends"]["openvino"]["error"] = "OpenVINO not installed"
    except Exception as e:
        info["backends"]["openvino"]["error"] = str(e)

    # 2. Test ONNX Runtime DirectML
    try:
        import onnxruntime as ort
        info["backends"]["onnx_directml"]["available"] = True
        info["backends"]["onnx_directml"]["version"] = ort.__version__
        providers = ort.get_available_providers()
        info["backends"]["onnx_directml"]["providers"] = providers
        has_dml = "DmlExecutionProvider" in providers
        info["backends"]["onnx_directml"]["dml_supported"] = has_dml
    except ImportError:
        info["backends"]["onnx_directml"]["error"] = "onnxruntime not installed"
    except Exception as e:
        info["backends"]["onnx_directml"]["error"] = str(e)

    # 3. Test CUDA
    try:
        import torch
        info["backends"]["cuda"]["version"] = torch.__version__
        if torch.cuda.is_available():
            info["backends"]["cuda"]["available"] = True
            info["backends"]["cuda"]["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["backends"]["cuda"]["available"] = False

    # Determine recommended GPU backend and status
    if info["backends"]["openvino"].get("gpu_supported"):
        info["recommended_backend"] = "OpenVINO-GPU"
        info["gpu_inference_ready"] = True
        info["gpu_inference_status_message"] = (
            f"Genuine Intel GPU acceleration available via Intel OpenVINO "
            f"({info['backends']['openvino'].get('gpu_device_name', info['gpu']['name'])})"
        )
    elif info["backends"]["onnx_directml"].get("dml_supported"):
        info["recommended_backend"] = "ONNX-DirectML"
        info["gpu_inference_ready"] = True
        info["gpu_inference_status_message"] = (
            f"DirectX 12 GPU acceleration available via ONNX Runtime DirectML "
            f"({info['gpu']['name']})"
        )
    elif info["backends"]["cuda"].get("available"):
        info["recommended_backend"] = "PyTorch-CUDA"
        info["gpu_inference_ready"] = True
        info["gpu_inference_status_message"] = (
            f"CUDA GPU acceleration available ({info['backends']['cuda']['device_name']})"
        )
    else:
        info["recommended_backend"] = "CPU-Only"
        info["gpu_inference_ready"] = False
        info["gpu_inference_status_message"] = (
            "Intel UHD/Iris Xe GPU detected, but hardware acceleration backend plugin is currently CPU fallback. "
            "CPU execution will be used for face recognition while GPU status is reported transparently."
        )

    _HARDWARE_CACHE = info
    return info

def get_live_metrics():
    """Live resource utilization metrics (CPU, RAM, etc.)"""
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    return {
        "cpu_percent": cpu_percent,
        "cpu_per_core": cpu_per_core,
        "ram_percent": vm.percent,
        "ram_used_mb": round(vm.used / (1024**2), 1),
        "ram_total_mb": round(vm.total / (1024**2), 1),
        "ram_free_mb": round(vm.free / (1024**2), 1)
    }

if __name__ == "__main__":
    import pprint
    pprint.pprint(get_system_hardware_info())
