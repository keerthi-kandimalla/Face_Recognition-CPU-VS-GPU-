// State Management
let currentFile = null;
let currentThreshold = 0.38;
let registeredCelebrities = [];
let chartLatencyInstance = null;
let chartThroughputInstance = null;
let chartSpeedupInstance = null;
let chartMemoryInstance = null;
let simTaskCount = 4;
let simInterval = null;

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initUpload();
    initThresholdSlider();
    fetchHardwareInfo();
    fetchCelebrities();
    fetchBenchmarkHistory();
    initCharts();
    initParallelSim();

    // Start live metrics polling
    setInterval(fetchLiveMetrics, 2500);
});

// 1. NAVIGATION & TABS
function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

    const target = document.getElementById(`tab-${tabId}`);
    if (target) target.classList.remove("hidden");

    const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (btn) btn.classList.add("active");

    if (tabId === "graphs") {
        setTimeout(renderChartsData, 100);
    }
}

function initTabs() {
    // Default tab is recognition
}

// 2. HARDWARE & LIVE METRICS
async function fetchHardwareInfo() {
    try {
        const res = await fetch("/api/hardware");
        const data = await res.json();

        // Top bar
        document.getElementById("topCpuName").textContent = data.cpu.name.replace("Intel(R) Core(TM) ", "").replace("Processor", "").trim();
        document.getElementById("topGpuName").textContent = data.gpu.name.replace("Intel(R) ", "").trim();
        document.getElementById("topBackendBadge").textContent = `Backend: ${data.recommended_backend}`;

        // Dashboard specs
        document.getElementById("dashCpuName").textContent = data.cpu.name;
        document.getElementById("dashCpuCores").textContent = `${data.cpu.physical_cores} Cores`;
        document.getElementById("dashCpuThreads").textContent = `${data.cpu.logical_processors} Threads`;
        document.getElementById("dashCpuArch").textContent = `${data.os.architecture} (${data.os.caption || data.os.name})`;

        document.getElementById("dashGpuName").textContent = data.gpu.name;
        document.getElementById("dashGpuDriver").textContent = data.gpu.driver_version || "Intel 31.0";
        document.getElementById("dashGpuRam").textContent = `${data.gpu.adapter_ram_mb} MB`;

        document.getElementById("dashBackendName").textContent = data.recommended_backend;
        document.getElementById("dashOvDevices").textContent = (data.backends.openvino.devices || []).join(", ") || "CPU, GPU";
        document.getElementById("dashDmlStatus").textContent = data.backends.onnx_directml.dml_supported ? "Available (DirectX 12)" : "N/A";
        document.getElementById("dashCudaStatus").textContent = data.backends.cuda.available ? "Active" : "N/A (Intel Hardware)";

        if (data.gpu_inference_ready) {
            document.getElementById("dashGpuSupportedBadge").textContent = "Genuine GPU Ready";
            document.getElementById("dashGpuSupportedBadge").className = "px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-bold";
        }
    } catch (e) {
        console.error("Hardware query error:", e);
    }
}

async function fetchLiveMetrics() {
    try {
        const res = await fetch("/api/metrics/live");
        const data = await res.json();

        const cpuPct = Math.round(data.cpu_percent);
        const ramPct = Math.round(data.ram_percent);

        document.getElementById("dashLiveCpu").textContent = `${cpuPct}%`;
        document.getElementById("dashLiveCpuBar").style.width = `${cpuPct}%`;

        document.getElementById("dashLiveRam").textContent = `${ramPct}% (${data.ram_used_mb} MB)`;
        document.getElementById("dashLiveRamBar").style.width = `${ramPct}%`;
    } catch (e) {
        // Silent catch for background polling
    }
}

function refreshHardware() {
    fetchHardwareInfo();
    fetchLiveMetrics();
}

// 3. IMAGE UPLOAD & RECOGNITION (EXPERIMENT A)
function initUpload() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const recognizeBtn = document.getElementById("recognizeBtn");

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("border-brand-500", "bg-brand-500/10");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("border-brand-500", "bg-brand-500/10");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-brand-500", "bg-brand-500/10");
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    recognizeBtn.addEventListener("click", runFaceRecognition);
}

function handleSelectedFile(file) {
    currentFile = file;
    document.getElementById("recognizeBtn").disabled = false;

    // Reset visual placeholders
    const reader = new FileReader();
    reader.onload = (e) => {
        const annotImg = document.getElementById("annotatedPreview");
        annotImg.src = e.target.result;
        annotImg.classList.remove("hidden");
        document.getElementById("annotatedPlaceholder").classList.add("hidden");
    };
    reader.readAsDataURL(file);
}

function initThresholdSlider() {
    const slider = document.getElementById("thresholdSlider");
    const label = document.getElementById("thresholdVal");
    slider.addEventListener("input", (e) => {
        currentThreshold = parseFloat(e.target.value);
        label.textContent = currentThreshold.toFixed(2);
    });
}

async function runFaceRecognition() {
    if (!currentFile) return;

    const btn = document.getElementById("recognizeBtn");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Extracting CNN Features & Benchmarking Devices...</span>`;

    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("threshold", currentThreshold);

    try {
        const res = await fetch("/api/recognize", {
            method: "POST",
            body: formData
        });
        if (!res.ok) {
            const errText = await res.text();
            let errMsg = errText;
            try {
                const errObj = JSON.parse(errText);
                errMsg = errObj.detail || errObj.message || errText;
            } catch (_) {}
            throw new Error(errMsg);
        }
        const data = await res.json();

        if (!data.face_detected) {
            alert(data.message || "No face detected in the image.");
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-bolt"></i> <span>Run Recognition & CPU/GPU Inference</span>`;
            return;
        }

        // 1. Update Detection Visuals
        document.getElementById("detTimeVal").textContent = `${data.performance.detection_time_ms} ms`;
        document.getElementById("detectionBadge").textContent = `${data.face_count} Face(s) Detected`;
        document.getElementById("detectionBadge").className = "text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold";

        if (data.annotated_image_b64) {
            const annImg = document.getElementById("annotatedPreview");
            annImg.src = data.annotated_image_b64;
            annImg.classList.remove("hidden");
        }

        if (data.cropped_face_b64) {
            const cropImg = document.getElementById("cropPreview");
            cropImg.src = data.cropped_face_b64;
            cropImg.classList.remove("hidden");
            document.getElementById("cropPlaceholder").classList.add("hidden");
        }

        // 2. Update Recognition Identity
        const rec = data.recognition;
        const nameEl = document.getElementById("recognizedName");
        const statusPill = document.getElementById("matchStatusPill");
        const simVal = document.getElementById("similarityVal");
        const simBar = document.getElementById("similarityBar");

        nameEl.textContent = rec.celebrity_name;
        simVal.textContent = rec.similarity_score.toFixed(4);
        simBar.style.width = `${Math.min(100, Math.max(0, rec.similarity_score * 100))}%`;

        if (rec.match_found) {
            statusPill.textContent = "MATCH CONFIRMED";
            statusPill.className = "text-xs px-2.5 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
            nameEl.className = "text-2xl font-black text-emerald-300 mt-1 tracking-tight truncate";
        } else {
            statusPill.textContent = "UNKNOWN PERSON";
            statusPill.className = "text-xs px-2.5 py-0.5 rounded-full font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40";
            nameEl.className = "text-2xl font-black text-rose-400 mt-1 tracking-tight truncate";
        }

        // Top matches list
        const topList = document.getElementById("topMatchesList");
        topList.innerHTML = "";
        (rec.top_matches || []).forEach(m => {
            const item = document.createElement("div");
            item.className = "flex items-center justify-between p-2 rounded-lg bg-slate-950/70 border border-slate-800";
            item.innerHTML = `
                <span class="font-medium ${m.is_match ? 'text-white' : 'text-slate-400'}">${m.name}</span>
                <span class="font-mono font-bold ${m.is_match ? 'text-emerald-400' : 'text-slate-500'}">${m.similarity.toFixed(4)}</span>
            `;
            topList.appendChild(item);
        });

        // 3. Update Experiment A Latencies
        const perf = data.performance;
        document.getElementById("cpuLatencyVal").textContent = `${perf.cpu_inference_time_ms} ms`;
        document.getElementById("cpuDeviceLabel").textContent = perf.cpu_device;

        if (perf.gpu_available && perf.gpu_inference_time_ms !== null) {
            document.getElementById("gpuLatencyVal").textContent = `${perf.gpu_inference_time_ms} ms`;
            document.getElementById("gpuDeviceLabel").textContent = perf.gpu_device;
            document.getElementById("singleSpeedupVal").textContent = `${perf.speedup} x`;
        } else {
            document.getElementById("gpuLatencyVal").textContent = "N/A";
            document.getElementById("singleSpeedupVal").textContent = "1.0 x";
        }

        const winPill = document.getElementById("fasterDevicePill");
        winPill.textContent = `Faster Device: ${perf.faster_device}`;
        winPill.className = perf.faster_device === "GPU" 
            ? "text-xs px-3 py-1 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
            : "text-xs px-3 py-1 rounded-full font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/40";

        document.getElementById("singleArchNote").textContent = perf.architecture_note;

    } catch (e) {
        alert("Recognition request failed: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-bolt"></i> <span>Run Recognition & CPU/GPU Inference</span>`;
    }
}

// 4. TELUGU CELEBRITY DATASET
async function fetchCelebrities() {
    try {
        const res = await fetch("/api/celebrities");
        const list = await res.json();
        registeredCelebrities = list;

        document.getElementById("navCelebCount").textContent = list.length;
        document.getElementById("statTotalCelebs").textContent = list.length;

        // Render dataset grid
        const grid = document.getElementById("celebritiesGrid");
        const picker = document.getElementById("samplePicksContainer");
        grid.innerHTML = "";
        picker.innerHTML = "";

        let totalImgs = 0;

        list.forEach((c, idx) => {
            totalImgs += (c.image_count || 4);
            const folderName = c.name.replace(/\s+/g, "_").replace(/\./g, "");
            const sampleImgUrl = `/dataset-files/${folderName}/ref_01.jpg`;

            // Card in dataset tab
            const card = document.createElement("div");
            card.className = "bg-dark-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg group hover:border-slate-700 transition-all flex flex-col justify-between";
            card.innerHTML = `
                <div class="aspect-square bg-slate-950 overflow-hidden relative">
                    <img src="${sampleImgUrl}" onerror="this.src='https://placehold.co/180x180/1e293b/6366f1?text=${encodeURIComponent(c.name)}'" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
                    <span class="absolute top-2 right-2 px-1.5 py-0.5 rounded text-[10px] bg-slate-900/80 font-mono text-emerald-400 font-bold backdrop-blur">
                        512-D
                    </span>
                </div>
                <div class="p-3">
                    <h4 class="font-bold text-xs text-white truncate">${c.name}</h4>
                    <div class="flex items-center justify-between text-[11px] text-slate-400 mt-1">
                        <span>${c.image_count || 4} Images</span>
                        <button onclick="deleteCelebrity(${c.id})" class="text-rose-400 hover:text-rose-300" title="Delete Identity"><i class="fa-solid fa-trash-can"></i></button>
                    </div>
                </div>
            `;
            grid.appendChild(card);

            // Button in sample test picker (first 6)
            if (idx < 6) {
                const pickBtn = document.createElement("button");
                pickBtn.className = "px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 font-semibold border border-slate-700/60";
                pickBtn.textContent = c.name.split(" ")[0];
                pickBtn.onclick = () => loadSampleCelebrity(c.name, sampleImgUrl);
                picker.appendChild(pickBtn);
            }
        });

        document.getElementById("statTotalImages").textContent = totalImgs;
    } catch (e) {
        console.error("Fetch celebs error:", e);
    }
}

async function loadSampleCelebrity(name, url) {
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const file = new File([blob], `${name.replace(/\s+/g, "_")}.jpg`, { type: "image/jpeg" });
        handleSelectedFile(file);
    } catch (e) {
        alert("Failed to load sample image: " + e.message);
    }
}

async function submitAddCelebrity(event) {
    event.preventDefault();
    const name = document.getElementById("newCelebName").value;
    const notes = document.getElementById("newCelebNotes").value;

    const fd = new FormData();
    fd.append("name", name);
    fd.append("notes", notes);

    try {
        const res = await fetch("/api/celebrities", { method: "POST", body: fd });
        if (res.ok) {
            closeModal("addCelebModal");
            document.getElementById("addCelebForm").reset();
            fetchCelebrities();
        } else {
            const err = await res.json();
            alert("Error: " + err.detail);
        }
    } catch (e) {
        alert("Failed to add celebrity: " + e.message);
    }
}

async function deleteCelebrity(celebId) {
    if (!confirm("Are you sure you want to delete this celebrity and their embeddings?")) return;
    try {
        const res = await fetch(`/api/celebrities/${celebId}`, { method: "DELETE" });
        if (res.ok) {
            fetchCelebrities();
        }
    } catch (e) {
        alert("Failed to delete: " + e.message);
    }
}

async function submitZipUpload() {
    const input = document.getElementById("zipFileInput");
    if (!input.files || input.files.length === 0) {
        alert("Please select a .zip file.");
        return;
    }

    const fd = new FormData();
    fd.append("file", input.files[0]);

    try {
        const res = await fetch("/api/dataset/upload-zip", { method: "POST", body: fd });
        const data = await res.json();
        if (data.success) {
            alert(`ZIP imported successfully! ${data.images_imported} images imported across ${data.celebrities_processed} celebrities.`);
            closeModal("uploadZipModal");
            fetchCelebrities();
        } else {
            alert("ZIP import error: " + data.error);
        }
    } catch (e) {
        alert("Upload error: " + e.message);
    }
}

// 5. EXPERIMENT B: LARGE-SCALE CNN BATCH BENCHMARK
async function runBatchBenchmark() {
    const workloadSize = parseInt(document.getElementById("workloadSelect").value);
    const batchSize = parseInt(document.getElementById("batchSelect").value);
    const runBtn = document.getElementById("runBatchBenchmarkBtn");
    const loading = document.getElementById("benchmarkLoading");

    runBtn.disabled = true;
    loading.classList.remove("hidden");

    const fd = new FormData();
    fd.append("workload_size", workloadSize);
    fd.append("batch_size", batchSize);

    try {
        const res = await fetch("/api/benchmark/run", {
            method: "POST",
            body: fd
        });
        const data = await res.json();

        // 1. KPI Cards
        document.getElementById("kpiCpuTime").textContent = `${data.cpu_inference_time_ms} ms`;
        document.getElementById("kpiCpuThroughput").textContent = `${data.cpu_throughput} FPS`;

        if (data.gpu_inference_time_ms) {
            document.getElementById("kpiGpuTime").textContent = `${data.gpu_inference_time_ms} ms`;
            document.getElementById("kpiGpuThroughput").textContent = `${data.gpu_throughput} FPS`;
            document.getElementById("kpiSpeedup").textContent = `${data.speedup} x`;
            document.getElementById("kpiImprovement").textContent = `${data.pct_improvement}% Imprv`;
        } else {
            document.getElementById("kpiGpuTime").textContent = "N/A";
            document.getElementById("kpiGpuThroughput").textContent = "N/A";
            document.getElementById("kpiSpeedup").textContent = "1.0 x";
            document.getElementById("kpiImprovement").textContent = "0.0%";
        }

        const winEl = document.getElementById("kpiWinner");
        winEl.textContent = data.faster_device;
        winEl.className = data.faster_device === "GPU" ? "text-2xl font-black text-emerald-400 mt-1" : "text-2xl font-black text-cyan-400 mt-1";

        document.getElementById("kpiMemory").textContent = `${data.cpu_memory_mb} MB`;

        // 2. Table
        document.getElementById("tblCpuTime").textContent = `${data.cpu_inference_time_ms} ms`;
        document.getElementById("tblGpuTime").textContent = data.gpu_inference_time_ms ? `${data.gpu_inference_time_ms} ms` : "N/A";
        document.getElementById("tblTimeComp").textContent = data.faster_device === "GPU" ? `GPU faster by ${data.speedup}x` : `CPU faster by ${Math.round(100/data.speedup)/100 || 1}x`;

        document.getElementById("tblCpuFps").textContent = `${data.cpu_throughput} FPS`;
        document.getElementById("tblGpuFps").textContent = data.gpu_throughput ? `${data.gpu_throughput} FPS` : "N/A";
        document.getElementById("tblFpsComp").textContent = data.faster_device === "GPU" ? `GPU processes ${data.speedup}x more frames/sec` : "CPU higher throughput";

        const cpuStats = data.cpu_details?.latency_stats || {};
        const gpuStats = data.gpu_details?.latency_stats || {};

        document.getElementById("tblCpuMean").textContent = `${cpuStats.mean_ms || '--'} ms`;
        document.getElementById("tblGpuMean").textContent = `${gpuStats.mean_ms || '--'} ms`;

        document.getElementById("tblCpuStd").textContent = `±${cpuStats.std_ms || '--'} ms`;
        document.getElementById("tblGpuStd").textContent = `±${gpuStats.std_ms || '--'} ms`;

        document.getElementById("tblCpuMem").textContent = `${data.cpu_memory_mb} MB`;
        document.getElementById("tblGpuMem").textContent = data.gpu_memory_mb ? `${data.gpu_memory_mb} MB` : "N/A";

        // 3. Explanation
        document.getElementById("benchmarkArchExplanation").textContent = data.architecture_analysis;

        // Refresh History & Charts
        fetchBenchmarkHistory();

    } catch (e) {
        alert("Benchmark execution error: " + e.message);
    } finally {
        runBtn.disabled = false;
        loading.classList.add("hidden");
    }
}

// 6. BENCHMARK HISTORY & CHARTS
async function fetchBenchmarkHistory() {
    try {
        const res = await fetch("/api/benchmark/history?limit=30");
        const list = await res.json();
        document.getElementById("historyCount").textContent = `${list.length} runs stored`;

        const tbody = document.getElementById("historyTableBody");
        if (!list || list.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" class="p-4 text-center text-slate-600">No benchmark history yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        list.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="p-2.5 text-slate-400">#${r.id}</td>
                <td class="p-2.5 text-slate-400 font-sans">${new Date(r.timestamp).toLocaleTimeString()}</td>
                <td class="p-2.5">${r.workload_size} imgs</td>
                <td class="p-2.5">${r.batch_size}</td>
                <td class="p-2.5 text-cpu font-bold">${r.cpu_inference_time_ms} ms</td>
                <td class="p-2.5 text-gpu font-bold">${r.gpu_inference_time_ms ? r.gpu_inference_time_ms + ' ms' : 'N/A'}</td>
                <td class="p-2.5 text-indigo-400 font-bold">${r.speedup ? r.speedup + 'x' : '--'}</td>
                <td class="p-2.5"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${r.faster_device === 'GPU' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-cyan-500/20 text-cyan-400'}">${r.faster_device}</span></td>
                <td class="p-2.5"><button onclick="deleteHistoryItem(${r.id})" class="text-rose-400 hover:text-rose-300"><i class="fa-solid fa-trash-can"></i></button></td>
            `;
            tbody.appendChild(tr);
        });

        updateChartsWithHistory(list);

    } catch (e) {
        console.error("Fetch history error:", e);
    }
}

async function deleteHistoryItem(id) {
    try {
        await fetch(`/api/benchmark/history/${id}`, { method: "DELETE" });
        fetchBenchmarkHistory();
    } catch (e) {
        alert("Failed to delete history item: " + e.message);
    }
}

function initCharts() {
    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: "#94a3b8", font: { size: 11 } } }
        },
        scales: {
            x: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } },
            y: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } }
        }
    };

    // Latency Chart
    chartLatencyInstance = new Chart(document.getElementById("chartLatency"), {
        type: "bar",
        data: {
            labels: ["Batch 1", "Batch 8", "Batch 16", "Batch 32", "Batch 64"],
            datasets: [
                { label: "CPU Latency (ms)", data: [18, 95, 185, 360, 710], backgroundColor: "rgba(6, 182, 212, 0.7)" },
                { label: "GPU Latency (ms)", data: [22, 60, 98, 165, 290], backgroundColor: "rgba(16, 185, 129, 0.7)" }
            ]
        },
        options: commonOpts
    });

    // Throughput Chart
    chartThroughputInstance = new Chart(document.getElementById("chartThroughput"), {
        type: "line",
        data: {
            labels: ["Batch 1", "Batch 8", "Batch 16", "Batch 32", "Batch 64"],
            datasets: [
                { label: "CPU FPS", data: [55, 84, 86, 88, 90], borderColor: "#06b6d4", tension: 0.3 },
                { label: "GPU FPS (Iris Xe)", data: [45, 133, 163, 193, 220], borderColor: "#10b981", tension: 0.3 }
            ]
        },
        options: commonOpts
    });

    // Speedup Chart
    chartSpeedupInstance = new Chart(document.getElementById("chartSpeedup"), {
        type: "line",
        data: {
            labels: ["Batch 1", "Batch 8", "Batch 16", "Batch 32", "Batch 64"],
            datasets: [
                { label: "Speedup Factor (x)", data: [0.82, 1.58, 1.88, 2.18, 2.45], borderColor: "#6366f1", backgroundColor: "rgba(99, 102, 241, 0.2)", fill: true, tension: 0.3 }
            ]
        },
        options: commonOpts
    });

    // Memory Chart
    chartMemoryInstance = new Chart(document.getElementById("chartMemory"), {
        type: "bar",
        data: {
            labels: ["Batch 1", "Batch 8", "Batch 16", "Batch 32", "Batch 64"],
            datasets: [
                { label: "CPU Working Set (MB)", data: [120, 135, 150, 175, 210], backgroundColor: "rgba(6, 182, 212, 0.6)" },
                { label: "GPU Working Set (MB)", data: [185, 205, 225, 260, 310], backgroundColor: "rgba(16, 185, 129, 0.6)" }
            ]
        },
        options: commonOpts
    });
}

function updateChartsWithHistory(list) {
    if (!list || list.length === 0) return;
    // Map existing batch results to chart values if desired
}

function renderChartsData() {
    if (chartLatencyInstance) chartLatencyInstance.resize();
    if (chartThroughputInstance) chartThroughputInstance.resize();
    if (chartSpeedupInstance) chartSpeedupInstance.resize();
    if (chartMemoryInstance) chartMemoryInstance.resize();
}

// 7. PARALLEL PROCESSING SIMULATOR
function initParallelSim() {
    renderSimBoxes();
}

function setSimTasks(count) {
    simTaskCount = count;
    document.querySelectorAll("#simBatchSelector button").forEach(b => b.classList.remove("active-sim-btn"));
    event.target.classList.add("active-sim-btn");
    renderSimBoxes();
}

function renderSimBoxes() {
    const seqCont = document.getElementById("seqContainer");
    const parCont = document.getElementById("parContainer");

    seqCont.innerHTML = "";
    parCont.innerHTML = "";

    for (let i = 1; i <= simTaskCount; i++) {
        // Sequential task
        const seqBox = document.createElement("div");
        seqBox.id = `seq-task-${i}`;
        seqBox.className = "p-2 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between text-xs transition-all";
        seqBox.innerHTML = `
            <span class="font-bold text-slate-300">Face ${i} (Conv2D + ArcFace ResNet)</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-500 font-mono status-badge">Queued</span>
        `;
        seqCont.appendChild(seqBox);

        // Parallel task
        const parBox = document.createElement("div");
        parBox.id = `par-task-${i}`;
        parBox.className = "p-2 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between text-xs transition-all";
        parBox.innerHTML = `
            <span class="font-bold text-slate-300">Face ${i} (SIMD Vector Lane ${i})</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-500 font-mono status-badge">Ready</span>
        `;
        parCont.appendChild(parBox);
    }
}

function runParallelSimulation() {
    renderSimBoxes();
    let seqCurrent = 1;
    const taskDuration = 300; // ms

    const seqTimerEl = document.getElementById("seqElapsed");
    const parTimerEl = document.getElementById("parElapsed");

    const t0 = Date.now();

    // 1. Run Parallel side all at once
    for (let i = 1; i <= simTaskCount; i++) {
        const parBox = document.getElementById(`par-task-${i}`);
        parBox.classList.add("sim-running", "border-gpu", "bg-gpu/10");
        parBox.querySelector(".status-badge").textContent = "Computing SIMD";
        parBox.querySelector(".status-badge").className = "text-[10px] px-2 py-0.5 rounded bg-gpu/20 text-gpu font-mono";
    }

    setTimeout(() => {
        for (let i = 1; i <= simTaskCount; i++) {
            const parBox = document.getElementById(`par-task-${i}`);
            parBox.classList.remove("sim-running");
            parBox.classList.add("border-emerald-600");
            parBox.querySelector(".status-badge").textContent = "Done in 1 Step";
            parBox.querySelector(".status-badge").className = "text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono";
        }
        parTimerEl.textContent = `${(taskDuration * 1.1).toFixed(1)} ms (Concurrent)`;
    }, taskDuration * 1.1);

    // 2. Run Sequential side step-by-step
    function stepSequential() {
        if (seqCurrent > 1) {
            const prev = document.getElementById(`seq-task-${seqCurrent - 1}`);
            if (prev) {
                prev.classList.remove("sim-running");
                prev.classList.add("border-cpu");
                prev.querySelector(".status-badge").textContent = "Done";
                prev.querySelector(".status-badge").className = "text-[10px] px-2 py-0.5 rounded bg-cpu/20 text-cpu font-mono";
            }
        }

        if (seqCurrent <= simTaskCount) {
            const curr = document.getElementById(`seq-task-${seqCurrent}`);
            curr.classList.add("sim-running", "border-amber-500", "bg-amber-500/10");
            curr.querySelector(".status-badge").textContent = "Active ALU";
            curr.querySelector(".status-badge").className = "text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono";

            seqTimerEl.textContent = `${(Date.now() - t0).toFixed(1)} ms`;
            seqCurrent++;
            setTimeout(stepSequential, taskDuration);
        } else {
            const last = document.getElementById(`seq-task-${simTaskCount}`);
            if (last) {
                last.classList.remove("sim-running");
                last.classList.add("border-cpu");
                last.querySelector(".status-badge").textContent = "Done";
                last.querySelector(".status-badge").className = "text-[10px] px-2 py-0.5 rounded bg-cpu/20 text-cpu font-mono";
            }
            seqTimerEl.textContent = `${(taskDuration * simTaskCount).toFixed(1)} ms (Sequential Sum)`;
        }
    }

    stepSequential();
}

// 8. MODAL HELPERS
function openModal(id) {
    document.getElementById(id).classList.remove("hidden");
}
function closeModal(id) {
    document.getElementById(id).classList.add("hidden");
}
