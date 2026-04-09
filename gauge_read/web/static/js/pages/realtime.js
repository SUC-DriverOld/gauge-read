import { $, state } from "/static/js/shared/state.js";
import { request, showToast } from "/static/js/shared/ui.js";

function setRealtimeManualInputsEnabled(enabled) {
    $("realtimeValueGrid").classList.toggle("hidden", !enabled);
    $("realtimeStartValueInput").disabled = !enabled;
    $("realtimeEndValueInput").disabled = !enabled;
}

function updateRealtimeMetrics(payload = {}) {
    $("realtimeReadingLabel").textContent = payload.reading_label || "读数结果";
    $("realtimeReading").textContent = payload.reading || "-";
    $("realtimeRatio").textContent = payload.ratio || "-";
    $("realtimeReadingTime").textContent = payload.time || "--:--:--";
    $("realtimeRatioTime").textContent = payload.time || "--:--:--";
    if (payload.start_value !== undefined) {
        $("realtimeStartValueInput").value = payload.start_value;
    }
    if (payload.end_value !== undefined) {
        $("realtimeEndValueInput").value = payload.end_value;
    }
}

function getRealtimeClockText() {
    return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function setRealtimeStatus(text, isError = false, elapsedMs = null) {
    const node = $("realtimeStatus");
    const effectiveElapsed = typeof elapsedMs === "number" && Number.isFinite(elapsedMs)
        ? elapsedMs
        : state.realtime.lastElapsedMs;
    const suffix = typeof effectiveElapsed === "number" && Number.isFinite(effectiveElapsed)
        ? ` · 耗时 ${(effectiveElapsed / 1000).toFixed(2)} 秒`
        : "";
    node.textContent = `${text}${suffix}`;
    node.classList.toggle("error", isError);
}

async function ensureCameraAccess(deviceId = "") {
    const legacyGetUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    const getLegacyStream = (constraints) => new Promise((resolve, reject) => {
        if (!legacyGetUserMedia) {
            reject(new Error("UNSUPPORTED_CAMERA_API"));
            return;
        }
        legacyGetUserMedia.call(navigator, constraints, resolve, reject);
    });

    const attempts = deviceId
        ? [
            { video: { deviceId: { exact: deviceId } }, audio: false },
            { video: { deviceId }, audio: false },
            { video: true, audio: false }
        ]
        : [
            { video: { facingMode: { ideal: "environment" } }, audio: false },
            { video: { facingMode: "environment" }, audio: false },
            { video: { facingMode: { ideal: "user" } }, audio: false },
            { video: true, audio: false }
        ];

    let lastError = null;
    for (const constraints of attempts) {
        try {
            if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") {
                return await navigator.mediaDevices.getUserMedia(constraints);
            }
            return await getLegacyStream(constraints);
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError || new Error("无法打开摄像头");
}

export function stopRealtimeCamera() {
    if (state.realtime.stream) {
        state.realtime.stream.getTracks().forEach((track) => track.stop());
        state.realtime.stream = null;
    }
    $("cameraVideo").pause();
    $("cameraVideo").srcObject = null;
    $("cameraVideo").classList.add("hidden");
    $("cameraStageEmpty").classList.remove("hidden");
}

async function refreshCameraDevices() {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== "function") {
        $("cameraControls").classList.add("hidden");
        state.realtime.devices = [];
        return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    state.realtime.devices = devices.filter((device) => device.kind === "videoinput");
    const select = $("cameraSelect");
    select.innerHTML = "";
    state.realtime.devices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `摄像头 ${index + 1}`;
        if (device.deviceId === state.realtime.selectedDeviceId) {
            option.selected = true;
        }
        select.appendChild(option);
    });
    $("cameraControls").classList.toggle("hidden", state.realtime.devices.length === 0);
}

async function startRealtimeCamera(deviceId = "") {
    try {
        stopRealtimeCamera();
        const stream = await ensureCameraAccess(deviceId);
        state.realtime.stream = stream;
        const [track] = stream.getVideoTracks();
        state.realtime.selectedDeviceId = track?.getSettings?.().deviceId || deviceId || "";
        $("cameraVideo").srcObject = stream;
        $("cameraVideo").classList.remove("hidden");
        $("cameraStageEmpty").classList.add("hidden");
        await $("cameraVideo").play();
        await refreshCameraDevices();
    } catch (error) {
        stopRealtimeCamera();
        const errorName = error?.name || error?.message || "UnknownError";
        const isInsecureContext = window.isSecureContext === false
            && location.hostname !== "localhost"
            && location.hostname !== "127.0.0.1";

        let message = `摄像头开启失败（${errorName}）`;
        if (isInsecureContext) {
            message = "摄像头开启失败：当前页面不是安全上下文，请使用 HTTPS 或 localhost 访问";
        } else if (errorName === "TypeError" || errorName === "UNSUPPORTED_CAMERA_API") {
            message = "摄像头开启失败：当前浏览器环境不支持摄像头接口，或当前访问方式不被允许";
        } else if (errorName === "NotAllowedError") {
            message = "摄像头开启失败：浏览器未允许摄像头权限";
        } else if (errorName === "NotFoundError") {
            message = "摄像头开启失败：未检测到可用摄像头";
        } else if (errorName === "OverconstrainedError") {
            message = "摄像头开启失败：当前摄像头约束不兼容，已尝试自动降级";
        }

        showToast(message, true);
        throw error;
    }
}

function captureRealtimeFrameBlob() {
    const video = $("cameraVideo");
    const canvas = $("realtimeCaptureCanvas");
    if (!video.videoWidth || !video.videoHeight) {
        return null;
    }

    const maxWidth = 960;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.72);
    });
}

async function sendRealtimeFrame() {
    if (!state.realtime.active || state.realtime.inflight || state.activeTab !== "realtimeTab") {
        return;
    }
    if (!state.realtime.stream) {
        return;
    }

    const blob = await captureRealtimeFrameBlob();
    if (!blob) {
        window.setTimeout(() => {
            if (state.realtime.active) {
                sendRealtimeFrame();
            }
        }, 500);
        return;
    }

    state.realtime.inflight = true;
    const startedAt = performance.now();
    setRealtimeStatus("推理中...");
    try {
        const manualEnabled = $("realtimeManualToggle").checked;
        const formData = new FormData();
        formData.append("image", blob, "realtime.jpg");
        formData.append("use_stn", $("realtimeUseStn").checked ? "true" : "false");
        formData.append("use_yolo", $("realtimeUseYolo").checked ? "true" : "false");
        formData.append("manual_values", manualEnabled ? "true" : "false");
        formData.append("start_value", $("realtimeStartValueInput").value || "0");
        formData.append("end_value", $("realtimeEndValueInput").value || "0");

        const payload = await request("/api/realtime/frame", { method: "POST", body: formData });
        if (payload.valid) {
            const timestamp = getRealtimeClockText();
            const elapsedMs = performance.now() - startedAt;
            state.realtime.lastValidReading = payload.reading;
            state.realtime.lastValidRatio = payload.ratio;
            state.realtime.lastValidReadingLabel = payload.reading_label || "读数结果";
            state.realtime.lastValidTime = timestamp;
            state.realtime.lastElapsedMs = elapsedMs;
            updateRealtimeMetrics({ ...payload, time: timestamp });
            setRealtimeStatus("等待下一帧", false, elapsedMs);
        } else {
            const elapsedMs = performance.now() - startedAt;
            state.realtime.lastElapsedMs = elapsedMs;
            updateRealtimeMetrics({
                reading_label: state.realtime.lastValidReadingLabel,
                reading: state.realtime.lastValidReading,
                ratio: state.realtime.lastValidRatio,
                time: state.realtime.lastValidTime,
                start_value: $("realtimeStartValueInput").value,
                end_value: $("realtimeEndValueInput").value
            });
            const reason = payload.reason || "当前帧无有效结果";
            setRealtimeStatus(`等待下一帧，当前帧无效：${reason}`, true, elapsedMs);
        }
    } catch (error) {
        const elapsedMs = performance.now() - startedAt;
        state.realtime.lastElapsedMs = elapsedMs;
        showToast(error.message, true);
        state.realtime.active = false;
        $("toggleRealtimeBtn").textContent = "开始实时推理";
        setRealtimeStatus("实时推理已停止", true, elapsedMs);
    } finally {
        state.realtime.inflight = false;
    }

    if (state.realtime.active && state.activeTab === "realtimeTab") {
        window.setTimeout(() => sendRealtimeFrame(), 500);
    }
}

async function openCameraStage() {
    if (state.realtime.stream) {
        return;
    }
    await startRealtimeCamera(state.realtime.selectedDeviceId);
    setRealtimeStatus("摄像头已连接");
}

async function handleCameraSelectChange(event) {
    const deviceId = event.target.value;
    if (!deviceId) {
        return;
    }
    await startRealtimeCamera(deviceId);
    if (state.realtime.active && !state.realtime.inflight) {
        sendRealtimeFrame();
    }
}

async function toggleRealtimeInference() {
    const button = $("toggleRealtimeBtn");
    if (!state.realtime.stream) {
        await openCameraStage();
    }

    if (state.realtime.active) {
        state.realtime.active = false;
        button.textContent = "开始实时推理";
        setRealtimeStatus("实时推理已关闭");
        return;
    }

    state.realtime.active = true;
    button.textContent = "关闭实时推理";
    setRealtimeStatus("等待当前帧");
    sendRealtimeFrame();
}

export function initRealtimePage() {
    setRealtimeManualInputsEnabled(false);
    updateRealtimeMetrics({
        reading_label: "读数结果",
        reading: "-",
        ratio: "-",
        time: "--:--:--"
    });
    setRealtimeStatus("待开始");
}

export function bindRealtimePage() {
    $("cameraStage").addEventListener("click", openCameraStage);
    $("cameraSelect").addEventListener("change", handleCameraSelectChange);
    $("toggleRealtimeBtn").addEventListener("click", toggleRealtimeInference);
    $("realtimeManualToggle").addEventListener("change", (event) => {
        setRealtimeManualInputsEnabled(event.target.checked);
    });
}

export function onRealtimeTabActivated() {
    if (state.realtime.active && !state.realtime.inflight) {
        sendRealtimeFrame();
    }
}
