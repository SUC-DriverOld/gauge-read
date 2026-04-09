import { $, state } from "/static/js/shared/state.js";
import {
    bindDropZone,
    getCurrentEditMode,
    openImageModal,
    request,
    setLoading,
    showToast
} from "/static/js/shared/ui.js";

function applyResult(payload) {
    $("resultImage").src = payload.result_image;
    $("resultStage").classList.remove("empty");
    const debugStage = $("debugStage");
    const debugImage = $("debugImage");
    if (payload.debug_image) {
        debugImage.src = payload.debug_image;
        debugStage.classList.remove("empty");
    } else {
        debugImage.removeAttribute("src");
        debugStage.classList.add("empty");
    }
    $("heroReadingLabel").textContent = payload.ocr_error ? "OCR失败" : "读数结果";
    $("heroReading").textContent = payload.ocr_error ? "OCR失败" : payload.reading;
    $("heroRatio").textContent = payload.ratio;
    $("startValueInput").value = payload.start_value;
    $("endValueInput").value = payload.end_value;
    state.resultNaturalWidth = payload.image_size?.width || 0;
    state.resultNaturalHeight = payload.image_size?.height || 0;
}

function setInputPreview(file) {
    const empty = $("uploadEmptyState");
    const preview = $("uploadPreviewState");
    const image = $("inputPreviewImage");
    const name = $("inputPreviewName");
    if (state.currentFilePreviewUrl) {
        URL.revokeObjectURL(state.currentFilePreviewUrl);
        state.currentFilePreviewUrl = null;
    }
    if (!file) {
        empty.classList.remove("hidden");
        preview.classList.add("hidden");
        image.removeAttribute("src");
        name.textContent = "";
        return;
    }
    state.currentFilePreviewUrl = URL.createObjectURL(file);
    image.src = state.currentFilePreviewUrl;
    name.textContent = file.name;
    empty.classList.add("hidden");
    preview.classList.remove("hidden");
}

function applySingleFile(file) {
    state.currentFile = file || null;
    $("uploadHint").textContent = file ? `已选择: ${file.name}` : "支持 JPG / PNG / BMP / WEBP，也可直接拖拽上传";
    setInputPreview(state.currentFile);
}

function onImageSelected(event) {
    const [file] = event.target.files || [];
    applySingleFile(file || null);
}

async function runSingleInference() {
    if (!state.currentFile) {
        showToast("请先上传图片", true);
        return;
    }
    const button = $("runSingleBtn");
    setLoading(button, true, "推理中...");
    try {
        const formData = new FormData();
        formData.append("image", state.currentFile);
        formData.append("use_stn", $("singleUseStn").checked ? "true" : "false");
        formData.append("use_yolo", $("singleUseYolo").checked ? "true" : "false");
        const payload = await request("/api/infer", { method: "POST", body: formData });
        applyResult(payload);
        showToast("单图推理完成");
    } catch (error) {
        showToast(error.message, true);
    } finally {
        setLoading(button, false, "");
    }
}

async function updateValue(field, value) {
    if (!$("resultImage").src) {
        return;
    }
    try {
        const payload = await request("/api/session/update-value", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ field, value })
        });
        applyResult(payload);
    } catch (error) {
        showToast(error.message, true);
    }
}

async function handleResultClick(event) {
    const mode = getCurrentEditMode();
    if (mode === "未选择") {
        showToast("请先选择修正模式", true);
        return;
    }
    if (!state.resultNaturalWidth || !state.resultNaturalHeight) {
        return;
    }

    const image = $("resultImage");
    const rect = image.getBoundingClientRect();
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;
    const x = Math.max(0, Math.min(state.resultNaturalWidth - 1, Math.round(displayX * (state.resultNaturalWidth / rect.width))));
    const y = Math.max(0, Math.min(state.resultNaturalHeight - 1, Math.round(displayY * (state.resultNaturalHeight / rect.height))));

    try {
        const payload = await request("/api/session/update-point", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode, x, y })
        });
        applyResult(payload);
        showToast(`${mode} 已更新`);
    } catch (error) {
        showToast(error.message, true);
    }
}

export function bindSinglePage() {
    $("imageInput").addEventListener("change", onImageSelected);
    $("runSingleBtn").addEventListener("click", runSingleInference);
    $("resultImage").addEventListener("click", handleResultClick);
    $("debugImage").addEventListener("click", () => {
        if ($("debugStage").classList.contains("empty") || !$("debugImage").src) {
            return;
        }
        openImageModal($("debugImage").src);
    });
    $("startValueInput").addEventListener("change", (event) => updateValue("start", event.target.value));
    $("endValueInput").addEventListener("change", (event) => updateValue("end", event.target.value));
    bindDropZone($("singleUploadZone"), (files) => applySingleFile(files[0] || null));
}
