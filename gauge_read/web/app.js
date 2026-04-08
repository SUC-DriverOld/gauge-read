const state = {
    currentFile: null,
    resultNaturalWidth: 0,
    resultNaturalHeight: 0,
    toastTimer: null,
    batchRows: [],
    batchPage: 1,
    pageSize: 10,
    currentBatchJobId: null,
    batchPollTimer: null
};

function $(id) {
    return document.getElementById(id);
}

function showToast(message, isError = false) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.remove("hidden", "error");
    if (isError) {
        toast.classList.add("error");
    }
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
        toast.classList.add("hidden");
    }, 3200);
}

async function request(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        let detail = "请求失败";
        try {
            const payload = await response.json();
            detail = payload.detail || detail;
        } catch (_error) {
            detail = response.statusText || detail;
        }
        throw new Error(detail);
    }
    return response.json();
}

function setModelStatus(loaded) {
    const badge = $("modelStatus");
    badge.textContent = loaded ? "已加载" : "未加载";
    badge.className = `status-badge ${loaded ? "ready" : "idle"}`;
}

function fillSelect(select, values, preferred) {
    select.innerHTML = "";
    values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        if (value === preferred) {
            option.selected = true;
        }
        select.appendChild(option);
    });
    if (!values.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "未找到可用模型";
        select.appendChild(option);
    }
}

function setLoading(button, loading, loadingText) {
    if (!button.dataset.originalText) {
        button.dataset.originalText = button.textContent;
    }
    button.disabled = loading;
    button.textContent = loading ? loadingText : button.dataset.originalText;
}

function applyResult(payload) {
    $("resultImage").src = payload.result_image;
    $("resultStage").classList.remove("empty");
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
    if (!file) {
        empty.classList.remove("hidden");
        preview.classList.add("hidden");
        image.removeAttribute("src");
        name.textContent = "";
        return;
    }
    const fileUrl = URL.createObjectURL(file);
    image.src = fileUrl;
    name.textContent = file.name;
    empty.classList.add("hidden");
    preview.classList.remove("hidden");
}

function getCurrentEditMode() {
    const checked = document.querySelector('input[name="editMode"]:checked');
    return checked ? checked.value : "未选择";
}

function setDownloadLink(id, href) {
    const link = $(id);
    link.href = href || "#";
    link.setAttribute("aria-disabled", href ? "false" : "true");
    link.classList.toggle("disabled", !href);
}

function openImageModal(src) {
    $("imageModalImg").src = src;
    $("imageModal").classList.remove("hidden");
}

function closeImageModal() {
    $("imageModal").classList.add("hidden");
    $("imageModalImg").removeAttribute("src");
}

function renderPagination(totalPages) {
    const wrap = $("paginationWrap");
    if (totalPages <= 1) {
        wrap.classList.add("hidden");
        wrap.innerHTML = "";
        return;
    }

    wrap.classList.remove("hidden");
    wrap.innerHTML = `
        <button id="prevPageBtn" class="button secondary" type="button" ${state.batchPage <= 1 ? "disabled" : ""}>上一页</button>
        <span class="pagination-info">${state.batchPage}/${totalPages}</span>
        <button id="nextPageBtn" class="button secondary" type="button" ${state.batchPage >= totalPages ? "disabled" : ""}>下一页</button>
    `;

    $("prevPageBtn").addEventListener("click", () => {
        if (state.batchPage > 1) {
            state.batchPage -= 1;
            renderBatchTable();
            $("batchTableWrap").scrollIntoView({ behavior: "smooth", block: "start" });
        }
    });
    $("nextPageBtn").addEventListener("click", () => {
        if (state.batchPage < totalPages) {
            state.batchPage += 1;
            renderBatchTable();
            $("batchTableWrap").scrollIntoView({ behavior: "smooth", block: "start" });
        }
    });
}

function renderBatchTable() {
    const wrap = $("batchTableWrap");
    const rows = state.batchRows;
    if (!rows.length) {
        wrap.className = "table-wrap empty";
        wrap.innerHTML = '<div class="placeholder-copy">没有可显示的批量结果</div>';
        $("paginationWrap").classList.add("hidden");
        return;
    }

    const totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
    state.batchPage = Math.min(state.batchPage, totalPages);
    const start = (state.batchPage - 1) * state.pageSize;
    const pageRows = rows.slice(start, start + state.pageSize);

    const body = pageRows.map((row) => `
        <tr>
            <td>
                <div class="thumb-block">
                    <img class="thumb-action" data-fullsrc="${row.full_image || row.thumbnail}" src="${row.thumbnail}" alt="${row.filename}">
                    <span>${row.filename}</span>
                </div>
            </td>
            <td>${row.start}</td>
            <td>${row.end}</td>
            <td>${row.ratio}</td>
            <td>${row.reading}</td>
        </tr>
    `).join("");

    wrap.className = "table-wrap";
    wrap.innerHTML = `
        <table class="results-table">
            <thead>
                <tr>
                    <th>结果图片</th>
                    <th>起始值</th>
                    <th>结束值</th>
                    <th>读数 Ratio</th>
                    <th>读数值</th>
                </tr>
            </thead>
            <tbody>${body}</tbody>
        </table>
    `;

    wrap.querySelectorAll(".thumb-action").forEach((image) => {
        image.addEventListener("click", () => openImageModal(image.dataset.fullsrc));
    });

    renderPagination(totalPages);
}

function updateBatchProgress(completed, total) {
    const wrap = $("batchProgressWrap");
    const value = total > 0 ? Math.min(100, (completed / total) * 100) : 0;
    wrap.classList.remove("hidden");
    $("batchProgressText").textContent = `${completed}/${total}`;
    $("batchProgressBar").style.width = `${value}%`;
}

async function bootstrap() {
    const payload = await request("/api/bootstrap");
    fillSelect($("modelSelect"), payload.model_options, payload.defaults.model_path);
    fillSelect($("stnSelect"), payload.stn_options, payload.defaults.stn_path);
    fillSelect($("yoloSelect"), payload.yolo_options, payload.defaults.yolo_path);
    $("instructionList").innerHTML = payload.instructions.map((item) => `<li>${item}</li>`).join("");
    setModelStatus(payload.loaded);
}

async function loadModels() {
    const button = $("loadModelsBtn");
    setLoading(button, true, "加载中...");
    try {
        await request("/api/models/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_path: $("modelSelect").value,
                stn_path: $("stnSelect").value,
                yolo_path: $("yoloSelect").value
            })
        });
        setModelStatus(true);
        showToast("模型加载完成");
    } catch (error) {
        showToast(error.message, true);
    } finally {
        setLoading(button, false, "");
    }
}

function onImageSelected(event) {
    const [file] = event.target.files || [];
    state.currentFile = file || null;
    $("uploadHint").textContent = file ? `已选择: ${file.name}` : "支持 JPG / PNG / BMP / WEBP";
    setInputPreview(state.currentFile);
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

async function pollBatchJob(jobId, button) {
    try {
        const payload = await request(`/api/batch/jobs/${jobId}`);
        updateBatchProgress(payload.progress.completed, payload.progress.total);

        if (payload.status === "completed") {
            state.batchRows = payload.rows || [];
            state.batchPage = 1;
            renderBatchTable();
            setDownloadLink("downloadZip", payload.downloads.zip);
            setDownloadLink("downloadCsv", payload.downloads.csv);
            setLoading(button, false, "");
            showToast(`批量推理完成，共 ${payload.rows.length} 张`);
            state.currentBatchJobId = null;
            state.batchPollTimer = null;
            return;
        }

        if (payload.status === "failed") {
            setLoading(button, false, "");
            showToast(payload.error || "批量推理失败", true);
            state.currentBatchJobId = null;
            state.batchPollTimer = null;
            return;
        }

        state.batchPollTimer = window.setTimeout(() => pollBatchJob(jobId, button), 600);
    } catch (error) {
        setLoading(button, false, "");
        showToast(error.message, true);
        state.currentBatchJobId = null;
        state.batchPollTimer = null;
    }
}

async function runBatch() {
    const inputDir = $("batchDirInput").value.trim();
    if (!inputDir) {
        showToast("请输入图片文件夹路径", true);
        return;
    }

    const button = $("runBatchBtn");
    state.batchRows = [];
    renderBatchTable();
    setDownloadLink("downloadZip", "");
    setDownloadLink("downloadCsv", "");
    updateBatchProgress(0, 0);
    setLoading(button, true, "处理中...");

    try {
        const payload = await request("/api/batch/jobs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                input_dir: inputDir,
                use_stn: $("batchUseStn").checked,
                use_yolo: $("batchUseYolo").checked
            })
        });
        state.currentBatchJobId = payload.job_id;
        pollBatchJob(payload.job_id, button);
    } catch (error) {
        setLoading(button, false, "");
        showToast(error.message, true);
    }
}

function bindTabs() {
    document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
            button.classList.add("active");
            $(button.dataset.tab).classList.add("active");
        });
    });
}

function bindEvents() {
    $("loadModelsBtn").addEventListener("click", loadModels);
    $("imageInput").addEventListener("change", onImageSelected);
    $("runSingleBtn").addEventListener("click", runSingleInference);
    $("runBatchBtn").addEventListener("click", runBatch);
    $("resultImage").addEventListener("click", handleResultClick);
    $("startValueInput").addEventListener("change", (event) => updateValue("start", event.target.value));
    $("endValueInput").addEventListener("change", (event) => updateValue("end", event.target.value));
    $("imageModalClose").addEventListener("click", closeImageModal);
    $("imageModalBackdrop").addEventListener("click", closeImageModal);
    bindTabs();
}

window.addEventListener("DOMContentLoaded", async () => {
    bindEvents();
    try {
        await bootstrap();
    } catch (error) {
        showToast(error.message, true);
    }
});
