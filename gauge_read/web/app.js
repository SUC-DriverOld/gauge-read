const state = {
    currentFile: null,
    currentFilePreviewUrl: null,
    resultNaturalWidth: 0,
    resultNaturalHeight: 0,
    toastTimer: null,
    batchRows: [],
    batchPage: 1,
    pageSize: 10,
    currentBatchJobId: null,
    batchPollTimer: null,
    batchRowsShown: false,
    batchPackagingNoticeShown: false
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

function fillSelect(select, values, preferred) {
    select.innerHTML = "";
    values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        const normalized = String(value || "").replace(/\\/g, "/");
        option.textContent = normalized.split("/").pop() || value;
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

function syncPanelHeights() {
    const sidebar = document.querySelector(".sidebar");
    const batchPanel = document.querySelector("#batchTab .batch-panel");
    const canvasPanel = document.querySelector(".canvas-panel");
    const resultPanel = document.querySelector(".result-panel");

    if (window.innerWidth <= 1200) {
        [canvasPanel, resultPanel, batchPanel].forEach((panel) => {
            if (panel) {
                panel.style.minHeight = "";
            }
        });
        return;
    }

    if (!sidebar) {
        return;
    }

    const sidebarHeight = sidebar.getBoundingClientRect().height;
    [canvasPanel, resultPanel, batchPanel].forEach((panel) => {
        if (panel) {
            panel.style.minHeight = `${Math.ceil(sidebarHeight)}px`;
        }
    });
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
    if (!src) {
        return;
    }
    $("imageModalImg").src = src;
    $("imageModal").classList.remove("hidden");
    document.body.classList.add("modal-open");
}

function closeImageModal() {
    $("imageModal").classList.add("hidden");
    $("imageModalImg").removeAttribute("src");
    document.body.classList.remove("modal-open");
}

function scrollBatchTableToTop() {
    const wrap = $("batchTableWrap");
    if (!wrap) {
        return;
    }
    window.requestAnimationFrame(() => {
        wrap.scrollIntoView({ behavior: "smooth", block: "start" });
    });
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
            scrollBatchTableToTop();
        }
    });
    $("nextPageBtn").addEventListener("click", () => {
        if (state.batchPage < totalPages) {
            state.batchPage += 1;
            renderBatchTable();
            scrollBatchTableToTop();
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
    const isMobile = window.innerWidth <= 560;

    if (isMobile) {
        const cards = pageRows.map((row) => `
            <article class="mobile-batch-card">
                <div class="mobile-batch-thumb">
                    <img class="thumb-action" data-fullsrc="${row.full_image || row.thumbnail}" src="${row.thumbnail}" alt="${row.filename}">
                    <span>${row.filename}</span>
                </div>
                <div class="mobile-batch-metrics">
                    <div class="mobile-batch-item">
                        <label>起始值</label>
                        <strong>${row.start}</strong>
                    </div>
                    <div class="mobile-batch-item">
                        <label>结束值</label>
                        <strong>${row.end}</strong>
                    </div>
                    <div class="mobile-batch-item">
                        <label>读数 Ratio</label>
                        <strong>${row.ratio}</strong>
                    </div>
                    <div class="mobile-batch-item">
                        <label>读数值</label>
                        <strong>${row.reading}</strong>
                    </div>
                </div>
            </article>
        `).join("");

        wrap.className = "table-wrap";
        wrap.innerHTML = `<div class="mobile-batch-list">${cards}</div>`;

        wrap.querySelectorAll(".thumb-action").forEach((image) => {
            image.addEventListener("click", () => openImageModal(image.dataset.fullsrc));
        });

        renderPagination(totalPages);
        return;
    }

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
}

async function loadModels() {
    const button = $("loadModelsBtn");
    setLoading(button, true, "加载中...");
    try {
        const payload = await request("/api/models/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_path: $("modelSelect").value,
                stn_path: $("stnSelect").value,
                yolo_path: $("yoloSelect").value
            })
        });
        if (payload.config_mode === "matched") {
            showToast(`模型加载完成，已匹配配置: ${payload.config_path}`);
        } else {
            showToast("模型加载完成，未找到同名配置，已使用默认配置");
        }
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

async function onBatchImagesSelected(event) {
    const files = Array.from(event.target.files || []);
    const button = $("uploadBatchImagesBtn");
    const hint = $("batchUploadHint");
    if (!files.length) {
        hint.textContent = "未选择图片";
        return;
    }

    setLoading(button, true, "上传中...");
    try {
        const formData = new FormData();
        files.forEach((file) => formData.append("images", file));
        const payload = await request("/api/batch/uploads", { method: "POST", body: formData });
        $("batchDirInput").value = payload.input_dir;
        hint.textContent = `已选择 ${payload.count} 张图片`;
        showToast(`已上传 ${payload.count} 张图片，批量路径已自动填入`);
    } catch (error) {
        hint.textContent = "未选择图片";
        showToast(error.message, true);
    } finally {
        event.target.value = "";
        setLoading(button, false, "");
    }
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

        if ((payload.rows || []).length) {
            state.batchRows = payload.rows || [];
            if (!state.batchRowsShown) {
                state.batchPage = 1;
                renderBatchTable();
                scrollBatchTableToTop();
                state.batchRowsShown = true;
            }
        }

        if (payload.status === "packaging") {
            button.disabled = true;
            button.textContent = "打包中...";
            if (!state.batchPackagingNoticeShown) {
                showToast("推理已完成，正在打包下载文件");
                state.batchPackagingNoticeShown = true;
            }
            state.batchPollTimer = window.setTimeout(() => pollBatchJob(jobId, button), 600);
            return;
        }

        if (payload.status === "completed") {
            state.batchRows = payload.rows || [];
            if (!state.batchRowsShown) {
                state.batchPage = 1;
                renderBatchTable();
                scrollBatchTableToTop();
            }
            setDownloadLink("downloadZip", payload.downloads.zip);
            setDownloadLink("downloadCsv", payload.downloads.csv);
            setLoading(button, false, "");
            showToast(`批量推理完成，共 ${payload.rows.length} 张`);
            state.currentBatchJobId = null;
            state.batchPollTimer = null;
            state.batchRowsShown = false;
            state.batchPackagingNoticeShown = false;
            return;
        }

        if (payload.status === "failed") {
            setLoading(button, false, "");
            showToast(payload.error || "批量推理失败", true);
            state.currentBatchJobId = null;
            state.batchPollTimer = null;
            state.batchRowsShown = false;
            state.batchPackagingNoticeShown = false;
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
    state.batchRowsShown = false;
    state.batchPackagingNoticeShown = false;
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
    const buttons = Array.from(document.querySelectorAll(".tab-button"));

    const setActiveTab = (tabId) => {
        buttons.forEach((item) => {
            item.classList.toggle("active", item.dataset.tab === tabId);
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => {
            panel.classList.toggle("active", panel.id === tabId);
        });
        window.requestAnimationFrame(syncPanelHeights);
    };

    buttons.forEach((button) => {
        button.addEventListener("click", () => setActiveTab(button.dataset.tab));
    });
}

function bindEvents() {
    $("loadModelsBtn").addEventListener("click", loadModels);
    $("imageInput").addEventListener("change", onImageSelected);
    $("uploadBatchImagesBtn").addEventListener("click", () => $("batchImagesInput").click());
    $("batchImagesInput").addEventListener("change", onBatchImagesSelected);
    $("runSingleBtn").addEventListener("click", runSingleInference);
    $("runBatchBtn").addEventListener("click", runBatch);
    $("resultImage").addEventListener("click", handleResultClick);
    $("debugImage").addEventListener("click", () => {
        if ($("debugStage").classList.contains("empty") || !$("debugImage").src) {
            return;
        }
        openImageModal($("debugImage").src);
    });
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
    window.requestAnimationFrame(syncPanelHeights);
    window.addEventListener("resize", () => {
        syncPanelHeights();
        if (state.batchRows.length) {
            renderBatchTable();
        }
    });
});
