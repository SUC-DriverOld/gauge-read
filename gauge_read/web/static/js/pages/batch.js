import { $, state } from "/static/js/shared/state.js";
import { bindDropZone, openImageModal, request, setDownloadLink, setLoading, showToast } from "/static/js/shared/ui.js";

function setBatchUploadPreview(count) {
    const empty = $("batchUploadEmptyState");
    const preview = $("batchUploadPreviewState");
    const countLabel = $("batchUploadCount");
    const hasFiles = count > 0;
    countLabel.textContent = hasFiles ? `已上传 ${count} 张图片` : "已上传 0 张图片";
    empty.classList.toggle("hidden", hasFiles);
    preview.classList.toggle("hidden", !hasFiles);
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

async function uploadBatchFiles(files) {
    if (!files.length) {
        state.batchUploadDir = "";
        setBatchUploadPreview(0);
        return;
    }

    const uploadZone = $("batchUploadZone");
    uploadZone.classList.add("loading");
    try {
        const formData = new FormData();
        files.forEach((file) => formData.append("images", file));
        const payload = await request("/api/batch/uploads", { method: "POST", body: formData });
        state.batchUploadDir = payload.input_dir;
        setBatchUploadPreview(payload.count);
        showToast(`已上传 ${payload.count} 张图片`);
    } catch (error) {
        showToast(error.message, true);
    } finally {
        uploadZone.classList.remove("loading");
    }
}

async function onBatchImagesSelected(event) {
    const files = Array.from(event.target.files || []);
    try {
        await uploadBatchFiles(files);
    } finally {
        event.target.value = "";
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
            window.setTimeout(() => pollBatchJob(jobId, button), 600);
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
            state.batchRowsShown = false;
            state.batchPackagingNoticeShown = false;
            return;
        }

        if (payload.status === "failed") {
            setLoading(button, false, "");
            showToast(payload.error || "批量推理失败", true);
            state.batchRowsShown = false;
            state.batchPackagingNoticeShown = false;
            return;
        }

        window.setTimeout(() => pollBatchJob(jobId, button), 600);
    } catch (error) {
        setLoading(button, false, "");
        showToast(error.message, true);
    }
}

async function runBatch() {
    if (!state.batchUploadDir) {
        showToast("请先上传批量图片", true);
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
                input_dir: state.batchUploadDir,
                use_stn: $("batchUseStn").checked,
                use_yolo: $("batchUseYolo").checked
            })
        });
        pollBatchJob(payload.job_id, button);
    } catch (error) {
        setLoading(button, false, "");
        showToast(error.message, true);
    }
}

export function bindBatchPage() {
    $("batchImagesInput").addEventListener("change", onBatchImagesSelected);
    $("runBatchBtn").addEventListener("click", runBatch);
    bindDropZone($("batchUploadZone"), (files) => uploadBatchFiles(files));
}

export function handleBatchResize() {
    if (state.batchRows.length) {
        renderBatchTable();
    }
}
