import { $, state } from "/static/js/shared/state.js";

export function showToast(message, isError = false) {
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

export async function request(url, options = {}) {
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

export function fillSelect(select, values, preferred) {
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

export function syncPanelHeights() {
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

export function setLoading(button, loading, loadingText) {
    if (!button.dataset.originalText) {
        button.dataset.originalText = button.textContent;
    }
    button.disabled = loading;
    button.textContent = loading ? loadingText : button.dataset.originalText;
}

export function getCurrentEditMode() {
    const checked = document.querySelector('input[name="editMode"]:checked');
    return checked ? checked.value : "未选择";
}

export function setDownloadLink(id, href) {
    const link = $(id);
    link.href = href || "#";
    link.setAttribute("aria-disabled", href ? "false" : "true");
    link.classList.toggle("disabled", !href);
}

export function openImageModal(src) {
    if (!src) {
        return;
    }
    $("imageModalImg").src = src;
    $("imageModal").classList.remove("hidden");
    document.body.classList.add("modal-open");
}

export function closeImageModal() {
    $("imageModal").classList.add("hidden");
    $("imageModalImg").removeAttribute("src");
    document.body.classList.remove("modal-open");
}

export function bindDropZone(zone, onFiles) {
    ["dragenter", "dragover"].forEach((eventName) => {
        zone.addEventListener(eventName, (event) => {
            event.preventDefault();
            zone.classList.add("dragover");
        });
    });

    ["dragleave", "dragend", "drop"].forEach((eventName) => {
        zone.addEventListener(eventName, (event) => {
            event.preventDefault();
            if (eventName === "dragleave" && zone.contains(event.relatedTarget)) {
                return;
            }
            zone.classList.remove("dragover");
        });
    });

    zone.addEventListener("drop", (event) => {
        const files = Array.from(event.dataTransfer?.files || []);
        onFiles(files);
    });
}
