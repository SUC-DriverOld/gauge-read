import { $ } from "/static/js/shared/state.js";
import { bootstrap, loadModels } from "/static/js/shared/models.js";
import { bindTabs } from "/static/js/shared/tabs.js";
import { closeImageModal, showToast, syncPanelHeights } from "/static/js/shared/ui.js";
import { bindBatchPage, handleBatchResize } from "/static/js/pages/batch.js";
import { bindRealtimePage, initRealtimePage, onRealtimeTabActivated, stopRealtimeCamera } from "/static/js/pages/realtime.js";
import { bindSinglePage } from "/static/js/pages/single.js";

function bindSharedEvents() {
    $("loadModelsBtn").addEventListener("click", loadModels);
    $("imageModalClose").addEventListener("click", closeImageModal);
    $("imageModalBackdrop").addEventListener("click", closeImageModal);
    bindTabs((tabId) => {
        if (tabId === "realtimeTab") {
            onRealtimeTabActivated();
        }
    });
}

window.addEventListener("DOMContentLoaded", async () => {
    bindSharedEvents();
    bindSinglePage();
    bindBatchPage();
    bindRealtimePage();
    initRealtimePage();

    try {
        await bootstrap();
    } catch (error) {
        showToast(error.message, true);
    }

    window.requestAnimationFrame(syncPanelHeights);
    window.addEventListener("resize", () => {
        syncPanelHeights();
        handleBatchResize();
    });
});

window.addEventListener("beforeunload", () => {
    stopRealtimeCamera();
});
