import { state } from "/static/js/shared/state.js";
import { syncPanelHeights } from "/static/js/shared/ui.js";

export function bindTabs(onTabChange) {
    const buttons = Array.from(document.querySelectorAll(".tab-button"));

    const setActiveTab = (tabId) => {
        state.activeTab = tabId;
        buttons.forEach((item) => {
            item.classList.toggle("active", item.dataset.tab === tabId);
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => {
            panel.classList.toggle("active", panel.id === tabId);
        });
        if (typeof onTabChange === "function") {
            onTabChange(tabId);
        }
        window.requestAnimationFrame(syncPanelHeights);
    };

    buttons.forEach((button) => {
        button.addEventListener("click", () => setActiveTab(button.dataset.tab));
    });
}
