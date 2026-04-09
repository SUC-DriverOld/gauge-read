import { $ } from "/static/js/shared/state.js";
import { fillSelect, request, setLoading, showToast } from "/static/js/shared/ui.js";

export async function bootstrap() {
    const payload = await request("/api/bootstrap");
    fillSelect($("modelSelect"), payload.model_options, payload.defaults.model_path);
    fillSelect($("stnSelect"), payload.stn_options, payload.defaults.stn_path);
    fillSelect($("yoloSelect"), payload.yolo_options, payload.defaults.yolo_path);
    $("instructionList").innerHTML = payload.instructions.map((item) => `<li>${item}</li>`).join("");
}

export async function loadModels() {
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
