export const state = {
    currentFile: null,
    currentFilePreviewUrl: null,
    batchUploadDir: "",
    activeTab: "singleTab",
    realtime: {
        stream: null,
        active: false,
        inflight: false,
        selectedDeviceId: "",
        devices: [],
        lastValidReading: "-",
        lastValidRatio: "-",
        lastValidReadingLabel: "读数结果",
        lastValidTime: "--:--:--",
        lastElapsedMs: null
    },
    resultNaturalWidth: 0,
    resultNaturalHeight: 0,
    toastTimer: null,
    batchRows: [],
    batchPage: 1,
    pageSize: 10,
    batchRowsShown: false,
    batchPackagingNoticeShown: false
};

export function $(id) {
    return document.getElementById(id);
}
