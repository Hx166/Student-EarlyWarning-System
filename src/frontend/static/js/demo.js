/* 阶段 G：DAiSEE 片段回放页面。 */

const playbackChart = echarts.init(document.getElementById("playback-chart"));
const levelNames = { level_0: "很低", level_1: "较低", level_2: "较高", level_3: "很高" };
let clips = [];

playbackChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["参与度", "疲劳度"] },
    xAxis: { type: "category", data: [], name: "秒" },
    yAxis: { type: "value", min: 0, max: 100 },
    series: [
        { name: "参与度", type: "line", smooth: true, data: [] },
        { name: "疲劳度", type: "line", smooth: true, data: [] },
    ],
});

function fmt(value, digits = 1) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";
}

function setStatus(text, isError = false) {
    const element = document.getElementById("playback-status");
    element.textContent = text;
    element.classList.toggle("error", isError);
}

function renderProbabilities(probabilities) {
    document.getElementById("playback-probabilities").innerHTML =
        Object.entries(levelNames).map(([level, name]) => {
            const value = Number(probabilities?.[level] || 0);
            const width = Math.max(0, Math.min(100, value * 100));
            return `<div class="probability-row">
                <span>${name}</span>
                <div class="probability-track"><div class="probability-bar" style="width:${width}%"></div></div>
                <span>${fmt(value * 100)}%</span>
            </div>`;
        }).join("");
}

function renderPlayback(data) {
    const probabilityValues = Object.values(data.probabilities || {})
        .reduce((sum, value) => sum + Number(value), 0);
    document.getElementById("true-level").textContent =
        levelNames[data.true_level] || data.true_level;
    document.getElementById("pred-level").textContent =
        levelNames[data.pred_level] || data.pred_level;
    document.getElementById("pred-confidence").textContent =
        `${fmt(Number(data.confidence) * 100)}%`;
    document.getElementById("pred-elapsed").textContent =
        `${fmt(data.elapsed_seconds, 2)} 秒`;
    renderProbabilities(data.probabilities || {});

    const series = data.series || [];
    playbackChart.setOption({
        xAxis: { data: series.map(item => item.second) },
        series: [
            { name: "参与度", data: series.map(item => item.attention) },
            { name: "疲劳度", data: series.map(item => item.fatigue) },
        ],
    });
    document.getElementById("playback-result").hidden = false;
    setStatus(`回放完成，概率值合计 ${fmt(probabilityValues * 100)}%。`);
}

function renderVideo(clip) {
    const panel = document.getElementById("video-panel");
    const video = document.getElementById("playback-video");
    const status = document.getElementById("video-status");
    if (!clip?.playable) {
        panel.hidden = true;
        video.removeAttribute("src");
        return;
    }
    panel.hidden = false;
    status.textContent = "";
    video.onerror = () => {
        video.hidden = true;
        status.textContent = "浏览器无法播放该 MP4，逐秒曲线仍可正常使用。";
    };
    video.hidden = false;
    video.src = `/api/demo/video/${encodeURIComponent(clip.clip_id)}`;
    video.load();
}

async function loadClips() {
    try {
        const response = await fetch("/api/demo/clips");
        const payload = await response.json();
        if (payload.code !== 0) throw new Error(payload.message || "failed");
        clips = payload.data || [];
        const select = document.getElementById("clip-select");
        select.innerHTML = clips.map((clip, index) => `
            <option value="${clip.clip_id}" data-index="${index}">
                ${clip.clip_id} · 等级 ${clip.engagement} · ${clip.frames} 帧
            </option>`).join("");
        select.disabled = clips.length === 0;
        document.getElementById("run-playback").disabled = clips.length === 0;
        setStatus(
            clips.length ? `已加载 ${clips.length} 条演示片段。` : "演示数据不可用",
            !clips.length
        );
    } catch (error) {
        setStatus("演示数据不可用", true);
    }
}

async function runPlayback() {
    const select = document.getElementById("clip-select");
    const clip = clips[Number(select.selectedOptions[0]?.dataset.index)];
    if (!clip) {
        setStatus("演示数据不可用", true);
        return;
    }
    const button = document.getElementById("run-playback");
    button.disabled = true;
    setStatus("正在完整处理片段，请稍候...");
    try {
        const response = await fetch(
            `/api/demo/run/${encodeURIComponent(clip.clip_id)}`
        );
        const payload = await response.json();
        if (payload.code !== 0) throw new Error(payload.message || "failed");
        renderPlayback(payload.data);
        renderVideo(clip);
    } catch (error) {
        setStatus("演示数据不可用", true);
    } finally {
        button.disabled = false;
    }
}

document.getElementById("run-playback").addEventListener("click", runPlayback);
loadClips();
