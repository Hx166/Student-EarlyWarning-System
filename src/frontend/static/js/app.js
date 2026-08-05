/* src/frontend/static/js/app.js —— Dashboard 前端逻辑 */

const attentionChart = echarts.init(document.getElementById("attention-chart"));
const fatigueChart = echarts.init(document.getElementById("fatigue-chart"));

function initChart(chart, title) {
    chart.setOption({
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: [] },
        yAxis: { type: "value", min: 0, max: 100 },
        series: [{ name: title, type: "line", smooth: true, data: [] }],
    });
}

initChart(attentionChart, "\u4e13\u6ce8\u5ea6");
initChart(fatigueChart, "\u75b2\u52b3\u5ea6");

// \u5c0f\u6570\u70b9\u8f93\u51fa
function fmt(v, digits) {
    if (v === null || v === undefined || isNaN(v)) return "--";
    return Number(v).toFixed(digits);
}

// FPS \u4f30\u7b97\uff1a\u57fa\u4e8e\u6700\u8fd1\u4e24\u6761\u8bb0\u5f55\u7684\u65f6\u95f4\u95f4\u9694
let lastFetchTime = null;
function estimateFps(nowTs) {
    if (lastFetchTime) {
        const dt = (nowTs - lastFetchTime) / 1000;
        if (dt > 0) return (1 / dt).toFixed(1);
    }
    return "--";
}

// \u66f4\u65b0\u5b9e\u65f6\u6307\u6807\u5361
function renderMetrics(latest) {
    if (!latest) return;
    document.getElementById("m-ear").textContent = fmt(latest.avg_ear, 3);
    document.getElementById("m-perclos").textContent = fmt(latest.perclos, 3);
    document.getElementById("m-pitch").textContent = fmt(latest.avg_pitch, 1);
    document.getElementById("m-yaw").textContent = fmt(latest.avg_yaw, 1);
    document.getElementById("m-roll").textContent = fmt(latest.avg_roll, 1);
    document.getElementById("m-fatigue").textContent = fmt(latest.fatigue_score, 1);
    document.getElementById("m-engagement").textContent = fmt(latest.attention_score, 1);
}

// \u66f4\u65b0\u9884\u8b66\u5217\u8868
function renderWarnings(warnings) {
    const body = document.querySelector("#warning-table tbody");
    if (!warnings || warnings.length === 0) {
        body.innerHTML = "<tr><td colspan='4'>\u6682\u65e0\u9884\u8b66</td></tr>";
        return;
    }
    body.innerHTML = warnings.slice(0, 20).map(w => `
        <tr>
            <td>${(w.create_time || "").slice(5, 19)}</td>
            <td>${w.warning_type || "--"}</td>
            <td>${w.warning_level || "--"}</td>
            <td>${w.warning_reason || "--"}</td>
        </tr>`).join("");
}

// \u66f4\u65b0\u6570\u636e\u5e93\u6700\u65b0\u8bb0\u5f55\u8868
function renderStateTable(records) {
    const body = document.querySelector("#state-table tbody");
    if (!records || records.length === 0) {
        body.innerHTML = "<tr><td colspan='8'>\u6682\u65e0\u6570\u636e</td></tr>";
        return;
    }
    body.innerHTML = records.slice(0, 15).map(r => `
        <tr>
            <td>${(r.record_time || "").slice(5, 19)}</td>
            <td>${fmt(r.avg_ear, 3)}</td>
            <td>${fmt(r.perclos, 3)}</td>
            <td>${fmt(r.avg_pitch, 1)}</td>
            <td>${fmt(r.avg_yaw, 1)}</td>
            <td>${fmt(r.avg_roll, 1)}</td>
            <td>${fmt(r.attention_score, 1)}</td>
            <td>${fmt(r.fatigue_score, 1)}</td>
        </tr>`).join("");
}

async function fetchJson(url) {
    const res = await fetch(url);
    return res.json();
}

// \u4e3b\u5237\u65b0\u6d41\u7a0b
async function refresh() {
    try {
        const nowTs = Date.now();
        const [statePayload, warningPayload] = await Promise.all([
            fetchJson("/api/state?limit=50"),
            fetchJson("/api/warnings?limit=20"),
        ]);

        if (statePayload.code === 0) {
            const records = statePayload.data || [];
            const times = records.map(r => (r.record_time || "").slice(5, 19));
            const attention = records.map(r => r.attention_score ?? 0);
            const fatigue = records.map(r => r.fatigue_score ?? 0);

            attentionChart.setOption({ xAxis: { data: times }, series: [{ data: attention }] });
            fatigueChart.setOption({ xAxis: { data: times }, series: [{ data: fatigue }] });

            renderStateTable(records);
            renderMetrics(records[0] || null);
            document.getElementById("m-fps").textContent = estimateFps(nowTs);
            lastFetchTime = nowTs;
        }

        if (warningPayload.code === 0) {
            renderWarnings(warningPayload.data || []);
        }

        document.getElementById("health-status").className = "badge ok";
        document.getElementById("health-status").textContent = "\u5728\u7ebf";
    } catch (e) {
        document.getElementById("health-status").className = "badge err";
        document.getElementById("health-status").textContent = "\u79bb\u7ebf";
    }
}

// \u6bcf 2 \u79d2\u5237\u65b0\u4e00\u6b21
refresh();
setInterval(refresh, 2000);
