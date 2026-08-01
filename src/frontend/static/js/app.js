/* src/frontend/static/js/app.js —— Dashboard 前端逻辑 */

// 初始化图表
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

initChart(attentionChart, "专注度");
initChart(fatigueChart, "疲劳度");

// 拉取状态数据并刷新图表
async function refresh() {
    try {
        const res = await fetch("/api/state?limit=50");
        const payload = await res.json();
        if (payload.code !== 0) return;

        const records = payload.data;
        const times = records.map(r => (r.record_time || "").slice(5, 19));
        const attention = records.map(r => r.attention_score ?? 0);
        const fatigue = records.map(r => r.fatigue_score ?? 0);

        attentionChart.setOption({
            xAxis: { data: times },
            series: [{ data: attention }],
        });
        fatigueChart.setOption({
            xAxis: { data: times },
            series: [{ data: fatigue }],
        });

        // 更新最新状态表
        const body = document.querySelector("#state-table tbody");
        if (records.length === 0) {
            body.innerHTML = "<tr><td colspan='4'>暂无数据</td></tr>";
        } else {
            body.innerHTML = records.slice(0, 10).map(r => `
                <tr>
                    <td>${(r.record_time || "").slice(5, 19)}</td>
                    <td>${(r.attention_score ?? 0).toFixed(1)}</td>
                    <td>${(r.fatigue_score ?? 0).toFixed(1)}</td>
                    <td>${(r.perclos ?? 0).toFixed(2)}</td>
                </tr>`).join("");
        }

        document.getElementById("health-status").className = "badge ok";
        document.getElementById("health-status").textContent = "在线";
    } catch (e) {
        document.getElementById("health-status").className = "badge err";
        document.getElementById("health-status").textContent = "离线";
    }
}

// 每 2 秒刷新一次
refresh();
setInterval(refresh, 2000);
