/* 阶段 G：模型说明页面。 */

const modelData = JSON.parse(document.getElementById("model-data").textContent);

function escapeText(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
}

function metricRow(split, label, values) {
    return `<tr>
        <td>${split}</td><td>${label}</td>
        <td>${Number(values.accuracy).toFixed(6)}</td>
        <td>${Number(values.macro_f1).toFixed(6)}</td>
        <td>${Number(values.weighted_f1).toFixed(6)}</td>
        <td>${values.roc_auc_ovr == null ? "不适用" : Number(values.roc_auc_ovr).toFixed(6)}</td>
    </tr>`;
}

function renderModel() {
    const root = document.getElementById("model-overview");
    if (!modelData.available) {
        root.textContent = "指标未生成";
        ["metric-tables", "per-class-table", "confusion-matrix", "limitations"]
            .forEach(id => { document.getElementById(id).textContent = "指标未生成"; });
        return;
    }
    const { metadata, metrics } = modelData;
    const mapping = metadata.label_mapping || {};
    const features = metadata.feature_names || [];
    root.innerHTML = `
        <div class="model-overview-grid">
            <div><label>模型版本</label><strong>${escapeText(metadata.version)}</strong></div>
            <div><label>数据版本</label><strong>${escapeText(metadata.dataset)}</strong></div>
            <div><label>训练样本</label><strong>${escapeText(metadata.rows)}</strong></div>
        </div>
        <div class="chip-list">${features.map(f => `<span>${escapeText(f)}</span>`).join("")}</div>
        <div class="chip-list">${Object.entries(mapping).map(([id, name]) =>
            `<span>${escapeText(id)} → ${escapeText(name)}</span>`).join("")}</div>`;

    const rows = [];
    for (const split of ["validation", "test"]) {
        const title = split === "test" ? "Test" : "Validation";
        rows.push(metricRow(title, "多数类基线", metrics[split].baseline));
        rows.push(metricRow(title, "engagement-v1", metrics[split].model));
    }
    document.getElementById("metric-tables").innerHTML = `<table>
        <thead><tr><th>数据</th><th>版本</th><th>Accuracy</th><th>Macro-F1</th><th>Weighted-F1</th><th>ROC-AUC</th></tr></thead>
        <tbody>${rows.join("")}</tbody></table>`;

    const classes = metrics.test.model.per_class || [];
    document.getElementById("per-class-table").innerHTML = `<table>
        <thead><tr><th>类别</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>${classes.map(item => `<tr><td>${escapeText(item.label)}</td>
            <td>${Number(item.precision).toFixed(4)}</td>
            <td>${Number(item.recall).toFixed(4)}</td>
            <td>${Number(item.f1).toFixed(4)}</td>
            <td>${item.support}</td></tr>`).join("")}</tbody></table>`;

    const matrix = metrics.test.model.confusion_matrix || [];
    document.getElementById("confusion-matrix").innerHTML = `<table>
        <thead><tr><th>实际</th><th>预测 L0</th><th>预测 L1</th><th>预测 L2</th><th>预测 L3</th></tr></thead>
        <tbody>${matrix.map((row, index) =>
            `<tr><td>L${index}</td>${row.map(value => `<td>${value}</td>`).join("")}</tr>`
        ).join("")}</tbody>
    </table>`;

    const full = metrics.test.model;
    const ablation = metrics.ablation.metrics.Test;
    document.getElementById("limitations").innerHTML = `
        <p><strong>消融：</strong>Test accuracy ${Number(ablation.accuracy).toFixed(6)}，
        macro-F1 ${Number(ablation.macro_f1).toFixed(6)}；完整 11 特征分别为
        ${Number(full.accuracy).toFixed(6)}、${Number(full.macro_f1).toFixed(6)}。</p>
        <p><strong>已知局限：</strong>该模型 accuracy 与多数类基线持平；Level 0 Test 仅 4 条且 F1 为 0；
        概率条是模型辅助信息，不代表校准后的现实置信度。</p>`;
}

renderModel();
