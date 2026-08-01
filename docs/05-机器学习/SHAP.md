# SHAP —— 模型可解释性分析设计

## 1. 模块概述

SHAP（SHapley Additive exPlanations）模块用于解释机器学习模型的预测结果，是本项目机器学习系统的重要组成部分。

虽然 XGBoost 能够获得较高的预测精度，但其内部决策过程较难直接理解。对于教育场景而言，仅给出"高风险"或"低风险"的预测结果并不足够，教师更希望了解导致该结果的原因，例如学生是由于疲劳、持续低头还是课堂参与度下降而被判定为风险对象。

因此，本项目引入 SHAP 对模型进行可解释性分析，对每一次预测结果给出特征贡献度，帮助教师理解模型决策依据，同时为模型优化和科研分析提供支持。

---

# 2. 模块定位

SHAP 位于模型预测之后。

```text
Camera
    │
    ▼
CV Modules
    │
    ▼
Feature Engineering
    │
    ▼
XGBoost
    │
    ├────────► Risk Probability
    │
    ▼
SHAP Explainability
    │
    ├────────► Dashboard
    ├────────► Report
    └────────► Research Analysis
```

SHAP 不参与模型训练，仅负责解释模型预测结果。

---

# 3. SHAP 原理

SHAP 基于博弈论中的 Shapley Value。

核心思想：

假设每一个输入特征都是一名"参与者"，模型预测结果就是所有参与者共同完成的一项任务。

SHAP 会计算：

> 每一个特征对最终预测结果贡献了多少。

例如：

学生预测风险：

```text
Risk Probability

=

0.87
```

SHAP 可以进一步说明：

```text
PERCLOS

+0.26

Fatigue Score

+0.21

Head Down

+0.18

Attention Score

-0.09

Blink Rate

-0.04
```

教师即可知道：

本次预测主要受到疲劳和持续低头影响，而课堂参与度略微降低了风险评分。

---

# 4. 输入数据

SHAP 的输入包括：

| 输入 | 来源 |
|------|------|
| Feature Vector | Feature Engineering |
| Trained Model | XGBoost |
| Prediction Result | XGBoost |

其中 Feature Vector 与模型预测时保持一致。

---

# 5. 输出结果

SHAP 输出每一个特征的贡献值。

例如：

| Feature | SHAP Value |
|----------|------------|
| perclos | +0.26 |
| fatigue_score | +0.21 |
| head_down_duration | +0.18 |
| attention_score | -0.09 |
| blink_rate | -0.04 |

正值表示增加风险。

负值表示降低风险。

SHAP Value 的绝对值越大，说明该特征影响越大。

---

# 6. 全局解释（Global Explanation）

全局解释用于分析整个模型。

主要回答：

- 哪些特征最重要？
- 模型主要依据哪些因素进行预测？
- 哪些特征长期影响最大？

建议输出：

```text
Feature Importance Ranking
```

例如：

| 排名 | 特征 |
|------|------|
| 1 | fatigue_score |
| 2 | perclos |
| 3 | attention_score |
| 4 | head_down_duration |
| 5 | pitch_mean |
| 6 | blink_rate |

全局解释适用于论文实验分析。

---

# 7. 局部解释（Local Explanation）

局部解释针对单个学生、单次预测进行分析。

例如：

学生 A：

```text
Risk Probability

=

0.91
```

对应 SHAP：

| Feature | Contribution |
|----------|--------------|
| Fatigue Score | +0.28 |
| PERCLOS | +0.24 |
| Pitch | +0.15 |
| Attention Score | -0.05 |

教师可以直接了解：

本次高风险主要由疲劳状态和持续低头导致。

---

# 8. 可视化分析

SHAP 支持丰富的可视化方式。

建议输出：

### 1）Summary Plot

展示所有特征的重要性排序。

---

### 2）Bar Plot

显示平均 SHAP 值。

便于快速了解模型重点关注哪些特征。

---

### 3）Waterfall Plot

展示单次预测过程中：

每一个特征如何一步一步影响最终预测结果。

---

### 4）Force Plot

动态展示预测形成过程。

适合网页展示。

---

### 5）Dependence Plot

分析：

某一个特征变化时，

预测结果如何变化。

例如：

```text
PERCLOS

↑

↓

Risk Probability

↑
```

---

# 9. Dashboard 展示

建议在 Flask Dashboard 中增加：

```text
学生：

A001

预测：

高风险

Risk：

91%

主要原因：

① 疲劳程度高

② 持续低头

③ PERCLOS增加

建议：

提醒学生适当休息
```

教师无需阅读复杂图表即可理解预测依据。

---

# 10. 数据存储

建议保存：

```text
PredictionLog

├── Student ID
├── Prediction Time
├── Risk Probability
├── Top Feature
├── SHAP Summary
└── Model Version
```

重要预测结果可长期保存，用于科研分析。

---

# 11. 与其他模块关系

```text
Feature Engineering
        │
        ▼
     XGBoost
        │
        ▼
       SHAP
        │
 ┌──────┼────────────┐
 ▼      ▼            ▼
Dashboard Report Research
```

SHAP 与 XGBoost 紧密配合，共同构成机器学习分析模块。

---

# 12. 性能要求

| 指标 | 要求 |
|------|------|
| 单次解释时间 | ≤1 秒 |
| 支持实时解释 | 是（单样本） |
| 支持批量分析 | 是 |
| 输出图表 | PNG、SVG、HTML |

实时预测阶段可仅计算当前样本 SHAP 值，离线分析时生成完整解释报告。

---

# 13. 工程实现建议

建议采用：

```python
import shap

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)
```

对于 XGBoost，优先使用 `TreeExplainer`，具有更高的计算效率和更准确的解释结果。

---

# 14. 后续优化

当前版本主要面向 XGBoost 提供可解释性分析。

后续版本将逐步扩展：

1. 支持 LSTM 等深度学习模型的 SHAP 分析；
2. 自动生成学生个人学习状态分析报告；
3. 支持班级整体风险特征统计；
4. 建立 SHAP 历史变化趋势分析，观察学生风险来源随时间的变化；
5. 与 Dashboard 深度集成，支持教师交互式查看模型解释结果。

SHAP 模块不仅提高了模型的透明度和可信度，也能够帮助教师理解学生课堂状态变化的主要原因，为教学干预提供科学依据，同时满足科研项目对于模型可解释性的要求。