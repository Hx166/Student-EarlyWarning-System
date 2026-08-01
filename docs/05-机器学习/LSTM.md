# LSTM —— 学生课堂状态时序预测模型设计

## 1. 模块概述

长短期记忆网络（Long Short-Term Memory，LSTM）是本项目 V1.0 阶段计划引入的深度学习模型，主要用于学习学生课堂行为的时间变化规律，对学生未来一段时间内的课堂状态和学业风险进行预测。

与 XGBoost 基于统计特征进行分类不同，LSTM 能够处理连续时间序列数据，学习学生行为随时间变化的趋势。例如，一名学生课堂参与度持续下降、疲劳程度不断增加，这种变化趋势往往比某一时刻的单次状态更具有预测价值。

因此，本项目采用"XGBoost + LSTM"两阶段模型架构：

- XGBoost：当前状态分类；
- LSTM：未来状态趋势预测。

---

# 2. 模块定位

LSTM 位于机器学习模块的高级分析层。

```text
Camera
    │
    ▼
CV Modules
    │
    ▼
Feature Engineering
    │
    ├────────► XGBoost（当前状态预测）
    │
    └────────► LSTM（未来趋势预测）
                     │
                     ▼
              Risk Forecast
                     │
                     ▼
             Warning System
```

LSTM 不直接处理图像，而是处理 Feature Engineering 输出的连续特征序列。

---

# 3. 为什么选择 LSTM

课堂行为具有明显的时间连续性。

例如：

- 学生不会在一秒钟内突然进入疲劳状态；
- 注意力下降通常是一个逐渐发展的过程；
- 头部姿态、眨眼频率、课堂参与度等指标都会随时间变化。

传统机器学习模型通常将每个时间窗口视为独立样本，无法充分利用这些时间信息。

LSTM 具有记忆能力，能够学习长期依赖关系，更适合分析课堂行为序列。

---

# 4. 输入数据

LSTM 输入为连续多个时间窗口组成的特征序列。

单个时间窗口包含：

| 特征 | 来源 |
|------|------|
| attention_score | Engagement |
| fatigue_score | Fatigue |
| perclos | EyeEAR |
| blink_rate | EyeEAR |
| pitch_mean | HeadPose |
| yaw_mean | HeadPose |
| roll_mean | HeadPose |
| head_down_duration | HeadPose |
| face_lost_duration | FaceDetection |

每个窗口由 Feature Engineering 模块生成。

---

# 5. 时间序列构建

假设：

每秒生成一个 Feature Vector。

连续采集：

```text
60 秒
```

则形成：

```text
X1

↓

X2

↓

X3

↓

...

↓

X60
```

最终形成：

```text
Sequence Length

=

60
```

输入 LSTM。

每一个 Feature Vector 包含多个特征维度。

例如：

```text
(60 × 12)
```

表示：

60 个时间步，每个时间步包含 12 个特征。

---

# 6. 输出目标

LSTM 可支持以下预测任务：

### 1）课堂状态预测

预测未来一段时间课堂状态。

输出：

| Label | 含义 |
|--------|------|
| 0 | 正常 |
| 1 | 注意力下降 |
| 2 | 疲劳 |
| 3 | 高风险 |

---

### 2）风险概率预测

输出：

```text
Risk Probability

0~1
```

例如：

```text
未来5分钟

Risk

=

0.84
```

---

### 3）趋势预测

预测：

```text
Attention Score

↓

Fatigue Score

↑
```

帮助教师提前发现学生状态变化。

---

# 7. 网络结构

建议采用单层或双层 LSTM。

结构如下：

```text
Input Sequence
        │
        ▼
      LSTM
        │
        ▼
     Dropout
        │
        ▼
   Fully Connected
        │
        ▼
    Softmax / Sigmoid
```

第一阶段不采用复杂网络结构，保证模型易于训练和部署。

---

# 8. 推荐参数

建议初始参数如下：

| 参数 | 建议值 |
|------|---------|
| Sequence Length | 60 |
| Hidden Size | 64 |
| Num Layers | 2 |
| Dropout | 0.2 |
| Batch Size | 32 |
| Epoch | 100 |
| Learning Rate | 0.001 |
| Optimizer | Adam |

训练过程中可根据实验结果进行调整。

---

# 9. 损失函数

分类任务建议采用：

```text
Cross Entropy Loss
```

若后续改为风险概率预测，可采用：

```text
Binary Cross Entropy

或

Mean Squared Error
```

根据任务类型选择合适的损失函数。

---

# 10. 模型训练流程

```mermaid
flowchart TD

Feature Sequence
        --> Train

Train
        --> Validation

Validation
        --> Test

Test
        --> Save Model

Save Model
        --> Prediction
```

模型训练完成后保存最佳权重。

---

# 11. 模型保存

建议保存：

```text
models/

best/

lstm_model.pth
```

同时保存：

```text
training_config.yaml

feature_names.json

label_mapping.json
```

确保模型能够正确加载和推理。

---

# 12. 在线预测流程

课堂运行过程中：

```text
每秒

↓

生成 Feature

↓

加入时间窗口

↓

长度达到60

↓

LSTM预测

↓

更新未来风险
```

系统采用滑动窗口方式持续更新预测结果。

---

# 13. 与 XGBoost 的关系

本项目采用分阶段模型架构。

| 模型 | 功能 |
|------|------|
| XGBoost | 当前课堂状态分类 |
| LSTM | 长期行为趋势预测 |

两者并非相互替代，而是互为补充。

教师可同时查看：

- 当前课堂状态；
- 未来风险趋势。

---

# 14. 模块输出

统一输出：

```text
LSTMResult

├── prediction
├── probability
├── trend
├── prediction_time
└── confidence
```

其中：

| 字段 | 描述 |
|------|------|
| prediction | 预测类别 |
| probability | 风险概率 |
| trend | 状态变化趋势 |
| prediction_time | 预测时间 |
| confidence | 模型置信度 |

---

# 15. 后续优化

当前版本将 LSTM 定位为项目 V1.0 的高级分析模块，主要用于时间序列预测。

未来可进一步扩展：

1. 引入双向 LSTM（Bi-LSTM），学习前后时间依赖关系；
2. 尝试 GRU、Transformer 等新型时序模型，对比不同模型性能；
3. 融合成绩、考勤、作业完成情况等多模态数据，提高长期预测能力；
4. 结合注意力机制（Attention Mechanism），增强模型对关键时间片段的关注；
5. 与 SHAP 等可解释性方法结合，分析时间序列中哪些行为变化对预测结果影响最大。

LSTM 模块将作为本项目后期版本的重要升级方向，与 XGBoost 共同构建"当前状态识别 + 未来趋势预测"的双层智能分析体系，为学生课堂学习状态监测和学业风险预警提供更加全面、准确的技术支持。