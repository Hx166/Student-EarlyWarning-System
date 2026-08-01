# FeatureEngineering —— 特征工程模块设计

## 1. 模块概述

特征工程（Feature Engineering）是连接计算机视觉模块与机器学习模型的重要桥梁，其主要任务是将摄像头采集的原始视觉数据转换为适合机器学习模型训练和预测的结构化特征。

本项目中，Face Detection、MediaPipe、HeadPose、EyeEAR、Fatigue、Engagement 等模块会持续产生大量时序数据，例如每帧 EAR、头部姿态角度、疲劳评分等。这些数据直接输入机器学习模型容易受到噪声、抖动及瞬时异常值影响，因此需要通过特征工程完成数据清洗、时间窗口统计、归一化及特征构造。

Feature Engineering 模块最终输出统一的 Feature Vector，为 XGBoost、LSTM 等模型提供输入。

---

# 2. 模块定位

Feature Engineering 位于整个系统的数据处理中间层。

```text
Camera
    │
    ▼
CV模块
(FaceDetection、MediaPipe、
HeadPose、EyeEAR、Fatigue、
Engagement)
    │
    ▼
Database(FrameRecord)
    │
    ▼
Feature Engineering
    │
    ├────────► XGBoost
    ├────────► LSTM
    ├────────► SHAP
    └────────► Dashboard
```

本模块不负责模型训练，仅负责生成机器学习特征。

---

# 3. 特征来源

第一阶段所有特征均来自视觉分析模块。

| 来源模块 | 输出特征 |
|-----------|-----------|
| FaceDetection | 人脸检测状态、人脸数量 |
| HeadPose | Pitch、Yaw、Roll |
| EyeEAR | EAR、Blink Rate、PERCLOS |
| Fatigue | Fatigue Score |
| Engagement | Attention Score |

后续版本可扩展：

- 成绩数据
- 考勤数据
- 作业数据
- 学习平台数据
- 问卷调查数据

当前版本重点完成视觉特征采集。

---

# 4. 原始特征定义

系统统一规定如下基础特征。

| 特征名称 | 类型 | 单位 | 来源 |
|-----------|------|------|------|
| avg_ear | Float | Ratio | EyeEAR |
| left_ear | Float | Ratio | EyeEAR |
| right_ear | Float | Ratio | EyeEAR |
| blink_rate | Float | 次/分钟 | EyeEAR |
| perclos | Float | % | EyeEAR |
| avg_pitch | Float | ° | HeadPose |
| avg_yaw | Float | ° | HeadPose |
| avg_roll | Float | ° | HeadPose |
| fatigue_score | Float | 0~100 | Fatigue |
| attention_score | Float | 0~100 | Engagement |
| face_detected | Boolean | - | FaceDetection |

所有模块必须遵循统一命名规范，避免同一特征出现多个名称。

---

# 5. 时间窗口设计

课堂行为具有连续性，因此采用滑动时间窗口统计，而非单帧分析。

建议配置：

| 参数 | 建议值 |
|------|---------|
| 窗口长度 | 30 秒 |
| 更新周期 | 1 秒 |
| 帧率 | 30 FPS |

每个窗口包含约：

```text
30 × 30

=

900 帧
```

系统每秒重新计算一次窗口统计值。

---

# 6. 统计特征构造

针对每个原始特征，计算统计指标。

例如：

EAR：

计算：

- 平均值
- 最大值
- 最小值
- 标准差
- 中位数

形成：

```text
ear_mean

ear_max

ear_min

ear_std

ear_median
```

Pitch：

计算：

```text
pitch_mean

pitch_std

pitch_max

pitch_min
```

Yaw：

计算：

```text
yaw_mean

yaw_std
```

Attention：

计算：

```text
attention_mean

attention_std
```

所有统计均基于滑动窗口完成。

---

# 7. 行为统计特征

除连续变量外，还统计课堂行为事件。

| 特征 | 描述 |
|------|------|
| blink_count | 眨眼次数 |
| blink_rate | 每分钟眨眼次数 |
| head_down_count | 低头次数 |
| head_down_duration | 累计低头时间 |
| face_lost_count | 离开画面次数 |
| face_lost_duration | 离开画面时间 |

行为事件由 BehaviorRecord 自动统计生成。

---

# 8. 特征归一化

由于不同特征量纲不同，需要统一缩放。

例如：

| 特征 | 原始范围 |
|------|-----------|
| Pitch | -90~90 |
| EAR | 0~1 |
| Fatigue | 0~100 |
| Attention | 0~100 |

建议采用 Min-Max 归一化：

```text
x'

=

(x-min)

/

(max-min)
```

归一化后：

```text
所有特征

↓

0~1
```

便于模型训练。

---

# 9. 缺失值处理

课堂环境中可能出现：

- 人脸丢失
- 摄像头遮挡
- Landmark 检测失败

处理策略如下：

| 情况 | 处理方式 |
|------|-----------|
| 少量缺失 | 前向填充 |
| 连续缺失 | 插值 |
| 长时间无人 | 删除窗口 |
| 摄像头异常 | 不参与训练 |

保证训练数据质量。

---

# 10. 异常值处理

为了避免极端值影响模型训练，系统采用异常值过滤。

建议：

连续变量：

使用：

```text
IQR

或

3σ
```

检测异常值。

例如：

Pitch：

```text
100°

↓

异常

↓

删除
```

EAR：

```text
1.8

↓

异常

↓

删除
```

---

# 11. Feature Vector

经过统计后形成统一特征向量。

例如：

```text
Feature Vector

{

ear_mean,

ear_std,

blink_rate,

perclos,

pitch_mean,

yaw_mean,

roll_mean,

fatigue_score,

attention_score,

head_down_duration,

face_lost_duration

}
```

每个时间窗口对应一个 Feature Vector。

该向量将直接输入 XGBoost。

---

# 12. 标签（Label）设计

监督学习需要标签。

第一阶段建议采用人工标注。

例如：

| Label | 含义 |
|--------|------|
| 0 | 正常课堂状态 |
| 1 | 注意力下降 |
| 2 | 疲劳状态 |
| 3 | 高风险状态 |

标签来源：

- 教师观察
- 实验录像
- 人工审核

后续可结合成绩数据自动生成标签。

---

# 13. 数据集划分

建议：

| 数据集 | 比例 |
|----------|------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

划分原则：

- 同一学生数据不得同时出现在训练集和测试集中；
- 保持各类别样本比例基本一致；
- 随机划分前进行数据去重和异常值清洗。

---

# 14. 输出文件格式

建议生成 CSV 文件作为训练数据。

示例：

| ear_mean | blink_rate | perclos | pitch_mean | attention_score | label |
|-----------|------------|----------|-------------|-----------------|-------|
| 0.29 | 18 | 0.12 | 8.5 | 86 | 0 |
| 0.18 | 35 | 0.41 | 27.3 | 42 | 2 |

文件保存位置：

```text
datasets/features/

student_feature.csv
```

供 XGBoost、LSTM 等模型直接读取。

---

# 15. 与其他模块关系

Feature Engineering 是机器学习模块的统一入口。

```text
CV模块
      │
      ▼
Feature Engineering
      │
      ├── XGBoost
      ├── SHAP
      ├── LSTM
      ├── Dashboard
      └── Warning System
```

所有机器学习模型必须使用 Feature Engineering 输出的标准化特征，不允许直接读取原始视觉数据。

---

# 16. 后续优化

当前版本主要基于统计特征构建 Feature Vector，能够满足 XGBoost 等传统机器学习模型的输入需求。

后续版本将逐步扩展特征工程能力：

1. 增加时间序列特征（滑动平均、变化率、趋势项）；
2. 构建组合特征，如 PERCLOS × Fatigue Score、Pitch × Attention Score；
3. 引入特征选择算法（相关系数分析、递归特征消除、XGBoost Feature Importance）降低冗余特征；
4. 支持多模态特征融合，将视觉数据与成绩、考勤、学习行为等信息统一建模；
5. 建立自动化特征工程流程，为模型训练、在线预测及模型更新提供一致的数据输入。

Feature Engineering 模块是整个机器学习系统的基础，其输出的 Feature Vector 将作为 XGBoost、LSTM 等模型的标准输入，为学生课堂状态分析与学业风险预警提供高质量的数据支撑。