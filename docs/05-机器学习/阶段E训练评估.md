# 阶段 E 训练评估报告

> 项目：Student Early Warning System
>
> 阶段：阶段 E（DAiSEE Engagement 四分类训练与评估）
>
> 完成日期：2026-10-31
>
> 数据版本：`daisee@2026-10`
>
> 训练框架：XGBoost 3.2.0 / scikit-learn

---

## 1. 数据与参数

输入为 `datasets/features/daisee_features.csv`，校验结果：

| 项 | 数量 |
|---|---:|
| 总行数 | 8570 |
| 唯一 clip_id | 8570 |
| Train | 5357 |
| Validation | 1429 |
| Test | 1784 |
| Engagement 0 | 61 |
| Engagement 1 | 440 |
| Engagement 2 | 4312 |
| Engagement 3 | 3757 |

11 个输入特征无 NaN，列顺序严格使用 `FEATURE_VECTOR_COLUMNS`。Train、Validation、Test 按官方划分使用，三集被试无交叉。

训练参数：

```yaml
objective: multi:softprob
num_class: 4
max_depth: 6
learning_rate: 0.1
n_estimators: 100
subsample: 0.8
colsample_bytree: 0.8
gamma: 0
min_child_weight: 1
eval_metric: mlogloss
n_jobs: 1
random_state: 42
```

随机种子：42。没有做超参数搜索。主模型只使用 Train 拟合，Validation 和 Test 分别按最终口径评估。

Engagement 使用独立映射：

| ID | 名称 |
|---:|---|
| 0 | level_0 |
| 1 | level_1 |
| 2 | level_2 |
| 3 | level_3 |

该映射与风险模型的 `normal / attention_down / fatigue / high_risk` 完全独立。

## 2. Validation / Test 指标

### Test

| 指标 | 多数类基线 | 主模型 |
|---|---:|---:|
| Accuracy | 0.494395 | 0.494395 |
| Macro-F1 | 0.165416 | 0.258088 |
| Weighted-F1 | 0.327124 | 0.482663 |
| ROC-AUC（OVR macro） | 不适用 | 0.635055 |

主模型 Test accuracy 与多数类基线完全持平，**没有高于基线**。但宏 F1 和加权 F1 明显提高，说明模型不再只预测单一多数类，同时 Level 0、Level 1 的识别仍然很弱。

### Validation

| 指标 | 多数类基线 | 主模型 |
|---|---:|---:|
| Accuracy | 0.568929 | 0.478656 |
| Macro-F1 | 0.181311 | 0.263919 |
| Weighted-F1 | 0.412613 | 0.458630 |
| ROC-AUC（OVR macro） | 不适用 | 0.634498 |

Validation 上主模型 accuracy 低于基线，但两个 F1 指标更高，表现出相同的类别覆盖改善与整体准确率损失。

## 3. Test 每类指标

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| level_0 | 0.000000 | 0.000000 | 0.000000 | 4 |
| level_1 | 0.100000 | 0.011905 | 0.021277 | 84 |
| level_2 | 0.503525 | 0.566893 | 0.533333 | 882 |
| level_3 | 0.487836 | 0.468059 | 0.477743 | 814 |

Level 0 在 Test 中只有 4 条，全部被预测为 Level 2，F1 为 0。该类别样本极少，指标不稳定，本报告不隐藏、不合并，也不将其作为稳定性能结论。

### Test 混淆矩阵

| 实际类别 | 预测 Level 0 | 预测 Level 1 | 预测 Level 2 | 预测 Level 3 |
|---|---:|---:|---:|---:|
| Level 0 | 0 | 0 | 4 | 0 |
| Level 1 | 0 | 1 | 62 | 21 |
| Level 2 | 0 | 3 | 500 | 379 |
| Level 3 | 0 | 6 | 427 | 381 |

## 4. 类别频率加权版本

由于主模型 Test accuracy 不高于多数类基线，按要求追加一次基于 Train 类别频率的加权训练。权重只在 Train 上计算，没有在 Test 上调参。

| split | 版本 | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Validation | 加权模型 | 0.446466 | 0.287642 | 0.447731 | 0.621472 |
| Test | 加权模型 | 0.461323 | 0.279720 | 0.465888 | 0.600954 |

加权后 Test macro-F1 从 0.258088 提高到 0.279720，但 accuracy 从 0.494395 降至 0.461323，ROC-AUC 也下降。因此不把加权模型作为默认保存模型；保存的是未加权主模型。

## 5. 消融实验

消融移除 `fatigue_score`、`attention_score`，使用剩余 9 个特征，以相同参数和同一 Train/Test 直接训练 `xgboost.XGBClassifier`。

| split | 特征集 | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Validation | full 11 | 0.478656 | 0.263919 | 0.458630 | 0.634498 |
| Validation | ablation 9 | 0.476557 | 0.249278 | 0.455294 | 0.629149 |
| Test | full 11 | 0.494395 | 0.258088 | 0.482663 | 0.635055 |
| Test | ablation 9 | 0.486547 | 0.249395 | 0.474914 | 0.613149 |

Test 相对 full 的变化：

| 指标 | Δ（ablation - full） |
|---|---:|
| Accuracy | -0.007848 |
| Macro-F1 | -0.008693 |
| Weighted-F1 | -0.007748 |
| ROC-AUC | -0.021906 |

移除两个规则特征后所有 Test 指标均下降，尤其 ROC-AUC 下降约 0.0219。该证据暂不支持直接删除两个规则特征，但仍不能在阶段 E 把相关性解释为因果增益。

## 6. 特征重要性

主模型按 gain 排序的前 10 个特征：

| 排名 | 特征 | Gain |
|---:|---|---:|
| 1 | fatigue_score | 1.684542 |
| 2 | attention_score | 1.633461 |
| 3 | ear_std | 1.613268 |
| 4 | ear_mean | 1.593320 |
| 5 | pitch_mean | 1.590812 |
| 6 | perclos | 1.555057 |
| 7 | yaw_mean | 1.517856 |
| 8 | roll_mean | 1.515345 |
| 9 | face_lost_duration | 1.507192 |
| 10 | blink_rate | 1.234590 |

未进入前十的 `head_down_duration` gain 为 1.097955。

## 7. 模型保存与加载验证

模型目录：`models/xgboost/engagement-v1/`

包含：

- `xgboost_model.json`
- `feature_names.json`
- `label_mapping.json`
- `training_config.yaml`
- `model_metadata.json`
- `metrics.json`

保存后通过新的 `ModelRegistry()` 和 `get_model("xgboost", "engagement-v1")` 成功加载。前 3 条 Test 样本预测为 `[2, 2, 2]`，对应名称为 `["level_2", "level_2", "level_2"]`，确认加载后的标签映射为 `level_0` 到 `level_3`。

模型目录大小约 1.73 MB；`outputs/stage_e/` 产物约 0.20 MB。

## 8. 产物

| 文件 | 内容 |
|---|---|
| `outputs/stage_e/metrics.json` | 数据、参数、Validation/Test、加权与消融完整指标 |
| `outputs/stage_e/per_class_test.csv` | Test 每类 P/R/F1/support |
| `outputs/stage_e/confusion_test.csv` | 4×4 Test 混淆矩阵 |
| `outputs/stage_e/feature_importance.csv` | gain 特征重要性 |
| `outputs/stage_e/test_predictions.csv` | 1784 条 clip_id、真实/预测标签和四类概率 |
| `outputs/stage_e/ablation.csv` | full 11 与 ablation 9 对照 |
| `models/xgboost/engagement-v1/metrics.json` | 与输出目录一致的指标快照 |

Test 概率每行之和与 1 的误差小于 `1e-6`。每类 support 与混淆矩阵元素之和均为 1784。

## 9. 运行与复现

最终训练运行：

```powershell
python scripts/train_engagement.py
```

最终运行耗时约 1.409 秒，其中主模型拟合约 0.484 秒。命令行总墙钟约 2.7 秒，包含 Python 与 XGBoost 初始化。

测试命令：

```powershell
pytest tests/model/test_engagement_labels.py tests/model/test_engagement_training.py
```

新增测试通过；连同现有 XGBoost 与模型注册测试，本次验收共 `37 passed`。

## 10. 结论

阶段 E 的训练、评估、消融、模型注册和可加载验证均已执行完成，但性能验收项“Test accuracy 高于多数类基线”**未通过**：

- Test accuracy：主模型 0.494395，多数类基线 0.494395，完全相同；
- 主模型 macro-F1、weighted-F1 和 AUC 高于基线，但 accuracy 没有提升；
- 频率加权版本牺牲 accuracy，换取更高的 macro-F1，仍不能视为优于基线；
- Level 0 在 Test 中只有 4 条且 recall 为 0，不能据此得出稳定分类能力；
- 9 特征消融在 Test 的所有主要指标均低于完整 11 特征。

因此该结果应作为 V1 基线保留，不能美化为主模型优于多数类基线。后续若要提升，需要在阶段 E 之外重新讨论数据不平衡、特征有效性或模型策略；本阶段不继续调参。

本阶段未接入 API、未修改前端、未操作数据库、未修改风险 `LABEL_MAPPING`/`RiskPredictor`、未执行 SHAP 或 LSTM、未进入阶段 F。

---

## 11. 验收记录（规划侧，2026-10-31）

- **工程验收**：通过。指标、每类表、混淆矩阵、消融、特征重要性、模型注册与加载均独立复现；`test_predictions.csv` 复算 accuracy 0.494395、macro-F1 0.258088，与报告一致；概率行和误差 8.75e-08。
- **性能验收**：不通过。Test accuracy 与多数类基线完全相同（0.494395），不满足“高于基线”；按计划 R9 交付“基线 + 差距说明”，不调低标准、不美化。
- **模型可用性**：`ModelRegistry().get_model("xgboost","engagement-v1")` 加载成功，`label_mapping` 为 `level_0`–`level_3`，3 条样本预测可复现。
- **测试**：新增 6 项全部通过；`tests/model` 其余用例在本会话因临时目录权限报错，非断言失败。
- **口径提醒**：加权版本的触发条件用了 Test accuracy 比较；默认保存的仍是无加权主模型，未发生 Test 选型，但该写法在报告中应继续披露。
- **下一步**：阶段 F 按基线模型接入，展示时明确“基线模型”，不宣称高准确率。
