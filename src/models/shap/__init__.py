"""src.models.shap —— SHAP 可解释性分析模块。

基于已训练的 XGBoost 模型生成特征贡献解释，供 API / Dashboard 使用。
"""

from src.models.shap.explainer import ShapExplainer

__all__ = ["ShapExplainer"]
