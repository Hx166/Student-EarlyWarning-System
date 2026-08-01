# AGENTS.md

# Student EarlyWarning System AI Development Specification

---

# 1. 项目简介

项目名称：

Student EarlyWarning System

目标：

基于 Python + OpenCV + MediaPipe + Flask + XGBoost
实现课堂学生状态分析与学业预警系统。

AI助手必须始终围绕项目目标进行开发。

---

# 2. AI角色

AI不是聊天机器人。

AI属于项目开发成员。

职责：

- 编写代码
- 修复Bug
- 编写文档
- 编写测试
- 重构代码
- Review代码
- 优化性能

禁止脱离项目讨论无关内容。

---

# 3. 技术栈（固定）

语言：

Python 3.11

IDE：

PyCharm

CV：

OpenCV

MediaPipe

AI：

PyTorch

XGBoost

Scikit-learn

数据库：

SQLite（第一阶段）

MySQL（后期）

Web：

Flask

ECharts

配置：

YAML

日志：

logging

测试：

pytest

任何新增技术栈必须说明原因。

---

# 4. 项目目录

AI不得随意新增目录。

必须遵循：

src/

camera/

cv/

database/

models/

api/

frontend/

utils/

tests/

docs/

configs/

scripts/

---

# 5. 编码规范

遵守PEP8。

必须：

- 使用Type Hint
- 使用Docstring
- 使用logging
- 不允许print调试
- 函数保持单一职责
- 类保持单一职责

禁止：

超过100行函数。

超过500行单文件。

---

# 6. 注释规范

所有公开函数：

必须：

Docstring。

例如：

```python
def detect_face(frame):
    """
    Detect face.

    Args:
        frame: OpenCV frame

    Returns:
        Detection Result
    """
```

禁止无意义注释。

---

# 7. AI输出规范

生成代码前：

必须先阅读相关模块。

不得：

重复实现已有功能。

不得：

复制已有代码。

修改已有代码：

必须保持接口兼容。

---

# 8. 开发原则

遵循：

SOLID

DRY

KISS

YAGNI

所有代码保持低耦合。

---

# 9. Git规范

Commit：

遵循：

feat:

fix:

docs:

style:

refactor:

test:

perf:

chore:

禁止：

直接修改main。

---

# 10. 测试规范

新增功能：

必须：

提供pytest。

Bug：

必须：

增加Regression Test。

---

# 11. 文档规范

新增模块：

必须：

同步更新：

docs/

README

API文档

---

# 12. 数据规范

不得：

提交：

datasets/

models/

logs/

outputs/

到Git。

模型：

统一放：

models/

数据：

统一放：

datasets/

---

# 13. 配置规范

禁止：

硬编码：

路径

数据库

端口

必须：

configs/

yaml

统一管理。

---

# 14. AI禁止事项

禁止：

修改README结构。

禁止：

删除目录。

禁止：

修改接口名称。

禁止：

修改数据库字段。

禁止：

私自引入大型第三方依赖。

禁止：

生成未经验证代码。

---

# 15. AI工作流程

需求

↓

分析

↓

方案

↓

实现

↓

测试

↓

文档

↓

Commit

↓

Review

任何阶段失败必须停止继续生成。

---

# 16. 输出要求

AI回复必须包含：

修改文件

修改原因

影响模块

是否影响接口

是否需要数据库迁移

是否需要重新训练模型

是否需要补充测试

不得只给代码。
