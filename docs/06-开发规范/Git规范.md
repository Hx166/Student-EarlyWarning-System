# Git 开发规范

## 1. 分支策略

项目采用 Git Flow 简化版本。

| 分支 | 用途 |
|------|------|
| master | 生产版本（当前实际默认分支） |
| develop | 日常开发 |
| dev-cv | CV视觉开发 |
| dev-backend | 后端开发 |
| dev-front | 前端开发 |
| dev-ai | AI算法开发 |

> 注：仓库当前默认分支为 `master`。上表中 `develop` 及 `dev-*` 分支为规划中的开发分支，按需创建。

---

## 2. 开发流程

1. 从 develop 创建个人开发分支。
2. 在对应开发分支完成开发。
3. 提交 Pull Request 合并到 develop。
4. develop 测试稳定后，再合并至 master。

---

## 3. 提交规范

- 所有代码必须经过测试后提交。
- 每个功能使用独立 Commit。
- 合并前解决所有冲突。
- 保持 Commit History 清晰。

---

## 4. 重要规定

> **禁止直接向 master 分支提交代码。**

所有功能必须：

```
开发分支
    ↓
develop
    ↓
测试
    ↓
master
```

所有正式发布版本均来自 master 分支。
