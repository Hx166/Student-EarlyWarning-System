# Commit 规范

## 1. Commit 类型

统一采用以下 Commit 类型：

| 类型 | 说明 |
|------|------|
| feat: | 新功能 |
| fix: | 修复 Bug |
| docs: | 文档更新 |
| style: | 代码格式调整（不影响逻辑） |
| refactor: | 代码重构 |
| test: | 测试相关 |
| perf: | 性能优化 |
| chore: | 杂项维护、配置修改 |

---

## 2. Commit 格式

```
类型(模块): 描述
```

建议模块名称统一采用英文，例如：

- cv
- database
- frontend
- backend
- api
- model
- docs
- config
- test

---

## 3. 标准示例

```text
feat(cv): 完成MediaPipe人脸检测封装
```

```text
fix(database): 修复SQLite写入并发锁问题
```

---

## 4. 注意事项

- 每次 Commit 尽量只完成一个功能点。
- Commit 描述应简洁、准确。
- 禁止提交无意义信息（如 update、修改、test 等）。
- 每个功能完成后及时提交，避免一次 Commit 包含大量无关修改。