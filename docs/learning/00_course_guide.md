# 00｜课程指南

> 状态：**已实现** ｜ 路径：快速离线 + 可选真实模型

## 学习目标与先修知识

- 会使用 Python 函数、类、虚拟环境和命令行。
- 能阅读基础 NumPy 代码；不要求预先掌握 C++、信息检索或 Transformer。
- 学会以“代码、测试、数据”而不是宣传语判断功能是否完成。

## 环境

```powershell
python -m pip install -e ".[embedding,test]"
python -m src.cli.main --help
python -m pytest tests -q
```

快速路径不需要 GGUF、Cross-Encoder 或网络访问。C++ 章节需要先构建 `agentrag_core`：

```powershell
scripts\build_cpp.bat
python -c "import agentrag_core; print(agentrag_core.add(1, 2))"
```

## 学习方法

每章依次完成：读概念 → 画数据流 → 定位源码 → 运行实验 → 解释输出 → 完成练习。不要背诵某次机器上的毫秒数，重点解释指标为何变化。

## 实验记录模板

```text
实验：
代码提交：
环境：Python / CPU / GPU / 依赖版本
固定变量：
自变量：
原始输出：
观察：
是否支持假设：
反例或异常：
下一步：
```

## 常见错误

- 把 MockLLM 的通过当成真实模型质量验证。
- 把逻辑 Prefix Cache 命中率当成真实 prefill 加速。
- 用合成向量的 Recall 代替真实问答相关性。
- 只报告最好一次结果，不保存参数、种子和失败实验。

## 练习

1. 当前项目哪些阶段完整，哪些阶段仍是部分实现或预研？
2. 为什么 `PLAN.md` 不能作为当前事实来源？

<details><summary>参考答案</summary>

Phase 0、1、2 已完成；Phase 3 的 Mock Agent 与逻辑缓存已实现，但真实 GGUF、真实 KV 复用未完成；Phase 4、5 是预研或规划。`PLAN.md` 写于项目启动阶段，包含预测和候选方案，必须由当前代码、测试和日志校验。

</details>

## 完成检查

- [ ] CLI help 可运行。
- [ ] 完整测试达到当前的 49 个通过，且不低于课程建立前的 46 个基线。
- [ ] 能区分“未执行”“失败”和“通过”。

下一章：[01｜RAG 全链路](01_rag_architecture.md)
