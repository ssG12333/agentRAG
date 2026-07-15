# 11｜CLI 集成与调试

> 状态：**已实现** ｜ `index/ask/chat` 已接线；`serve` 仍是占位

## 学习目标与先修知识

- 能从 Click 命令映射到前面各层组件。
- 理解后端自动识别、索引文件配套和显式功能开关。
- 建立“先定位阶段，再看异常”的调试方法。

## 当前实现边界

CLI 已支持 NumPy/IVF-PQ、dense/hybrid、可选 reranker、Mock/GGUF 生成和 Agent chat。`configs/default.yaml` 目前只是参考配置，CLI 尚未自动读取；`serve` 只打印开发中信息。

## 数据流与接口

| 命令 | 主要输入 | 主要输出 | 常见失败点 |
|---|---|---|---|
| `index` | 文档目录、分块、嵌入、后端参数 | `.npz` 或 `.ivfpq` + metadata | 无文档、模型依赖、维度不能整除 |
| `ask` | query、索引、检索模式、模型 | 流式回答与可选来源 | 索引不存在、模型维度不一致 |
| `chat` | 索引、工具、模型、历史 | Agent 多轮回答 | 空索引、工具格式、真实模型不遵循 |
| `serve` | host/port | 当前仅占位信息 | Phase 5 未实现 |

`.ivfpq` 后缀在 `--backend auto` 下选择 IVF-PQ，否则默认为 NumPy。显式传入 `--backend` 会覆盖后缀推断。

## 项目调用链

- 工厂函数集中在 `src/cli/main.py` 顶部，避免命令重复组装。
- `index` 先批量生成全部嵌入，再一次性构建静态 IVF-PQ。
- `ask` 找不到索引时提前返回，不会静默创建空索引。
- `chat` 可从空 store 启动，但搜索工具只会返回无结果。
- `/cache` 显示公开 `agent.cache_stats`，并明确 mode 为 logical。

## 最小实验

```powershell
python examples/learning/run_lab.py --lab 11
python -m src.cli.main --help
python -m src.cli.main ask --help
```

实验只检查命令注册和后端解析，不下载模型。完整真实 embedding 路径：

```powershell
agentrag index --path test_data --save data/learning.npz
agentrag ask "自注意力复杂度是多少？" --index-path data/learning.npz --show-sources
```

该路径会加载真实 embedding；留空 `--model-path` 时生成阶段明确使用 MockLLM。

## 常见错误、边界与反例

- 修改 YAML 不会改变当前 CLI 行为，应显式传参数。
- IVF-PQ 必须同时保留主索引和 `.meta.npz`。
- `ask` 的 Mock 回答不是检索证据；应同时检查 `--show-sources`。
- CLI 捕获部分异常用于用户提示，调试时需要回到对应组件运行最小测试。

## 练习

1. 为什么索引后端不能只依靠文件后缀判断？
2. 发现答案错误时，如何区分检索错误与生成错误？

<details><summary>参考答案</summary>

1. 用户可能显式指定后端、路径可能没有标准后缀，且同名文件不能证明内部格式正确。2. 先打印来源和候选；若相关 Chunk 未召回，检查检索层；若证据正确但回答错误，检查 Prompt、模型模板和生成参数。

</details>

## 完成检查

- [ ] 能解释 auto、numpy、ivfpq 的优先关系。
- [ ] 知道配置 YAML 当前未接线。
- [ ] 能按文档→嵌入→索引→检索→生成逐层定位问题。

上一章：[10｜Agent](10_agent.md) ｜ 下一章：[12｜评估与实验](12_evaluation.md)
