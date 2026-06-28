# LangChain for Beginners 本地接入说明

上游仓库：<https://github.com/microsoft/langchain-for-beginners>

当前目录保留了上游课程内容，并额外接入了本仓库统一 `.env` 配置。

当前仓库默认推荐使用 DeepSeek 作为学习阶段的 OpenAI 兼容模型服务。

## 上游快照

- 提交：见 [UPSTREAM_VERSION](./UPSTREAM_VERSION)
- 许可证：见 [LICENSE](./LICENSE)

## 这里的改动边界

- 保留上游原始目录与课程章节。
- 不把密钥直接写入上游示例。
- 通过根目录 `.env` 统一生成本项目所需的 `.env`。

## 第一次使用

在仓库根目录执行：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

执行后会生成当前目录下的 `.env`，映射为上游课程需要的字段：

- `AI_API_KEY`
- `AI_ENDPOINT`
- `AI_MODEL`
- `AI_EMBEDDING_MODEL`
- `AI_COMPARE_MODELS`

其中 `AI_COMPARE_MODELS` 默认会在 DeepSeek 场景下生成类似 `deepseek-v4-flash,deepseek-chat` 的列表，供模型对比示例直接复用。

## 运行建议

先确保根目录依赖已同步：

```powershell
uv sync
```

然后进入当前目录运行上游提供的测试脚本：

```powershell
uv run python .\scripts\test_setup.py
```

如果你想跟着章节走，建议顺序：

1. `00-course-setup`
2. `01-introduction`
3. `02-chat-models`

## 说明

上游 README 里默认推荐 GitHub Models 或 Microsoft Foundry。本仓库额外支持把根 `.env` 中的 OpenAI / DeepSeek / DashScope OpenAI 兼容配置映射过来。
