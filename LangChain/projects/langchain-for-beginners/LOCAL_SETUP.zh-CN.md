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
- 只读取仓库根目录 `.env`，不再在当前项目生成 `.env`。

## 第一次使用

在仓库根目录基于 `.env.example` 创建 `.env`，并补全本课程使用的兼容字段：

```dotenv
AI_API_KEY=
AI_ENDPOINT=
AI_MODEL=
AI_EMBEDDING_MODEL=
AI_COMPARE_MODELS=
```

其中 `AI_COMPARE_MODELS` 默认会在 DeepSeek 场景下生成类似 `deepseek-v4-flash,deepseek-chat` 的列表，供模型对比示例直接复用。

可选执行检查命令：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

## 运行建议

先确保根目录依赖已同步：

```powershell
uv sync
```

然后进入当前目录运行上游提供的测试脚本：

```powershell
uv run --env-file ..\..\..\.env python .\scripts\test_setup.py
```

如果你想跟着章节走，建议顺序：

1. `00-course-setup`
2. `01-introduction`
3. `02-chat-models`

## 说明

上游 README 里默认推荐 GitHub Models 或 Microsoft Foundry。本仓库统一从根 `.env` 读取 OpenAI / DeepSeek / DashScope 等 OpenAI 兼容配置。
