# 产品集成契约（5.81.0）

主入口为 `graphmem.MemoryService`；HTTP adapter 与 SDK 使用相同路径。安装 wheel 后可在任意工作目录运行 `graphmem` 或 `python -m graphmem`，不需要复制 `scripts/`、研究数据库或数据集。

## 配置与启用范围

`config/memory-service.json` 为部署参数；未指定字段使用 `ServiceConfig` 默认值。JSON 未知字段会报错。配置文件中的相对 `data_dir` 相对于配置文件目录；CLI 的 `--data-dir` 相对于当前目录，且优先级更高。密钥只从 `*_api_key_env` 指定的环境变量读取。

| 配置 | 默认 / 说明 |
|---|---|
| `semantic_build` | `true`：使用 lossless atomic / G5 / R3 建图 |
| `dense_retrieval` | `true`：使用同一 embedding 模型建索引、检索和构建关系 |
| `readout` | `v5_81`：图 + source-focus；`v5_73` 保留证据卡、聚合表等原始回答策略 |
| `require_exact_tokenizer` | `true`：本地 tokenizer 必须可用；预算模型为 `llm_model` |
| `generate_answers` | `false`：默认由产品消费 `prepare.messages`；设为 `true` 开放 `answer` |
| `answer_model` / `answer_base_url` / `answer_api_key_env` | 空值继承构建 LLM 配置；换回答模型时需自行验证 tokenizer 预算适配 |
| `request_profile` | 构建接口：`qwen` / `openai` |
| `answer_request_profile` | 回答接口：`qwen` / `openai` / `omit`（不发厂商专有参数） |
| `answer_reasoning_effort` | `none`；仅 `openai` profile 接受其他档位 |
| `max_output_tokens` | `512`；回答输出上限 |
| `model_timeout_seconds` | `180`；底层模型单次调用超时，现有 SDK 层有有限重试 |
| `build_overrides` | 对随包构建配置按节覆盖，如 `models`、`coarsen`、`edges` |
| `runtime_overrides` | 对随包运行配置按节覆盖，如 `retrieval`、`query_budget` |

服务拥有模型地址、数据路径、缓存目录以及 `semantic_build` / `dense_retrieval` 开关；这些字段以顶层配置为准。底层 `serving` 参数属于历史只读多进程服务，不改变本 HTTP adapter 的单进程一致性模型。HTTP 同时最多接收 16 个请求，可用 `--max-requests` 调整，超限返回 `429`。

在线配置来自 V5.76 自适应预算与 V5.78 语义 witness 配置，默认启用门控 fact / predicate witness。V5.81 组合器复用实验代码的确定性 source-focus planner 和 renderer：16 条基础 focus，加有限形态、直接关系、概念、dense、时间和事件转换 witness；只对 inference 显示图关系导航，对 lookup/list/temporal 显示原文 capsule。多标签只呈现 `comparison` 与 `geographic_resolution` 两类结构约束。

32/64/80 是图检索容量，2,200/3,800/4,500 是该阶段的 evidence-token 预算，均不等于最终完整 Prompt 的大小。最终图式 Prompt 包含辅助来源和指令，受 `query_budget.max_answer_tokens_hard`（默认 13,000）约束；先减少 source-focus，仍超限则保留已受限的 V5.73 Prompt，并在 `trace.optimized_readout.fallback` 标记。`/retrieve` 返回图检索；`/prepare` 返回最终回答所用证据与 Prompt。

本组合按在线自适应结果呈现，未承诺与某一冻结 benchmark Prompt 逐字节相同。历史 Best-of-N oracle、previous-answer verifier、多次回答合并等研究路径不自动作为产品默认路由；需要时可使用 `answer/ensemble.py`、`budget_controller.py`、`incremental.py` 与相应实验脚本。

## 请求与一致性

租户 ID + memory ID 决定隔离命名空间；session ID 与 turn ID 也在此命名空间内。产品网关必须验证用户是否有权访问其映射的 memory ID，不能直接转发终端用户自填的 `X-Tenant-ID`。服务 bearer token 是后端访问密钥，不是用户登录令牌。

消息字段为 `role`、`content`，可选 `speaker`、`listener`、ISO 8601 `timestamp`；角色可为 user/assistant/system/tool，持久化后均作为检索来源，而不是直接插入系统指令。提供真实说话者和原始消息日期有助于实体、相对时间解析。每批 1..256 条、最多 1 MiB；同一 session 支持分批追加。

`request_id` 在单个租户的单个 memory 内唯一，必须由调用方保存并在失败重试时复用。相同 request/payload 返回已有结果；相同 ID 不同 payload 返回 `409`。在回复丢失、进程崩溃或模型临时失败后，该规则仍有效。

写入事务一次提交原文、版本及幂等记录；之后同步重建该 memory 的图，复用已有语义 / embedding 缓存，完成后发布 ready 标记。当前 facade 按整个 memory 重建，未将底层 affected-path 增量调度接为后台任务。检索和写入在进程内串行，以免读到跨版本的 graph / source / vector；模型构建可能耗时较长，请配置调用方超时并按相同 request ID 重试。大规模部署可按租户或 memory 分片到独立目录 / 实例。

失败后不自动丢弃原文、不返回旧图作为新版本。重启保留 pending 状态，调用 `rebuild` 或重试原写入完成恢复。变更构建参数或 embedding 模型后也需 `rebuild`；仅调整查询参数不要求重建。数据目录使用进程文件锁，避免两个 SDK 或服务进程并行写入。

| 状态码 | 含义 |
|---|---|
| 400 | JSON / 字段 / 输入大小错误 |
| 401 | 服务 bearer token 缺失或错误 |
| 404 | 当前租户下没有该 memory / 路由不存在 |
| 409 | 幂等冲突或 memory 待重建 |
| 429 | HTTP 接入容量已满 |
| 503 | 模型配置不可用、建索引失败或未启用回答 |

## 部署、备份与验证

容器配置中的 `127.0.0.1` 指容器自身，部署前将模型地址改为可从容器访问的 DNS / 服务地址。配置中需要模型认证时可将 `llm_api_key_env` / `embedding_api_key_env` 设为 `GRAPHMEM_MODEL_API_KEY`，通过环境注入。然后执行：

```bash
export GRAPHMEM_API_KEY='replace-with-your-service-secret'
export GRAPHMEM_TOKENIZER_PATH=/absolute/path/to/tokenizer.json
docker compose -f deploy/memory/compose.yaml up -d --build
```

`memory.sqlite` 保存原文、图、源向量、模型调用缓存与幂等记录。在线备份使用 SQLite backup API，包含 WAL 中尚未 checkpoint 的事务；派生缓存无需纳入恢复。请妥善保管原文与模型缓存的备份。恢复到新目录时使用相同构建配置；配置路径自身和数据目录位置不影响已发布版本的有效性。

单元 / 集成测试覆盖持久化重开、租户隔离、并发幂等、索引失败恢复、写后读、备份恢复、Prompt 证据不丢失、HTTP 认证及错误映射；模型路径使用确定性的测试客户端，不产生外部 API 费用。产品上线前仍需在实际模型、数据、语言与流量上验证质量和吞吐。

本次交付验证（2026-10-13）：651 项测试通过；独立 wheel 安装后，在仓库外完成 HTTP 写入 / Prompt、SIGKILL 后重启、幂等重试、在线备份与新目录恢复。Docker Compose 配置校验通过；当前账户没有 Docker daemon 访问权限，镜像构建与容器运行未实测。
