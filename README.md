# GraphMem

GraphMem 是一种 **token-efficient memory 算法**，通过图结构组织长期记忆，根据查询自适应选择召回范围和上下文预算，为 LLM 提供精简、可追溯的记忆上下文。

GraphMem 提供 **Python SDK、HTTP API 和持久化 Memory 服务**，支持记忆写入、检索、Prompt 构建与回答生成。可以嵌入 Python 应用，也可以作为独立服务接入；生成的 `messages` 可直接交给产品现有的 LLM。

## 核心能力

| 能力 | 说明 |
|---|---|
| 记忆写入 | 保存对话原文，抽取事实、关系与时间信息，建立图结构和向量索引 |
| 自适应召回 | 根据查询调整召回容量与 token 预算，结合图关系和向量检索定位相关证据 |
| 上下文构建 | 将图关系、关键原文和时间信息整理为 LLM 可用的 Prompt，并返回证据来源 |
| 持久化服务 | 支持租户隔离、幂等写入、失败恢复和在线备份 |

配置选项与接口约束见 [集成说明](docs/INTEGRATION.md)。

## 安装与启动

支持 Python 3.11+、Linux。服务调用外部兼容接口，本身无需 GPU。

```bash
python -m pip install '.[service]'

# 无模型的持久化 / API 联调：省略语义抽取与向量召回
graphmem serve --offline --data-dir ./data/graphmem --port 8090
```

**持久化默认开启**，数据保存在 `--data-dir`，重启时使用同一路径即可继续读取。`--offline` 仅用于联调；下面是完整方案的启动方式：

1. 准备 chat-completions 与 embeddings 服务，编辑 [config/memory-service.json](config/memory-service.json) 中的模型名、地址和数据目录。
2. 准备与 `llm_model` 匹配的本地 `tokenizer.json`，用于预算计数。
3. 启动：

```bash
export GRAPHMEM_TOKENIZER_PATH=/models/qwen/tokenizer.json
# 模型服务需要认证时，设置配置中 *_api_key_env 指定的环境变量
graphmem check --config config/memory-service.json
graphmem serve --config config/memory-service.json --port 8090
```

`check` 校验配置和 tokenizer，不发模型请求。默认只监听 `127.0.0.1`；跨主机部署需设置 `GRAPHMEM_API_KEY`，并使用 `--host 0.0.0.0`。Docker 持久卷示例见 [deploy/memory](deploy/memory/compose.yaml)。

## HTTP 接入

`X-Tenant-ID` 由产品后端确定。下例使用本地无认证服务；配置了服务密钥时，每个业务请求还需 `Authorization: Bearer <GRAPHMEM_API_KEY>`。

```bash
# 写入并同步建立索引；重试必须保留 request_id 和原始 payload
curl -sS http://127.0.0.1:8090/v1/memories/alice/messages \
  -H 'Content-Type: application/json' -H 'X-Tenant-ID: product-a' \
  -d '{"request_id":"event-001","session_id":"chat-001","messages":[
    {"role":"user","speaker":"Alice","timestamp":"2026-10-13",
     "content":"I adopted a dog named Luna."}]}'

# 返回原文证据与自适应召回 trace
curl -sS http://127.0.0.1:8090/v1/memories/alice/retrieve \
  -H 'Content-Type: application/json' -H 'X-Tenant-ID: product-a' \
  -d '{"query":"What is the name of Alice\u0027s dog?"}'

# 返回 messages、证据 ID、Prompt 哈希，可直接交给产品自身的 LLM
curl -sS http://127.0.0.1:8090/v1/memories/alice/prepare \
  -H 'Content-Type: application/json' -H 'X-Tenant-ID: product-a' \
  -d '{"query":"What is the name of Alice\u0027s dog?"}'
```

| API | 用途 |
|---|---|
| `GET /healthz` | 服务模式、配置标识、待重建数量 |
| `GET /v1/memories/{id}` | 记忆状态、消息数量 |
| `POST /v1/memories/{id}/messages` | 幂等追加；成功返回 `status: ready` |
| `POST /v1/memories/{id}/retrieve` | 仅返回检索证据 |
| `POST /v1/memories/{id}/prepare` | 生成带记忆上下文的 Prompt，不调用回答模型 |
| `POST /v1/memories/{id}/answer` | 生成答案；须配置 `generate_answers: true` |
| `POST /v1/memories/{id}/rebuild` | 用 `{}` 请求重建持久化原文的图和向量 |

## Python SDK

```python
from graphmem import MemoryService, ServiceConfig

config = ServiceConfig.from_file("config/memory-service.json")
# 本地联调可改为 ServiceConfig.offline("./data/graphmem")
with MemoryService(config) as memory:
    memory.add("alice", [{"role": "user", "speaker": "Alice",
                          "content": "I adopted a dog named Luna."}],
               tenant_id="product-a", session_id="chat-001", request_id="sdk-event-001")
    request = memory.prepare("alice", "What is the name of Alice's dog?",
                             tenant_id="product-a")
    messages = list(request.messages)  # 产品 LLM 的 messages 参数
    print(request.evidence_turn_ids, request.prompt_payload_hash)
```

可直接运行的无模型示例：[examples/persistent_memory.py](examples/persistent_memory.py)。

## 数据保留与恢复

- 一个数据目录由一个服务进程独占；同步写入和查询串行执行，写入后即可读取最新记忆。SDK 与 HTTP 服务不能同时打开同一目录。大批量写入建议每次提交一个会话片段。
- 原文及幂等记录先提交，随后建图 / 索引。模型失败时返回 `503`，原文仍在磁盘，记忆为 `pending`；恢复模型后重试同一请求或调用 `rebuild`。待重建记忆的查询返回 `409`，完成重建后即可查询。
- Docker 必须保留 `/var/lib/graphmem` 持久卷。`memory.sqlite` 是权威数据；`compiled/`、`dense/`、`query_embeddings.sqlite` 是可重建缓存。
- 在线备份：`graphmem backup --data-dir ./data/graphmem --output ./memory-backup.sqlite`。恢复时停止服务，将备份放入**新的空目录**并命名为 `memory.sqlite`，使用相同模型 / 构建配置启动。不要只复制运行中的 SQLite 主文件而遗漏 WAL。
- 服务密钥是后端间认证；产品网关须从登录身份派生租户 ID，并负责用户权限与 TLS。API 不应直接开放给终端用户。

## 代码与验证

```text
src/graphmem/service/       SDK、HTTP API 与 CLI
src/graphmem/build/         建图与原子信息抽取
src/graphmem/retrieval/     自适应召回与证据组织
src/graphmem/answer/        Prompt 构建与回答生成
config/memory-service.json 服务配置
test/                      核心与服务回归测试
scripts/                   本地模型启动、LoCoMo 复现与服务验证
```

```bash
python -m pip install '.[dev,service]'
python -m pytest -q
python examples/persistent_memory.py --data-dir ./data/example-memory
```

## 本地模型与 LoCoMo 复现

在单独安装了 vLLM 的 Python 环境中，把两个模型放到同一张空闲 GPU：

```bash
python scripts/serve_local_models.py start --gpu 0 \
  --chat-path /models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --embedding-path /models/Qwen3-Embedding-0.6B
python scripts/serve_local_models.py status
# 完成后释放该脚本管理的模型：
# python scripts/serve_local_models.py stop
```

模型就绪后，按上面的完整服务配置启动 Memory 即可持久化使用。LoCoMo 的运行命令见 [复现说明](docs/REPRODUCTION.md)，评测口径、成绩及 token 开支见 [评测结果](docs/LOCOMO_RESULTS.md)。

目前主要验证英文对话，其他语言需结合产品场景评测。

## 许可证

本项目采用 [MIT License](LICENSE)。Copyright (c) 2026 Chen Han, Gao Sen, Pang Zixi, He Bingsheng, NUS Computing。

第三方依赖、模型权重和评测数据遵循各自的许可证。
