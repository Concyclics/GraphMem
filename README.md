# GraphMem

面向产品集成的持久化 Memory 框架，提供 **Python SDK、HTTP API 和可交给产品自身 LLM 的回答 Prompt**。当前包版本 **5.81.1**，整合截至 V5.81 的检索与图式证据呈现能力。服务与 SDK 共用一套实现；运行时无需 benchmark 数据或实验产物。

## 默认方案

| 阶段 | 产品配置启用的能力 |
|---|---|
| 写入 / 建图 | 原文持久化、原子信息覆盖、时间归一化、递归层次图、HNSW 候选、父节点门控关系、谓词归一化、构建缓存 |
| 检索 | QueryIR、双通道证据打包、witness 保留、自适应 **32 / 64 / 80** 条图证据及 **2,200 / 3,800 / 4,500** evidence-token 预算 |
| 回答准备 | V5.81 Graph + Source-focus、关系 / 时间 / dense 补充视图、Focus Capsule、按路由启用的导航提示、多标签解析和精简结构约束 |
| 工程 | SQLite WAL、租户隔离、幂等追加、重建恢复、稳定排序、持久化查询向量缓存、FAISS / 编译视图缓存、Prompt 哈希 |

Source-focus 有独立的有限补充预算；整个 Prompt 另受硬上限约束。V5.73 证据卡策略可通过 `readout: "v5_73"` 选择。线上默认每题生成一个答案；所有开关见 [集成说明](docs/INTEGRATION.md)。

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
| `GET /v1/memories/{id}` | 记忆状态、版本、消息数量 |
| `POST /v1/memories/{id}/messages` | 幂等追加；成功返回 `status: ready` |
| `POST /v1/memories/{id}/retrieve` | 仅返回检索证据 |
| `POST /v1/memories/{id}/prepare` | 生成 V5.81 回答 Prompt，不调用回答模型 |
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

- 一个数据目录由一个服务进程独占；同步写入和查询串行执行，写入后即可读取新版本。SDK 与 HTTP 服务不能同时打开同一目录。大批量写入建议每次提交一个会话片段。
- 原文及幂等记录先提交，随后建图 / 索引。模型失败时返回 `503`，原文仍在磁盘，记忆为 `pending`；恢复模型后重试同一请求或调用 `rebuild`。待重建记忆的查询返回 `409`，避免混用不同版本。
- Docker 必须保留 `/var/lib/graphmem` 持久卷。`memory.sqlite` 是权威数据；`compiled/`、`dense/`、`query_embeddings.sqlite` 是可重建缓存。
- 在线备份：`graphmem backup --data-dir ./data/graphmem --output ./memory-backup.sqlite`。恢复时停止服务，将备份放入**新的空目录**并命名为 `memory.sqlite`，使用相同模型 / 构建配置启动。不要只复制运行中的 SQLite 主文件而遗漏 WAL。
- 服务密钥是后端间认证；产品网关须从登录身份派生租户 ID，并负责用户权限与 TLS。API 不应直接开放给终端用户。

## 代码与验证

```text
src/graphmem/service/       SDK、HTTP、CLI、随 wheel 发布的产品配置
src/graphmem/build/         建图与原子信息抽取
src/graphmem/retrieval/     QueryIR、自适应召回、证据打包
src/graphmem/answer/        V5.81 Prompt 与回答策略
config/memory-service.json 产品启动配置
test/                      核心与服务回归测试
scripts/                   本地模型启动、LoCoMo 复现、配对结果审计
test/fixtures/profiles/     仅供核心回归测试使用的旧配置样本
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

模型就绪后，按上面的完整服务配置启动 Memory 即可持久化使用。LoCoMo 从原始会话冷建库、逐题回答、评分及 Token 核算的完整命令见 [复现说明](docs/REPRODUCTION.md)。评分仅在生成结束后读取参考答案；默认每题一个答案，不使用 best-of-N oracle。

GPU0 实测 1,540 题：本地 Qwen 评分 **91.49%**，同口径历史冻结对照 **91.95%**；平均回答 Token **5,770**，比对照少 **5.61%**，两组均零空答案 / 截断。586 项回归测试和安装 wheel 后的真实模型持久化验收通过。成绩口径、冷建库成本及质量差异见 [完整结果](docs/LOCOMO_RESULTS.md)。

仓库已移除废弃版本脚本、报告和部署入口；恢复历史研究代码的方法见 [代码清理记录](docs/ARCHIVE.md)。带旧版本编号但仍被当前方案调用的算法及其回归测试继续保留。现有路由规则主要按英文对话验证，其他语言需产品场景评测。
