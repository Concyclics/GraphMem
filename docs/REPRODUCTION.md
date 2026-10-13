# LoCoMo 本地复现

复现入口 `scripts/reproduce_locomo.py` 默认使用产品随包配置，从原始会话冷建图，运行自适应检索与 V5.81 Graph + Source-focus，然后每题生成一个答案。默认本地 Qwen3-30B-A3B-Instruct-2507-FP8 回答和评分，Qwen3-Embedding-0.6B 建索引。评分和建图不会把 gold / judge 反馈交给检索或回答链路。

## 准备

1. `python -m pip install -c config/reproduction-constraints.txt '.[service,dev]'`；客户端验证环境为 Python 3.11.15。vLLM 单独安装在模型环境中，本次实测 Python 3.12 / vLLM 0.23.0 / PyTorch 2.11.0 / Transformers 5.12.1。
2. 本地准备两份模型快照，包括完整权重、配置和 tokenizer；不在复现脚本中隐式下载模型。
3. 准备原版 LoCoMo `data/locomo10.json`，以及固定版本评分仓库：

```bash
git clone https://github.com/mem0ai/memory-benchmarks.git third_party/memory-benchmarks
git -C third_party/memory-benchmarks checkout 4b61c5d31b9c668a12b4f5e78064248a02c82d2b
python scripts/convert_locomo10.py --input /data/locomo10.json --output data/locomo10_graphmem.json
```

评分模板 SHA-256 固定为 `8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2`，不匹配则终止。评测 categories 1–4 共 **1,540** 题，category 5 不在此口径内。

## 启动本地模型

```bash
/path/to/vllm/python scripts/serve_local_models.py start \
  --python /path/to/vllm/python --gpu 0 \
  --chat-path /models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --embedding-path /models/Qwen3-Embedding-0.6B
export GRAPHMEM_TOKENIZER_PATH=/models/Qwen3-30B-A3B-Instruct-2507-FP8/tokenizer.json
```

默认使用一张 H100 80GB：chat 占 GPU 显存预算 72%，embedding 占 10%；端口分别为本机 8002 / 8001，最大上下文 16,384，最大并发序列 64。脚本拒绝已占用的 GPU / 端口，状态及日志在 `runs/local-models/`，不自动改用别的 GPU。`status` 检查服务，`stop` 仅停止本脚本记录且进程身份匹配的服务。

## 从零复现 / 断点续跑

```bash
python scripts/reproduce_locomo.py \
  --data data/locomo10_graphmem.json \
  --judge-repo third_party/memory-benchmarks \
  --output runs/locomo-product --workers 16 --build-workers 4
```

重跑相同命令会验证输入身份并从检查点继续。数据、配置、冻结输入或题目集合改变时必须使用新目录；历史答案不会自动导入。每个阶段也可单独执行：`--stage build|prepare|answer|judge|report`。`--limit 20` 仅做冒烟测试，不能作为全量成绩。

同一 GPU 上顺序运行实验。入口用文件锁防止本脚本的多个实验同时运行；不要同时向端点灌入其他大量推理请求，持续 prefill 会拖慢长输出建图并触发超时。评分输出缺少合法标签时最多补一次格式约束，重试调用同样计入账本。

| 产物 | 内容 |
|---|---|
| `manifest.json` | 数据 / 配置 / 固定评分模板、完整题目 ID、采样参数及可选冻结输入哈希 |
| `memory.sqlite`、`built.jsonl` | 原文、图、向量、调用缓存及逐 memory 建库检查点 |
| `prepared.jsonl` | 冻结 Prompt、证据 ID、Prompt 哈希、自适应召回 trace |
| `answers.jsonl` | 单次答案、截断状态、API Token 和缓存标志 |
| `judge.jsonl` | 逐题判定、理由、被评分答案哈希 |
| `api_calls.jsonl` | 实际发出的每次调用及 usage，包括失败重试、embedding 和评分 |
| `summary.json` | 完整率、分类成绩、Token 均值 / 分位数、错误及截断检查 |

`summary.actual_api_usage` 是实际开支账本；缓存中的旧 usage 仅用于描述每题逻辑 Token，不能重复计费。建库、查询 embedding、回答和评分分列；冷建库成本需按服务生命周期摊销。失败请求如果未返回 usage，无法精确知道其服务端消耗，因此另列失败次数。这里报告 Token，不把本地 GPU 开销伪装成 API 美元费用。

## 历史最佳对照

可以显式增加 `--source-db /path/to/graphmem.sqlite --prepared /path/to/prepared_answers.jsonl`，重新回答冻结 Prompt；该模式复用图与检索，**不是冷建库复现**。只传 `--source-db` 时会复用图并重新准备产品 Prompt，必须确保图的构建模型、embedding 和 schema 与配置一致。

历史 v27 的 88.25% 来自另一个评分模型（Luna-medium）及 temperature=0.7 的 candidate 1；本地新测使用 Qwen 评分、temperature=0、n=1。两者不能直接用百分点作质量变化结论。应比较本次使用相同本地模型、评分模板和采样设置的新产品与冻结对照，并同时报告建库 / 检索差异。best-of-8 oracle 不作为可上线成绩。

实际运行环境、结果和验收结论在完成全量验证后记录于本目录的复现结果文件。

真实模型持久化验收可单独运行（输出目录必须未存在）：

```bash
python scripts/smoke_memory_service.py --output runs/memory-service-live
```

它启动临时本机 HTTP 服务，验证建图 / 回答、SIGKILL 后重启、幂等重试、在线备份与新目录恢复，完成后关闭测试服务并保留数据及 `summary.json`。
