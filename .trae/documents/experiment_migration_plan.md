# 实验迁移与执行计划

## 1. 项目概述

将 `e:\Code_Personal\Subject\Agent` 和 `e:\Code_Personal\Subject\My_project` 中的所有实验迁移到 `e:\Code_Personal\Subject\test02` 框架中执行。

## 2. 源项目实验分析

### 2.1 Agent 项目实验

| 实验文件 | 实验类型 | 描述 |
|---------|---------|------|
| `baseline_naive.py` | Naive RAG | 基础向量检索 RAG |
| `baseline_hybrid.py` | Hybrid RAG | 混合检索（向量+BM25）RAG |
| `baseline_graph_rag.py` | Graph RAG | 知识图谱增强 RAG |
| `baseline_no_retrieval.py` | No Retrieval | 无检索直接生成 |
| `baseline_live_rag.py` | LiveRAG | 动态路由双路检索 |
| `evaluate_rag.py` | 综合评测 | 多工具多维度评测 |

### 2.2 My_project 项目

医疗图像分割相关项目，包含多个模型配置和推理脚本。

### 2.3 数据文件清单

| 数据文件 | 路径 | 用途 |
|---------|------|------|
| `train_docs.jsonl` | `Agent/data/1data/` | 训练文档 |
| `train_queries.jsonl` | `Agent/data/1data/` | 训练查询 |
| `train_gold_labels.jsonl` | `Agent/data/1data/` | 训练标签 |
| `validation_*.jsonl` | `Agent/data/1data/` | 验证数据 |
| `processed_md.json` | `Agent/` | 处理后的知识库文档 |
| `processed_md_expanded.json` | `Agent/` | 扩展知识库文档 |
| `real_kg_from_json.json` | `Agent/` | 知识图谱 |
| `rag_300_multihop.json` | `Agent/` | 多跳测试集 |
| `rag_benchmark_300.json` | `Agent/` | 基准测试集 |
| `rag_benchmark_300_hard.json` | `Agent/` | 困难测试集 |
| `markdown/*.md` | `Agent/data/markdown/` | 知识库源文件 |

## 3. 目标目录结构

```
e:\Code_Personal\Subject\test02\
├── data/
│   ├── agent/                          # Agent 项目数据
│   │   ├── 1data/
│   │   │   ├── train_docs.jsonl
│   │   │   ├── train_queries.jsonl
│   │   │   ├── train_gold_labels.jsonl
│   │   │   ├── validation_docs.jsonl
│   │   │   ├── validation_queries.jsonl
│   │   │   └── validation_gold_labels.jsonl
│   │   ├── knowledge_base/
│   │   │   ├── processed_md.json
│   │   │   ├── processed_md_expanded.json
│   │   │   └── markdown/               # 36个知识库文档
│   │   └── benchmarks/
│   │       ├── rag_300_multihop.json
│   │       ├── rag_benchmark_300.json
│   │       └── rag_benchmark_300_hard.json
│   ├── my_project/                     # My_project 数据
│   │   └── (待确认)
│   ├── faiss_index/                    # FAISS 索引
│   ├── local_graph/                    # 本地图存储
│   └── results/                        # 实验结果
│       ├── exp_01_naive_rag/
│       ├── exp_02_hybrid_rag/
│       ├── exp_03_graph_rag/
│       ├── exp_04_no_retrieval/
│       ├── exp_05_live_rag/
│       └── exp_06_comprehensive/
├── experiments/                        # 实验脚本
│   ├── exp_01_naive_rag.py
│   ├── exp_02_hybrid_rag.py
│   ├── exp_03_graph_rag.py
│   ├── exp_04_no_retrieval.py
│   ├── exp_05_live_rag.py
│   ├── exp_06_comprehensive.py
│   └── utils/
│       ├── data_loader.py
│       ├── evaluator.py
│       └── report_generator.py
└── src/                                # 源代码框架
    └── ...
```

## 4. 实施步骤

### 阶段 1: 数据迁移 (预计 30 分钟)

#### 步骤 1.1: 创建目录结构
```bash
mkdir -p data/agent/1data
mkdir -p data/agent/knowledge_base/markdown
mkdir -p data/agent/benchmarks
mkdir -p data/my_project
mkdir -p data/results
mkdir -p experiments/utils
```

#### 步骤 1.2: 复制 Agent 数据文件
- 复制 `Agent/data/1data/*.jsonl` → `test02/data/agent/1data/`
- 复制 `Agent/processed_md*.json` → `test02/data/agent/knowledge_base/`
- 复制 `Agent/real_kg*.json` → `test02/data/agent/knowledge_base/`
- 复制 `Agent/rag_*.json` → `test02/data/agent/benchmarks/`
- 复制 `Agent/data/markdown/*.md` → `test02/data/agent/knowledge_base/markdown/`

#### 步骤 1.3: 复制 My_project 数据文件
- 待确认具体数据文件

### 阶段 2: 实验脚本开发 (预计 2 小时)

#### 步骤 2.1: 创建通用工具模块

**文件: `experiments/utils/data_loader.py`**
- `load_json_data()` - 加载 JSON 数据
- `load_jsonl_data()` - 加载 JSONL 数据
- `load_knowledge_base()` - 加载知识库
- `load_benchmark()` - 加载测试集

**文件: `experiments/utils/evaluator.py`**
- `llm_judge()` - LLM 评判
- `calculate_metrics()` - 计算指标
- `RAGEvaluator` - 评测器类

**文件: `experiments/utils/report_generator.py`**
- `generate_report()` - 生成报告
- `save_results()` - 保存结果

#### 步骤 2.2: 创建实验脚本

**实验 1: `exp_01_naive_rag.py`**
- 使用 `test02/src` 的 `VectorRetriever` + `DeepSeekClient`
- 基础向量检索 RAG
- 输出: `data/results/exp_01_naive_rag/`

**实验 2: `exp_02_hybrid_rag.py`**
- 使用 `test02/src` 的 `HybridRetriever`
- 混合检索（向量 + BM25）
- 输出: `data/results/exp_02_hybrid_rag/`

**实验 3: `exp_03_graph_rag.py`**
- 使用 `test02/src` 的 `GraphRetriever` + `LocalGraphStore`
- 知识图谱增强 RAG
- 输出: `data/results/exp_03_graph_rag/`

**实验 4: `exp_04_no_retrieval.py`**
- 直接使用 `DeepSeekClient` 生成
- 无检索基线
- 输出: `data/results/exp_04_no_retrieval/`

**实验 5: `exp_05_live_rag.py`**
- 动态路由双路检索
- 模拟 LiveRAG 策略
- 输出: `data/results/exp_05_live_rag/`

**实验 6: `exp_06_comprehensive.py`**
- 综合评测所有方法
- 多维度对比分析
- 输出: `data/results/exp_06_comprehensive/`

### 阶段 3: 实验执行 (预计 4-8 小时)

#### 步骤 3.1: 执行单个实验
```bash
python experiments/exp_01_naive_rag.py
python experiments/exp_02_hybrid_rag.py
# ... 依次执行
```

#### 步骤 3.2: 执行综合评测
```bash
python experiments/exp_06_comprehensive.py
```

### 阶段 4: 结果分析 (预计 1 小时)

#### 步骤 4.1: 生成对比报告
- 汇总所有实验结果
- 生成对比表格
- 生成可视化图表

#### 步骤 4.2: 输出最终报告
- Markdown 格式报告
- JSON 格式详细数据

## 5. 实验配置

### 5.1 通用配置
```python
# 实验配置
EXPERIMENT_CONFIG = {
    "api_key": "sk-xxx",  # DeepSeek API Key
    "embedding_model": "BAAI/bge-large-zh-v1.5",
    "rerank_model": "BAAI/bge-reranker-large",
    "llm_model": "deepseek-chat",
    "top_k": 10,
    "chunk_size": 512,
    "chunk_overlap": 50,
}
```

### 5.2 数据路径配置
```python
DATA_PATHS = {
    "knowledge_base": "data/agent/knowledge_base/processed_md.json",
    "knowledge_graph": "data/agent/knowledge_base/real_kg_from_json.json",
    "benchmark": "data/agent/benchmarks/rag_300_multihop.json",
    "results_dir": "data/results",
}
```

## 6. 预期输出

### 6.1 每个实验的输出
```
data/results/exp_XX_name/
├── report.json           # 详细评测结果
├── summary.json          # 汇总统计
├── report.md             # Markdown 报告
└── metrics.csv           # 指标表格
```

### 6.2 综合评测输出
```
data/results/exp_06_comprehensive/
├── comparison.json       # 方法对比
├── comparison.md         # 对比报告
├── metrics_table.csv     # 指标对比表
└── charts/               # 可视化图表
    ├── accuracy.png
    ├── latency.png
    └── recall.png
```

## 7. 评测指标

### 7.1 生成质量指标
- **Correctness (准确性)**: 回答核心含义是否正确
- **Context Recall (上下文召回)**: 检索上下文是否包含关键信息
- **Faithfulness (忠实度)**: 回答是否基于检索上下文

### 7.2 检索质量指标
- **Recall@k**: 召回率
- **Precision@k**: 精确率
- **MRR**: 平均倒数排名
- **NDCG**: 归一化折损累积增益

### 7.3 系统性能指标
- **Latency**: 平均响应时间
- **Throughput**: 吞吐量
- **Token Cost**: Token 消耗

## 8. 注意事项

1. **API 限流**: DeepSeek API 有调用频率限制，需要添加适当延迟
2. **内存管理**: 大规模测试时注意内存使用，可分批处理
3. **结果缓存**: 支持断点续传，避免重复计算
4. **错误处理**: 完善的异常处理和日志记录

## 9. 时间估算

| 阶段 | 预计时间 |
|------|---------|
| 数据迁移 | 30 分钟 |
| 工具模块开发 | 1 小时 |
| 实验脚本开发 | 1 小时 |
| 实验执行 | 4-8 小时 |
| 结果分析 | 1 小时 |
| **总计** | **7.5-11.5 小时** |

## 10. 执行命令

```bash
# 1. 创建目录结构
python -c "import os; [os.makedirs(p, exist_ok=True) for p in ['data/agent/1data', 'data/agent/knowledge_base/markdown', 'data/agent/benchmarks', 'data/my_project', 'data/results', 'experiments/utils']]"

# 2. 复制数据文件
# (使用文件操作工具)

# 3. 运行实验
python experiments/exp_01_naive_rag.py
python experiments/exp_02_hybrid_rag.py
python experiments/exp_03_graph_rag.py
python experiments/exp_04_no_retrieval.py
python experiments/exp_05_live_rag.py
python experiments/exp_06_comprehensive.py
```
