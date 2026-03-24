# 实验执行计划

## 1. 概述

执行位于 `e:\Code_Personal\Subject\Agent` 和 `e:\Code_Personal\Subject\My_project` 中的所有实验，将实验脚本迁移到 `e:\Code_Personal\Subject\test02\experiments` 目录，结果保存至 `e:\Code_Personal\Subject\test02\data\results`。

## 2. 当前状态

### 2.1 已完成工作
- ✅ 目录结构已创建
- ✅ Agent 项目数据已复制到 `test02/data/agent/`
- ✅ 6 个实验脚本已创建并通过语法检查
- ✅ 工具模块已创建 (`data_loader.py`, `evaluator.py`, `report_generator.py`)
- ✅ 配置文件 `.env` 已设置

### 2.2 待完成工作
- ⬜ 复制 My_project 数据文件
- ⬜ 创建 My_project 实验脚本
- ⬜ 执行所有实验
- ⬜ 生成综合报告

## 3. 实验清单

### 3.1 Agent 项目实验 (已迁移)

| 实验ID | 实验名称 | 源文件 | 目标文件 | 状态 |
|--------|----------|--------|----------|------|
| exp_01 | Naive RAG | `baseline_naive.py` | `exp_01_naive_rag.py` | ✅ 已创建 |
| exp_02 | Hybrid RAG | `baseline_hybrid.py` | `exp_02_hybrid_rag.py` | ✅ 已创建 |
| exp_03 | Graph RAG | `baseline_graph_rag.py` | `exp_03_graph_rag.py` | ✅ 已创建 |
| exp_04 | No Retrieval | `baseline_no_retrieval.py` | `exp_04_no_retrieval.py` | ✅ 已创建 |
| exp_05 | Live RAG | `baseline_live_rag.py` | `exp_05_live_rag.py` | ✅ 已创建 |
| exp_06 | Comprehensive | `evaluate_rag.py` | `exp_06_comprehensive.py` | ✅ 已创建 |

### 3.2 My_project 项目实验 (待迁移)

| 实验ID | 实验名称 | 源文件 | 描述 | 状态 |
|--------|----------|--------|------|------|
| exp_07 | HotpotQA Main | `hotpotqa_main_update.py` | HotpotQA 数据集评测 | ⬜ 待创建 |
| exp_08 | Semantic Graph | `Semantic_Graph.py` | 语义图检索实验 | ⬜ 待创建 |
| exp_09 | Structural Graph | `Structural_Graph.py` | 结构图检索实验 | ⬜ 待创建 |

## 4. 数据迁移计划

### 4.1 My_project 数据文件

```
源路径: e:\Code_Personal\Subject\My_project\data\
目标路径: e:\Code_Personal\Subject\test02\data\my_project\

待复制文件:
- local_graph.json          → my_project/local_graph.json
- valid_title_sentence.json → my_project/valid_title_sentence.json

待复制索引:
- HotpotQA_single_sentence_store/    → my_project/single_sentence_store/
- HotpotQA_valid_title_sentence_store/ → my_project/valid_title_store/
```

## 5. 执行步骤

### 阶段 1: 数据迁移 (预计 10 分钟)

**步骤 1.1: 创建 My_project 数据目录**
```bash
mkdir -p data/my_project/single_sentence_store
mkdir -p data/my_project/valid_title_store
```

**步骤 1.2: 复制数据文件**
- 复制 `My_project/data/*.json` → `test02/data/my_project/`
- 复制 `My_project/HotpotQA_*_store/*` → `test02/data/my_project/*/`

### 阶段 2: 创建 My_project 实验脚本 (预计 30 分钟)

**步骤 2.1: 创建 exp_07_hotpotqa_main.py**
- 基于 `hotpotqa_main_update.py`
- 使用 test02 框架的组件
- 输出到 `data/results/exp_07_hotpotqa_main/`

**步骤 2.2: 创建 exp_08_semantic_graph.py**
- 基于 `Semantic_Graph.py`
- 集成知识图谱检索
- 输出到 `data/results/exp_08_semantic_graph/`

**步骤 2.3: 创建 exp_09_structural_graph.py**
- 基于 `Structural_Graph.py`
- 结构化图检索实验
- 输出到 `data/results/exp_09_structural_graph/`

### 阶段 3: 执行实验 (预计 2-4 小时)

**步骤 3.1: 执行 Agent 项目实验**
```bash
# 依次执行 6 个实验
python experiments/exp_01_naive_rag.py --benchmark multihop --num-samples 50
python experiments/exp_02_hybrid_rag.py --benchmark multihop --num-samples 50
python experiments/exp_03_graph_rag.py --benchmark multihop --num-samples 50
python experiments/exp_04_no_retrieval.py --benchmark multihop --num-samples 50
python experiments/exp_05_live_rag.py --benchmark multihop --num-samples 50
python experiments/exp_06_comprehensive.py --num-samples 50
```

**步骤 3.2: 执行 My_project 实验**
```bash
python experiments/exp_07_hotpotqa_main.py
python experiments/exp_08_semantic_graph.py
python experiments/exp_09_structural_graph.py
```

### 阶段 4: 结果汇总 (预计 20 分钟)

**步骤 4.1: 生成综合报告**
- 汇总所有实验结果
- 生成对比表格
- 创建可视化图表

**步骤 4.2: 输出最终报告**
- `data/results/final_report.md`
- `data/results/comparison_table.csv`
- `data/results/metrics_summary.json`

## 6. 输出目录结构

```
e:\Code_Personal\Subject\test02\data\results\
├── exp_01_naive_rag/
│   ├── report.json
│   ├── summary.json
│   ├── report.md
│   └── metrics.csv
├── exp_02_hybrid_rag/
│   └── ...
├── exp_03_graph_rag/
│   └── ...
├── exp_04_no_retrieval/
│   └── ...
├── exp_05_live_rag/
│   └── ...
├── exp_06_comprehensive/
│   ├── comparison.json
│   ├── comparison.md
│   └── metrics_table.csv
├── exp_07_hotpotqa_main/
│   └── ...
├── exp_08_semantic_graph/
│   └── ...
├── exp_09_structural_graph/
│   └── ...
├── final_report.md
├── comparison_table.csv
└── metrics_summary.json
```

## 7. 评测指标

### 7.1 生成质量指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Correctness | 回答准确性 | LLM 评判 1-5 分 |
| Context Recall | 上下文召回率 | 关键信息覆盖率 |
| Faithfulness | 忠实度 | 回答是否基于上下文 |

### 7.2 检索质量指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Recall@k | 召回率 | 相关文档/总相关文档 |
| Precision@k | 精确率 | 相关文档/k |
| MRR | 平均倒数排名 | 1/rank |
| NDCG | 归一化折损累积增益 | 标准化评分 |

### 7.3 系统性能指标
| 指标 | 说明 |
|------|------|
| Latency | 平均响应时间 |
| Token Cost | Token 消耗量 |

## 8. 注意事项

1. **API 限流**: 添加适当延迟避免 API 限流
2. **样本数量**: 先用小样本测试 (--num-samples 10)
3. **错误处理**: 完善的异常处理和日志记录
4. **断点续传**: 支持中断后继续执行

## 9. 执行命令汇总

```bash
# 1. 创建目录
mkdir -p data/my_project/single_sentence_store
mkdir -p data/my_project/valid_title_store
mkdir -p data/results

# 2. 执行实验 (小样本测试)
python experiments/exp_01_naive_rag.py --num-samples 10
python experiments/exp_02_hybrid_rag.py --num-samples 10
python experiments/exp_03_graph_rag.py --num-samples 10
python experiments/exp_04_no_retrieval.py --num-samples 10
python experiments/exp_05_live_rag.py --num-samples 10
python experiments/exp_06_comprehensive.py --num-samples 10

# 3. 完整执行 (取消样本限制)
python experiments/exp_01_naive_rag.py
# ... 其他实验
```

## 10. 时间估算

| 阶段 | 预计时间 |
|------|----------|
| 数据迁移 | 10 分钟 |
| 创建 My_project 实验脚本 | 30 分钟 |
| 小样本测试 | 20 分钟 |
| 完整实验执行 | 2-4 小时 |
| 结果汇总 | 20 分钟 |
| **总计** | **3-5 小时** |
