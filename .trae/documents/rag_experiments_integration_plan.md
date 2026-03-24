# RAG 实验整合实施计划

## 一、项目概述

本计划旨在将 `e:\Code_Personal\Subject\Agent` 和 `e:\Code_Personal\Subject\My_project` 两个项目中的所有实验迁移到 `e:\Code_Personal\Subject\test02` 项目中，利用 test02 已有的模块化架构实现实验功能。

***

## 二、实验清单

### 2.1 Agent 项目实验 (5个Baseline + 1个完整评测)

| 序号 | 实验名称  | 原文件                        | 检索策略            | 依赖数据                                                                              |
| -- | ----- | -------------------------- | --------------- | --------------------------------------------------------------------------------- |
| 1  | 无检索基线 | baseline\_no\_retrieval.py | 无检索             | rag\_300\_multihop.json                                                           |
| 2  | 纯向量检索 | baseline\_naive.py         | Vector Top-5    | processed\_md.json, rag\_300\_multihop.json                                       |
| 3  | 混合检索  | baseline\_hybrid.py        | BM25+Vector+RRF | processed\_md\_expanded.json, rag\_1000\_multihop.json                            |
| 4  | 图谱增强  | baseline\_graph\_rag.py    | Vector+Graph    | processed\_md\_expanded.json, real\_kg\_from\_json.json, rag\_1000\_multihop.json |
| 5  | 动态路由  | baseline\_live\_rag.py     | BM25/Vector动态选择 | processed\_md\_expanded.json, rag\_1000\_multihop.json                            |
| 6  | 完整系统  | evaluate\_rag.py           | 差异化路由+Rerank    | processed\_md.json, real\_kg\_from\_json.json, rag\_300\_multihop.json            |

### 2.2 My\_project 项目实验 (HotpotQA)

| 序号 | 实验名称         | 原文件                       | 功能描述              | 依赖数据                                            |
| -- | ------------ | ------------------------- | ----------------- | ----------------------------------------------- |
| 7  | HotpotQA完整流程 | hotpotqa\_main\_update.py | 文档向量化+检索+图谱+重排+评估 | validation parquet, valid\_title\_sentence.json |

***

## 三、数据准备计划

### 3.1 目录结构规划

```
e:\Code_Personal\Subject\test02\data\
├── agent\                          # Agent项目数据
│   ├── processed_md.json           # 知识库(已存在)
│   ├── processed_md_expanded.json  # 扩展知识库(已存在)
│   ├── real_kg_from_json.json      # 知识图谱(已存在)
│   ├── rag_300_multihop.json       # 300题测试集(已存在)
│   ├── rag_1000_multihop.json      # 1000题测试集(已存在)
│   └── rag_benchmark_300.json      # 标准测试集(需复制)
│
├── hotpotqa\                       # My_project数据
│   ├── validation-00000-of-00001.parquet  # 原始验证集(已存在)
│   ├── valid_title_sentence.json          # 处理后数据(已存在)
│   ├── local_graph.json                   # 本地图谱(已存在)
│   └── vector_stores\                     # 向量存储(已存在)
│       ├── single_sentence\
│       └── valid_title_sentence\
│
└── results\                        # 实验结果目录(新建)
    ├── exp_01_no_retrieval\
    ├── exp_02_naive_rag\
    ├── exp_03_hybrid_rag\
    ├── exp_04_graph_rag\
    ├── exp_05_live_rag\
    ├── exp_06_full_system\
    └── exp_07_hotpotqa\
```

### 3.2 数据复制清单

| 源路径                                                | 目标路径                  | 文件说明        |
| -------------------------------------------------- | --------------------- | ----------- |
| Agent/rag\_benchmark\_300.json                     | test02/data/agent/    | 标准难度测试集     |
| Agent/rag\_benchmark\_300\_hard.json               | test02/data/agent/    | 高难度测试集      |
| My\_project/data/validation-00000-of-00001.parquet | test02/data/hotpotqa/ | HotpotQA验证集 |

***

## 四、代码适配计划

### 4.1 需要修改的 src 文件

#### 4.1.1 配置文件适配 (src/utils/config.py)

**修改内容:**

* 确保 API 配置使用统一的 OpenAI 兼容接口

* 不添加新的 API Key，使用现有配置

* 添加实验相关配置项

#### 4.1.2 检索器增强 (src/retrievers/)

**需要增强的功能:**

* `vector_retriever.py`: 添加 BGE-M3 模型支持

* `keyword_retriever.py`: 添加 BM25 实现 (已有)

* `graph_retriever.py`: 添加实体链接和邻居展开功能

* `hybrid_retriever.py`: 添加 RRF 融合算法

#### 4.1.3 工具模块增强 (src/tools/)

**需要增强的功能:**

* `semantic_tool.py`: 添加 MMR 多样性检索

* `graph_tool.py`: 添加图谱实体链接功能

* 新增 `bm25_tool.py`: BM25 检索工具

#### 4.1.4 链模块增强 (src/chains/)

**需要增强的功能:**

* `baseline_chains.py`: 添加 LiveRAG 动态路由链

* 添加路由决策逻辑

#### 4.1.5 评估模块增强 (src/evaluation/)

**需要增强的功能:**

* `evaluator.py`: 添加断点续跑功能 (已有 CheckpointManager)

* 添加 LLM-based 评估指标 (Correctness, Faithfulness)

### 4.2 需要新增的功能

#### 4.2.1 问题分类器增强

在 `src/agents/specialized_agents/` 中增强分类功能:

* 规则分类器 (基于正则)

* LLM 分类器

* 混合分类器

#### 4.2.2 查询处理工具

新增查询处理功能:

* 查询改写

* 查询扩展

* 查询分解

***

## 五、实验文件实施计划

### 5.1 实验 01: 无检索基线

**文件**: `experiments/exp_01_no_retrieval.py`

**实现方案:**

```python
# 直接使用 src.llms 中的客户端
# 不加载任何文档，直接让 LLM 回答问题
# 使用 src.evaluation 中的评估指标
```

**核心逻辑:**

1. 加载测试集
2. 对每个问题调用 LLM 生成答案
3. 计算评估指标
4. 保存结果报告

### 5.2 实验 02: 纯向量检索

**文件**: `experiments/exp_02_naive_rag.py`

**实现方案:**

```python
# 使用 src.storage.FAISSVectorStore
# 使用 src.retrievers.VectorRetriever
# 使用 src.tools.SemanticSearchTool
# 使用 src.llms.create_client()
```

**核心逻辑:**

1. 加载知识库并构建向量索引
2. 对每个问题进行向量检索 Top-5
3. 使用检索结果生成答案
4. 计算评估指标

### 5.3 实验 03: 混合检索

**文件**: `experiments/exp_03_hybrid_rag.py`

**实现方案:**

```python
# 使用 src.retrievers.HybridRetriever
# 实现 RRF (Reciprocal Rank Fusion) 融合算法
# BM25 + Vector 混合检索
```

**核心逻辑:**

1. 构建 BM25 索引
2. 构建向量索引
3. 并行检索并融合结果
4. 生成答案并评估

### 5.4 实验 04: 图谱增强

**文件**: `experiments/exp_04_graph_rag.py`

**实现方案:**

```python
# 使用 src.storage.LocalGraphStore
# 使用 src.retrievers.GraphRetriever
# 使用 src.tools.GraphSearchTool
```

**核心逻辑:**

1. 加载知识图谱
2. 向量检索 + 图谱实体链接
3. 合并上下文
4. 生成答案并评估

### 5.5 实验 05: 动态路由

**文件**: `experiments/exp_05_live_rag.py`

**实现方案:**

```python
# 实现 LiveRAG 动态路由策略
# BM25/Vector 双路检索
# 置信度路由决策
```

**核心逻辑:**

1. 并行执行 BM25 和向量检索
2. 计算 Top-1 置信度分数
3. 根据置信度选择结果
4. 记录路由统计

### 5.6 实验 06: 完整系统

**文件**: `experiments/exp_06_full_system.py`

**实现方案:**

```python
# 使用 src.agents.MasterAgent
# 使用 src.chains.ComplexRAGChain
# 差异化路由 + Rerank
```

**核心逻辑:**

1. 问题分类 (Fact/Explanation/Reasoning)
2. 差异化检索策略
3. Rerank 重排序
4. 生成答案并评估

### 5.7 实验 07: HotpotQA

**文件**: `experiments/exp_07_hotpotqa.py`

**实现方案:**

```python
# 适配 HotpotQA 数据格式
# 使用 test02 的完整 RAG 流程
# 支持父切片映射
```

**核心逻辑:**

1. 加载 HotpotQA 数据
2. 构建向量索引 (段落级 + 句子级)
3. 检索 + 图谱扩展 + 重排序
4. 父切片映射
5. 生成答案并评估

***

## 六、实施步骤

### 阶段一: 数据准备 (预计 1 小时)

1. [ ] 创建结果目录结构
2. [ ] 复制缺失的数据文件
3. [ ] 验证数据完整性

### 阶段二: 代码适配 (预计 3 小时)

1. [ ] 修改配置文件
2. [ ] 增强检索器模块
3. [ ] 增强工具模块
4. [ ] 增强链模块
5. [ ] 增强评估模块

### 阶段三: 实验实现 (预计 4 小时)

1. [ ] 实现实验 01: 无检索基线
2. [ ] 实现实验 02: 纯向量检索
3. [ ] 实现实验 03: 混合检索
4. [ ] 实现实验 04: 图谱增强
5. [ ] 实现实验 05: 动态路由
6. [ ] 实现实验 06: 完整系统
7. [ ] 实现实验 07: HotpotQA

### 阶段四: 验证测试 (预计 1 小时)

1. [ ] 运行每个实验的基本测试
2. [ ] 验证结果输出格式
3. [ ] 确保断点续跑功能正常

***

## 七、技术规范

### 7.1 代码规范

* 使用 test02/src 中已有的功能，不重复实现

* 遵循 test02 的代码风格和架构设计

* 所有实验文件放在 experiments/ 目录

* 结果文件放在 data/results/ 对应子目录

### 7.2 配置规范

* 不添加新的 API Key

* 使用统一的 OpenAI 兼容接口配置

* 通过环境变量或 .env 文件管理配置

### 7.3 结果规范

每个实验结果目录包含:

```
data/results/exp_XX_name/
├── raw/                 # 原始数据
│   └── predictions.json
├── processed/           # 处理后数据
│   └── metrics.json
└── analysis/            # 分析报告
    └── report.md
```

***

## 八、风险评估

| 风险       | 影响 | 缓解措施        |
| -------- | -- | ----------- |
| 数据格式不兼容  | 中  | 编写数据转换工具    |
| API 调用限制 | 高  | 添加速率限制和重试机制 |
| 内存不足     | 中  | 使用批处理和流式处理  |
| 图谱加载失败   | 低  | 添加错误处理和降级策略 |

***

## 九、交付物

1. **代码文件**: 7 个实验脚本 (experiments/exp\_XX.py)
2. **适配代码**: 修改后的 src 模块
3. **数据文件**: 完整的数据目录结构
4. **结果文件**: 每个实验的运行结果
5. **文档**: 实验说明文档

***

## 十、总结

本计划将两个 RAG 项目的实验整合到 test02 的模块化架构中，充分利用已有的组件，避免重复开发，确保实验的可维护性和可扩展性。
