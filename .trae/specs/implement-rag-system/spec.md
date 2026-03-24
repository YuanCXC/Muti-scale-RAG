# RAG 系统架构实现 Spec

## Why
构建一个完整的检索增强生成 (RAG) 系统，整合多源检索能力（向量、关键词、知识图谱）、智能体调度、以及统一评估体系，以支持复杂问答场景的高效处理。

## What Changes
- 创建核心智能体调度模块 (agents/)
- 实现多策略检索器 (retrievers/)
- 构建业务工作流链 (chains/)
- 封装标准化工具接口 (tools/)
- 建立数据存储层 (storage/)
- 集成大语言模型客户端 (llms/)
- 搭建统一评估体系 (evaluation/)
- 提供通用工具类 (utils/)

## Impact
- Affected specs: 无 (新建项目)
- Affected code: 全新创建 src/ 目录结构

## ADDED Requirements

### Requirement: 智能体调度系统
系统 SHALL 提供主控 Agent 进行意图识别与任务分发，支持领域专用 Agent 处理不同类型问题。

#### Scenario: 意图识别与分发
- **WHEN** 用户提交查询请求
- **THEN** 主控 Agent 识别查询类型并分发给对应的专用 Agent 处理

#### Scenario: 提示词管理
- **WHEN** Agent 需要执行特定任务
- **THEN** 从集中管理的提示词库中获取对应的 prompt 模板

### Requirement: 多策略检索系统
系统 SHALL 支持向量检索、关键词检索、知识图谱检索及混合检索四种检索策略。

#### Scenario: 向量检索
- **WHEN** 执行语义相似度检索
- **THEN** 使用 FAISS 向量库返回 Top-K 相似文档

#### Scenario: 关键词检索
- **WHEN** 执行精确匹配检索
- **THEN** 使用 BM25 算法返回相关文档

#### Scenario: 知识图谱检索
- **WHEN** 需要实体关系扩展
- **THEN** 通过 Neo4j Cypher 或 NetworkX 进行图谱遍历

#### Scenario: 混合检索
- **WHEN** 需要多路召回融合
- **THEN** 使用 RRF (Reciprocal Rank Fusion) 融合多路检索结果

#### Scenario: 重排序
- **WHEN** 检索结果需要精排
- **THEN** 使用 BGE-reranker 模型进行重排序

### Requirement: 业务工作流系统
系统 SHALL 提供兼容 LangChain 1.0 架构的工作流链。

#### Scenario: 复杂 RAG 工作流
- **WHEN** 执行 9 步检索流程
- **THEN** 按序完成查询改写、实体提取、多路检索、重排序、答案生成等步骤

#### Scenario: 基线工作流
- **WHEN** 执行快速检索
- **THEN** 支持 Naive、Hybrid、Graph 三种基线模式

### Requirement: 标准化工具接口
系统 SHALL 提供供智能体或工作流调用的标准化工具。

#### Scenario: 语义检索工具
- **WHEN** 调用 semantic_tool
- **THEN** 返回语义检索结果

#### Scenario: 混合检索工具
- **WHEN** 调用 hybrid_search_tool
- **THEN** 返回混合检索融合结果

#### Scenario: 图谱工具
- **WHEN** 调用 graph_tool
- **THEN** 返回图谱遍历与实体链接结果

### Requirement: 数据存储系统
系统 SHALL 提供向量存储、图存储及文本切分映射能力。

#### Scenario: 向量存储管理
- **WHEN** 需要存储或检索向量
- **THEN** 通过 FAISS 进行段落级与句子级向量管理

#### Scenario: 图存储管理
- **WHEN** 需要图谱操作
- **THEN** 通过 Neo4j 或本地 real_kg 进行实体关系管理

#### Scenario: 文本切分映射
- **WHEN** 需要文档切分
- **THEN** 支持父切片映射与上下文关联

### Requirement: 大语言模型集成
系统 SHALL 支持 DeepSeek API 和 Ollama 本地模型两种 LLM 接入方式。

#### Scenario: DeepSeek API 调用
- **WHEN** 需要云端模型服务
- **THEN** 通过 DeepSeek API 客户端进行调用

#### Scenario: Ollama 本地模型调用
- **WHEN** 需要本地模型服务
- **THEN** 通过 Ollama 客户端调用 qwen2.5:7b-instruct 等模型

### Requirement: 统一评估体系
系统 SHALL 提供检索指标和生成指标的统一评估能力。

#### Scenario: 检索指标计算
- **WHEN** 执行检索评估
- **THEN** 计算 Recall, Precision, MRR, NDCG, MAP 指标

#### Scenario: 生成指标计算
- **WHEN** 执行生成评估
- **THEN** 计算 EM, F1, 语义相似度, Correctness, Faithfulness 指标

#### Scenario: 批量评测
- **WHEN** 执行批量评估
- **THEN** 支持断点续跑机制

### Requirement: 通用工具类
系统 SHALL 提供全局配置管理和统一日志记录。

#### Scenario: 配置管理
- **WHEN** 系统启动
- **THEN** 加载 API Keys、数据库 URL、检索 Top-K 等全局配置

#### Scenario: 日志记录
- **WHEN** 系统运行
- **THEN** 统一记录运行日志
