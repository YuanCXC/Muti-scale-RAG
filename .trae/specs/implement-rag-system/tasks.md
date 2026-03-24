# Tasks

- [x] Task 1: 创建项目基础结构与配置
  - [x] SubTask 1.1: 创建 src/ 目录结构
  - [x] SubTask 1.2: 实现 utils/config.py 全局配置管理
  - [x] SubTask 1.3: 实现 utils/logger.py 统一日志记录
  - [x] SubTask 1.4: 创建各模块的 __init__.py 文件

- [x] Task 2: 实现大语言模型客户端 (llms/)
  - [x] SubTask 2.1: 实现 llms/base_client.py 基础客户端抽象类
  - [x] SubTask 2.2: 实现 llms/deepseek_client.py DeepSeek API 客户端
  - [x] SubTask 2.3: 实现 llms/ollama_client.py Ollama 本地模型客户端

- [x] Task 3: 实现数据存储层 (storage/)
  - [x] SubTask 3.1: 实现 storage/vector_store/ FAISS 向量库管理
  - [x] SubTask 3.2: 实现 storage/graph_store/ Neo4j 图数据库交互
  - [x] SubTask 3.3: 实现 storage/chunking_mapping/ 文本切分与映射逻辑

- [x] Task 4: 实现检索器模块 (retrievers/)
  - [x] SubTask 4.1: 实现 retrievers/base_retriever.py 基础检索器抽象类
  - [x] SubTask 4.2: 实现 retrievers/vector_retriever.py 向量检索
  - [x] SubTask 4.3: 实现 retrievers/keyword_retriever.py BM25 关键词检索
  - [x] SubTask 4.4: 实现 retrievers/graph_retriever.py 知识图谱检索
  - [x] SubTask 4.5: 实现 retrievers/hybrid_retriever.py 混合检索 (RRF 融合)
  - [x] SubTask 4.6: 实现 retrievers/reranker.py 重排序模块

- [x] Task 5: 实现标准化工具 (tools/)
  - [x] SubTask 5.1: 实现 tools/base_tool.py 基础工具抽象类
  - [x] SubTask 5.2: 实现 tools/semantic_tool.py 语义检索工具
  - [x] SubTask 5.3: 实现 tools/hybrid_search_tool.py 混合检索工具
  - [x] SubTask 5.4: 实现 tools/graph_tool.py 图谱工具

- [x] Task 6: 实现智能体调度模块 (agents/)
  - [x] SubTask 6.1: 实现 agents/prompts/ 提示词管理模块
  - [x] SubTask 6.2: 实现 agents/master_agent/ 主控 Agent
  - [x] SubTask 6.3: 实现 agents/specialized_agents/ 领域专用 Agent

- [x] Task 7: 实现业务工作流 (chains/)
  - [x] SubTask 7.1: 实现 chains/base_chain.py 基础工作流抽象类
  - [x] SubTask 7.2: 实现 chains/baseline_chains.py 基线工作流 (Naive, Hybrid, Graph)
  - [x] SubTask 7.3: 实现 chains/complex_rag_chain.py 9 步检索流程工作流

- [x] Task 8: 实现评估体系 (evaluation/)
  - [x] SubTask 8.1: 实现 evaluation/metrics/retrieval_metrics.py 检索指标
  - [x] SubTask 8.2: 实现 evaluation/metrics/generation_metrics.py 生成指标
  - [x] SubTask 8.3: 实现 evaluation/evaluator.py 批量评测运行器

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 1]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 2, Task 5]
- [Task 7] depends on [Task 4, Task 5, Task 6]
- [Task 8] depends on [Task 7]
