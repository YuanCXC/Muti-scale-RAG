# Implementation Checklist

## 基础结构
- [x] src/ 目录结构创建完成
- [x] utils/config.py 实现全局配置管理 (API Keys, 数据库 URL, Top-K 参数)
- [x] utils/logger.py 实现统一日志记录
- [x] 所有模块 __init__.py 文件创建完成

## 大语言模型客户端 (llms/)
- [x] llms/base_client.py 基础客户端抽象类实现
- [x] llms/deepseek_client.py DeepSeek API 客户端封装完成
- [x] llms/ollama_client.py Ollama 本地模型客户端封装完成

## 数据存储层 (storage/)
- [x] storage/vector_store/ FAISS 向量库管理 (段落级与句子级存储)
- [x] storage/graph_store/ Neo4j 图数据库交互与 real_kg 管理
- [x] storage/chunking_mapping/ 文本切分与父切片映射逻辑

## 检索器模块 (retrievers/)
- [x] retrievers/base_retriever.py 基础检索器抽象类
- [x] retrievers/vector_retriever.py 向量检索逻辑 (FAISS 集成)
- [x] retrievers/keyword_retriever.py 关键词检索 (BM25 集成)
- [x] retrievers/graph_retriever.py 知识图谱扩展检索 (Neo4j Cypher/NetworkX)
- [x] retrievers/hybrid_retriever.py 混合检索 (含 RRF 融合逻辑)
- [x] retrievers/reranker.py 重排序模块 (BGE-reranker 模型接入)

## 标准化工具 (tools/)
- [x] tools/base_tool.py 基础工具抽象类
- [x] tools/semantic_tool.py 封装单一语义检索能力
- [x] tools/hybrid_search_tool.py 封装混合检索能力
- [x] tools/graph_tool.py 封装图谱遍历与实体链接能力

## 智能体调度模块 (agents/)
- [x] agents/prompts/ 集中管理各类提示词 (实体提取、打分等)
- [x] agents/master_agent/ 主控 Agent (意图识别与任务分发)
- [x] agents/specialized_agents/ 领域专用 Agent (事实型、解释型、推理型)

## 业务工作流 (chains/)
- [x] chains/base_chain.py 基础工作流抽象类 (兼容 LangChain 1.0)
- [x] chains/baseline_chains.py Naive, Hybrid, Graph 基线工作流
- [x] chains/complex_rag_chain.py 9 步检索流程工作流

## 评估体系 (evaluation/)
- [x] evaluation/metrics/retrieval_metrics.py Recall, Precision, MRR, NDCG, MAP 计算
- [x] evaluation/metrics/generation_metrics.py EM, F1, 语义相似度, Correctness, Faithfulness 计算
- [x] evaluation/evaluator.py 批量评测运行器 (含断点续跑机制)
