# v3 对比实验重写实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `exp_v3` 中重写 5 个完全独立的对比实验脚本，并新增 1 个严格汇总基线与 Proposed 结果的离线脚本。

**架构：** 每个基线脚本都自包含 v3 的数据模型、资源加载、检索算法、分阶段并发、生成评判和结果持久化逻辑，运行时不导入其他基线文件。汇总脚本只读取结果文件，不加载模型、向量库或 Neo4j，并在生成总表前严格校验样本与配置一致性。

**技术栈：** Python 3、pandas、NumPy、FAISS 向量存储、Neo4j、DeepSeek API、`concurrent.futures`、CSV/JSON。

**约定例外：** 用户明确要求不编写和运行测试。本计划不创建测试文件，不执行单元测试或集成测试；每个任务仅执行 Python 语法编译和 `--help` 启动检查。

---

## 文件结构

**创建：**

- `exp_v3/semantic_rag_concurrent200.py`：Semantic RAG 独立实验。
- `exp_v3/rerank_rag_concurrent200.py`：Rerank RAG 独立实验。
- `exp_v3/graphrag_concurrent200.py`：GraphRAG 独立实验。
- `exp_v3/kg_rag_concurrent200.py`：KG-RAG 独立实验。
- `exp_v3/macrag_concurrent200.py`：MacRAG 独立实验。
- `exp_v3/aggregate_comparison_results.py`：离线严格汇总 6 种方法。

**只读参考，不修改：**

- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:1`：v3 自包含评测框架。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:427`：资源加载。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:507`：向量检索。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:537`：关键词检索。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:590`：重排序。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:740`：图扩展。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:881`：预算选择。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:1285`：答案生成。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:1350`：语义评判。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:1459`：检索与生成分阶段并发。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:1657`：方法汇总。
- `exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py:1699`：逐样本语义记录。
- `new_experiments_v2/core.py:704`：旧基线算法语义，仅用于核对算法边界。

## 统一实现约束

每个基线脚本都必须独立定义以下类型与行为：

```python
@dataclass
class RetrievalConfig:
    sample_size: int = 5
    random_seed: int = 42
    k1: int = 10
    k2: int = 20
    k3: int = 7
    hmax: int = 2
    context_budget: int = 3600
    retrieval_workers: int = 32
    llm_concurrency: int = 200
    rerank_concurrency: int = 5
    generation_max_retries: int = 6
    judge_max_retries: int = 8
```

每个脚本必须保留相同的 `EvidenceUnit`、`ExperimentMetrics`、`MethodResult` 和
`QueryComplexityScorer` 数据契约，结果行必须至少包含：

```python
{
    "id": sample.get("id"),
    "question": question,
    "answer": ground_truth,
    "type": sample.get("type"),
    "level": sample.get("level"),
    "relevant_titles": sorted(relevant_titles),
    "retrieved_titles": retrieved_titles,
    "retrieved_contexts": [unit.content for unit in method_result.units],
    "generated_answer": answer,
    "generation_success": bool(answer),
    "generation_attempts": generation_attempts,
    "generation_error": generation_error,
    "retrieval_metrics": metrics,
    "semantic_metrics": semantic,
    "stats": method_result.stats,
    "complexity_score": complexity_score,
    "route": method_result.stats.get("route", ""),
}
```

每个脚本的 `run()` 只注册一个方法：

```python
method_fns = {METHOD_NAME: self.retrieve_method}
rows_by_method = self.evaluate_methods(method_fns, desc=METHOD_NAME)
```

输出文件固定为：

```text
result_summary.csv
semantic_records.csv
comparison_by_type.csv
details.json
config.json
checkpoint.json
```

所有提交命令必须显式列出目标文件，禁止使用 `git add .`，以免混入用户现有改动。

### 任务 1：重写 Semantic RAG 独立实验

**文件：**

- 创建：`exp_v3/semantic_rag_concurrent200.py`

- [ ] **步骤 1：建立自包含脚本骨架**

以 `adaptive_multiscale_rag_ablation_concurrent200.py` 的 v3 数据契约为行为参考，
在新文件中独立定义配置、数据类、资源加载、向量检索、生成评判、分阶段并发、
汇总、检查点、CSV/JSON 写入和 CLI。删除 Proposed 路由、父级回升、图扩展、
关键词检索和消融注册逻辑。

脚本常量和类名使用：

```python
METHOD_NAME = "Semantic RAG"
RUN_PREFIX = "semantic_rag"

class SemanticRAGExperiment:
    ...
```

- [ ] **步骤 2：实现 Semantic RAG 检索**

```python
def retrieve_method(self, query: str) -> MethodResult:
    start = time.perf_counter()
    candidates = self.vector_retrieve(query, "sentence", self.config.k1)
    units = unique_by_title(candidates)[: self.config.k3]
    return MethodResult(
        units=units,
        stats=self._stats(
            start,
            units,
            expanded_nodes=0,
            route="semantic_rag",
        ),
    )
```

该方法不得调用重排序、关键词召回、图扩展、父级映射或摘要生成。

- [ ] **步骤 3：实现单方法运行和统一输出**

`run()` 创建 `semantic_rag_<timestamp>` 目录，并写出统一的 6 个文件。
`comparison_by_type.csv` 按 `type` 聚合 Recall、Correctness、Faithfulness、
Context Relevance、平均上下文长度、延迟、拒答率。

- [ ] **步骤 4：实现统一 CLI**

入口构造 `RetrievalConfig`，应用命令行覆盖并执行：

```python
experiment = SemanticRAGExperiment(config)
result = experiment.run()
print(json.dumps(result, ensure_ascii=False, indent=2))
```

- [ ] **步骤 5：执行基本检查**

运行：

```powershell
python -m py_compile exp_v3/semantic_rag_concurrent200.py
python exp_v3/semantic_rag_concurrent200.py --help
```

预期：两条命令退出码均为 `0`，第二条显示统一并发与缓存参数。

- [ ] **步骤 6：提交**

```powershell
git add -- exp_v3/semantic_rag_concurrent200.py
git commit -m "feat(实验): 重写 Semantic RAG 对比实验"
```

### 任务 2：重写 Rerank RAG 独立实验

**文件：**

- 创建：`exp_v3/rerank_rag_concurrent200.py`

- [ ] **步骤 1：建立 Rerank RAG 自包含框架**

独立定义与任务 1 相同的评测契约，但加入 API 重排序器、确定性本地回退排序和
`rerank_semaphore`。不得导入 `semantic_rag_concurrent200.py`。

```python
METHOD_NAME = "Rerank RAG"
RUN_PREFIX = "rerank_rag"

class RerankRAGExperiment:
    ...
```

- [ ] **步骤 2：实现重排序检索**

```python
def retrieve_method(self, query: str) -> MethodResult:
    start = time.perf_counter()
    candidates = self.vector_retrieve(query, "sentence", self.config.k1)
    reranked = self.rerank_units(query, candidates, self.config.k3)
    units = unique_by_title(reranked)[: self.config.k3]
    return MethodResult(
        units=units,
        stats=self._stats(
            start,
            units,
            expanded_nodes=0,
            route="rerank_rag",
        ),
    )
```

API 重排序最终失败时调用本地词项重叠排序，并在 `stats` 中写入
`rerank_fallback=True` 与 `rerank_error`。

- [ ] **步骤 3：接入分阶段并发与输出**

检索线程池由 `retrieval_workers` 控制；重排序调用受
`rerank_concurrency` 信号量限制；生成与评判共享默认值为 200 的 LLM 信号量。
结果目录使用 `rerank_rag_<timestamp>`。

- [ ] **步骤 4：执行基本检查**

运行：

```powershell
python -m py_compile exp_v3/rerank_rag_concurrent200.py
python exp_v3/rerank_rag_concurrent200.py --help
```

预期：退出码均为 `0`，帮助信息包含 `--rerank-concurrency`。

- [ ] **步骤 5：提交**

```powershell
git add -- exp_v3/rerank_rag_concurrent200.py
git commit -m "feat(实验): 重写 Rerank RAG 对比实验"
```

### 任务 3：重写 GraphRAG 独立实验

**文件：**

- 创建：`exp_v3/graphrag_concurrent200.py`

- [ ] **步骤 1：建立 GraphRAG 自包含框架**

独立定义统一评测逻辑，并保留结构图连接、图节点转证据、图扩展和预算选择。
删除关键词召回、语义图协同、复杂度路由、父级回升和摘要补充。

```python
METHOD_NAME = "GraphRAG"
RUN_PREFIX = "graphrag"

class GraphRAGExperiment:
    ...
```

- [ ] **步骤 2：实现固定两跳 GraphRAG**

```python
def retrieve_method(self, query: str) -> MethodResult:
    start = time.perf_counter()
    candidates = self.vector_retrieve(query, "sentence", self.config.k1)
    seeds = unique_by_title(candidates)[: self.config.k3]
    expanded = self.graph_expand(
        [unit.title for unit in seeds],
        hops=self.config.hmax,
    )
    units = self.select_with_budget(query, [*seeds, *expanded])
    return MethodResult(
        units=units,
        stats=self._stats(
            start,
            units,
            expanded_nodes=len(expanded),
            route="graphrag",
            graph_hops=self.config.hmax,
        ),
    )
```

`graph_expand()` 对标题和内部节点 ID 分别去重，限制单节点最大度数与邻居数量，
不使用 Proposed 的相关性自适应跳数。

- [ ] **步骤 3：实现 Neo4j 严格与降级模式**

默认连接失败即终止资源加载。传入 `--no-neo4j` 时允许保留种子证据继续运行，
并在每条结果 `stats` 和 `config.json` 中记录 `neo4j_degraded=True`。

- [ ] **步骤 4：执行基本检查**

运行：

```powershell
python -m py_compile exp_v3/graphrag_concurrent200.py
python exp_v3/graphrag_concurrent200.py --help
```

预期：退出码均为 `0`，帮助信息包含 `--no-neo4j`、`--hmax`、
`--max-graph-neighbors`。

- [ ] **步骤 5：提交**

```powershell
git add -- exp_v3/graphrag_concurrent200.py
git commit -m "feat(实验): 重写 GraphRAG 对比实验"
```

### 任务 4：重写 KG-RAG 独立实验

**文件：**

- 创建：`exp_v3/kg_rag_concurrent200.py`

- [ ] **步骤 1：建立 KG-RAG 自包含框架**

独立定义统一评测逻辑，保留向量召回、关键词召回、重排序、结构图一跳扩展和预算选择。
不得导入其他基线脚本。

```python
METHOD_NAME = "KG-RAG"
RUN_PREFIX = "kg_rag"

class KGRAGExperiment:
    ...
```

- [ ] **步骤 2：实现混合召回和固定一跳扩展**

```python
def retrieve_method(self, query: str) -> MethodResult:
    start = time.perf_counter()
    vector_units = self.vector_retrieve(query, "sentence", self.config.k1)
    keyword_units = self.keyword_retrieve(query, self.config.k2)
    candidates = unique_by_title(
        [*vector_units, *keyword_units],
        keep_content_distinct=True,
    )
    seeds = self.rerank_units(query, candidates, self.config.k3)
    expanded = self.graph_expand(
        [unit.title for unit in seeds],
        hops=1,
    )
    units = self.select_with_budget(
        query,
        [*seeds, *expanded],
        max_units=self.config.k3 + self.config.max_graph_neighbors,
    )
    return MethodResult(
        units=units,
        stats=self._stats(
            start,
            units,
            expanded_nodes=len(expanded),
            route="kg_rag",
            graph_hops=1,
        ),
    )
```

- [ ] **步骤 3：实现重排序回退和 Neo4j 降级标记**

沿用 Rerank RAG 的本地回退规则和 GraphRAG 的严格/降级规则。
回退或降级必须进入逐样本 `stats`，不能只写日志。

- [ ] **步骤 4：执行基本检查**

运行：

```powershell
python -m py_compile exp_v3/kg_rag_concurrent200.py
python exp_v3/kg_rag_concurrent200.py --help
```

预期：退出码均为 `0`，帮助信息包含关键词召回、重排序并发和 Neo4j 参数。

- [ ] **步骤 5：提交**

```powershell
git add -- exp_v3/kg_rag_concurrent200.py
git commit -m "feat(实验): 重写 KG-RAG 对比实验"
```

### 任务 5：重写 MacRAG 独立实验

**文件：**

- 创建：`exp_v3/macrag_concurrent200.py`

- [ ] **步骤 1：建立 MacRAG 自包含框架**

独立定义统一评测逻辑，加载句子级与段落级向量库，保留查询复杂度评分和预算选择。
删除关键词召回、图扩展、父级回升和摘要补充。

```python
METHOD_NAME = "MacRAG"
RUN_PREFIX = "macrag"

class MacRAGExperiment:
    ...
```

- [ ] **步骤 2：实现三档尺度切换**

```python
def retrieve_method(self, query: str) -> MethodResult:
    start = time.perf_counter()
    complexity = self.complexity_scorer.compute(query)

    if complexity.score < 0.45:
        candidates = self.vector_retrieve(
            query, "sentence", self.config.k1
        )
        route = "macrag_sentence"
    elif complexity.score < self.config.complexity_threshold:
        sentence_units = self.vector_retrieve(
            query, "sentence", max(1, self.config.k1 // 2)
        )
        paragraph_units = self.vector_retrieve(
            query, "paragraph", max(1, self.config.k1 // 2)
        )
        candidates = self.rerank_units(
            query,
            [*sentence_units, *paragraph_units],
            self.config.k3,
        )
        route = "macrag_mixed"
    else:
        candidates = self.vector_retrieve(
            query, "paragraph", self.config.k1
        )
        route = "macrag_paragraph"

    units = self.select_with_budget(
        query,
        candidates,
        max_units=self.config.k3,
    )
    return MethodResult(
        units=units,
        stats=self._stats(
            start,
            units,
            expanded_nodes=0,
            route=route,
            complexity_score=complexity.score,
        ),
    )
```

- [ ] **步骤 3：输出尺度分布**

除统一 6 个文件外，在 `details.json` 中加入按
`macrag_sentence`、`macrag_mixed`、`macrag_paragraph` 聚合的路由分布。
不新增第 7 个结果文件，以保持各基线输出集合一致。

- [ ] **步骤 4：执行基本检查**

运行：

```powershell
python -m py_compile exp_v3/macrag_concurrent200.py
python exp_v3/macrag_concurrent200.py --help
```

预期：退出码均为 `0`，帮助信息包含 `--complexity-threshold`。

- [ ] **步骤 5：提交**

```powershell
git add -- exp_v3/macrag_concurrent200.py
git commit -m "feat(实验): 重写 MacRAG 对比实验"
```

### 任务 6：实现严格结果汇总脚本

**文件：**

- 创建：`exp_v3/aggregate_comparison_results.py`

- [ ] **步骤 1：定义结果契约**

```python
EXPECTED_METHODS = (
    "Semantic RAG",
    "Rerank RAG",
    "GraphRAG",
    "KG-RAG",
    "MacRAG",
    "Proposed",
)

RESULT_PREFIXES = {
    "Semantic RAG": "semantic_rag_",
    "Rerank RAG": "rerank_rag_",
    "GraphRAG": "graphrag_",
    "KG-RAG": "kg_rag_",
    "MacRAG": "macrag_",
    "Proposed": "adaptive_multiscale_rag_",
}

REQUIRED_FILES = {
    "result_summary.csv",
    "details.json",
    "config.json",
}
```

定义 `LoadedRun` 数据类，包含 `method`、`run_dir`、`summary_row`、
`sample_rows`、`config`。

- [ ] **步骤 2：实现最新有效目录发现**

```python
def discover_latest_run(
    results_root: Path,
    method: str,
) -> Path:
    prefix = RESULT_PREFIXES[method]
    candidates = sorted(
        (
            path
            for path in results_root.iterdir()
            if path.is_dir()
            and path.name.startswith(prefix)
            and REQUIRED_FILES.issubset(
                child.name for child in path.iterdir()
            )
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        try:
            loaded = load_run(candidate, method)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if loaded.sample_rows:
            return candidate
    raise FileNotFoundError(f"未找到 {method} 的有效结果目录")
```

Proposed 的前缀可能同时匹配消融目录，因此 `load_run()` 必须检查
`result_summary.csv` 和 `details.json` 中是否存在 `Proposed`。

- [ ] **步骤 3：实现结果加载与字段规范化**

`load_run()`：

1. 使用 `utf-8-sig` 读取 CSV。
2. 从摘要中精确选择当前方法行。
3. 从 `details.json` 的 `rows_by_method[method]` 读取逐样本结果。
4. 将样本 ID 转为字符串。
5. 拒绝重复 ID、空 ID、空逐样本结果和缺少关键指标的输入。

- [ ] **步骤 4：实现严格一致性校验**

```python
def validate_runs(
    runs: Mapping[str, LoadedRun],
    allow_intersection: bool,
) -> list[str]:
    reference = runs["Proposed"]
    reference_ids = {str(row["id"]) for row in reference.sample_rows}
    warnings: list[str] = []

    for method in EXPECTED_METHODS:
        run = runs[method]
        sample_ids = {str(row["id"]) for row in run.sample_rows}
        if sample_ids != reference_ids:
            if not allow_intersection:
                raise ValueError(
                    f"{method} 与 Proposed 的样本 ID 不一致"
                )
            warnings.append(f"{method} 将只保留共同样本")

        if run.config.get("random_seed") != reference.config.get("random_seed"):
            raise ValueError(f"{method} 与 Proposed 的 random_seed 不一致")

    return warnings
```

宽松模式使用所有方法样本 ID 集合的交集，并将警告写入 `manifest.json`。

- [ ] **步骤 5：生成总表和类型分层表**

`comparison_summary.csv` 从每个运行的摘要行提取统一列，按
`EXPECTED_METHODS` 排序。`comparison_by_type.csv` 必须从逐样本
`retrieval_metrics` 和 `semantic_metrics` 重新计算，避免依赖不同实验目录中
可能缺失的类型表。

- [ ] **步骤 6：写出明细与清单**

```python
manifest = {
    "created_at": datetime.now().isoformat(),
    "strict_mode": not args.allow_intersection,
    "methods": {
        method: str(runs[method].run_dir)
        for method in EXPECTED_METHODS
    },
    "sample_count": len(common_ids),
    "warnings": warnings,
}
```

创建 `comparison_<timestamp>` 目录并写出
`comparison_summary.csv`、`comparison_by_type.csv`、
`comparison_details.json` 和 `manifest.json`。

- [ ] **步骤 7：实现 CLI**

支持：

```text
--results-root
--semantic-rag-dir
--rerank-rag-dir
--graphrag-dir
--kg-rag-dir
--macrag-dir
--proposed-dir
--output-dir
--allow-intersection
```

未显式指定的方法使用最新有效目录发现逻辑。

- [ ] **步骤 8：执行基本检查**

运行：

```powershell
python -m py_compile exp_v3/aggregate_comparison_results.py
python exp_v3/aggregate_comparison_results.py --help
```

预期：退出码均为 `0`，帮助信息列出 6 个方法目录参数和宽松模式开关。

- [ ] **步骤 9：提交**

```powershell
git add -- exp_v3/aggregate_comparison_results.py
git commit -m "feat(实验): 添加 v3 对比结果汇总脚本"
```

### 任务 7：执行跨脚本一致性检查

**文件：**

- 检查：`exp_v3/semantic_rag_concurrent200.py`
- 检查：`exp_v3/rerank_rag_concurrent200.py`
- 检查：`exp_v3/graphrag_concurrent200.py`
- 检查：`exp_v3/kg_rag_concurrent200.py`
- 检查：`exp_v3/macrag_concurrent200.py`
- 检查：`exp_v3/aggregate_comparison_results.py`

- [ ] **步骤 1：编译全部新增脚本**

运行：

```powershell
python -m py_compile `
  exp_v3/semantic_rag_concurrent200.py `
  exp_v3/rerank_rag_concurrent200.py `
  exp_v3/graphrag_concurrent200.py `
  exp_v3/kg_rag_concurrent200.py `
  exp_v3/macrag_concurrent200.py `
  exp_v3/aggregate_comparison_results.py
```

预期：退出码为 `0`，没有语法错误。

- [ ] **步骤 2：检查所有 CLI 入口**

依次运行 6 个脚本的 `--help`。预期每条命令退出码均为 `0`。

- [ ] **步骤 3：检查基线间无导入依赖**

运行：

```powershell
Select-String `
  -Path exp_v3/*_rag_concurrent200.py `
  -Pattern 'from exp_v3|import .*_rag_concurrent200'
```

预期：无输出。

- [ ] **步骤 4：检查现有 v3 文件未被修改**

运行：

```powershell
git status --short
git diff -- `
  exp_v3/adaptive_multiscale_rag_ablation_concurrent200.py `
  exp_v3/adaptive_multiscale_rag_ablation.py `
  exp_v3/adaptive_multiscale_rag_optimized_v2.py `
  exp_v3/adaptive_multiscale_rag_optimized.py
```

预期：第二条命令无输出。第一条命令可显示用户原有改动，但本计划新增的 6 个脚本
应已分别提交。

- [ ] **步骤 5：检查提交历史**

运行：

```powershell
git log -8 --oneline
```

预期：存在 5 个基线提交、1 个汇总脚本提交，以及设计和计划文档提交。
