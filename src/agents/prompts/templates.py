# -*- coding: utf-8 -*-
"""预定义提示词模板

提供各类任务的标准提示词模板，包括：
- 实体提取
- 意图识别
- 答案生成
- 打分评估
- 查询改写
"""

from typing import Dict


class PromptTemplates:
    """提示词模板集合
    
    包含所有预定义的提示词模板，支持变量替换。
    """
    
    # ==================== 意图识别提示词 ====================
    
    INTENT_CLASSIFICATION = """你是一个专业的意图分类专家。请根据用户的问题，判断其意图类型。

## 意图类型说明：
1. FACTUAL（事实型）：需要精确的事实性答案，通常可以从单一来源获取
   - 例如："什么是 CLIP？"、"ResNet 是哪一年提出的？"
   
2. EXPLANATORY（解释型）：需要详细解释或综合多个来源的信息
   - 例如："请解释注意力机制的工作原理"、"比较 CNN 和 Transformer 的优缺点"
   
3. REASONING（推理型）：需要多步推理或逻辑分析
   - 例如："如果将 ResNet 的深度增加到 1000 层，可能会遇到什么问题？如何解决？"

## 用户问题：
{query}

## 请分析并返回：
请直接返回意图类型（FACTUAL、EXPLANATORY 或 REASONING），不要包含其他内容。"""

    INTENT_CLASSIFICATION_WITH_CONTEXT = """你是一个专业的意图分类专家。请根据用户的问题和上下文，判断其意图类型。

## 意图类型说明：
1. FACTUAL（事实型）：需要精确的事实性答案，通常可以从单一来源获取
   
2. EXPLANATORY（解释型）：需要详细解释或综合多个来源的信息
   
3. REASONING（推理型）：需要多步推理或逻辑分析

## 上下文信息：
{context}

## 用户问题：
{query}

## 请分析并返回：
请直接返回意图类型（FACTUAL、EXPLANATORY 或 REASONING），不要包含其他内容。"""

    # ==================== 实体提取提示词 ====================
    
    ENTITY_EXTRACTION = """请从以下文本中提取关键实体。

## 文本内容：
{text}

## 请提取以下类型的实体：
1. 方法/模型名称（如 ResNet, Transformer, CLIP 等）
2. 技术概念（如注意力机制、卷积神经网络等）
3. 评估指标（如准确率、F1 分数等）
4. 数据集名称（如 ImageNet, COCO 等）
5. 人物/机构（如作者、研究机构等）

## 输出格式：
请以 JSON 格式返回，格式如下：
{{
    "methods": ["方法1", "方法2"],
    "concepts": ["概念1", "概念2"],
    "metrics": ["指标1"],
    "datasets": ["数据集1"],
    "people": ["人物1"],
    "organizations": ["机构1"]
}}"""

    ENTITY_EXTRACTION_FOR_QUERY = """请从用户问题中提取关键实体和关键词。

## 用户问题：
{query}

## 请提取：
1. 核心概念和术语
2. 方法/模型名称
3. 需要查找的具体信息

## 输出格式：
请以 JSON 格式返回：
{{
    "keywords": ["关键词1", "关键词2"],
    "entities": ["实体1", "实体2"],
    "search_focus": "主要搜索焦点"
}}"""

    # ==================== 答案生成提示词 ====================
    
    ANSWER_GENERATION_FACTUAL = """你是一个专业的知识问答助手。请根据检索到的信息，回答用户的问题。

## 用户问题：
{query}

## 检索到的相关信息：
{context}

## 回答要求：
1. 直接回答问题，提供准确的事实信息
2. 如果信息来源明确，请标注引用
3. 如果检索信息不足以回答问题，请诚实说明
4. 保持回答简洁、准确

## 请回答："""

    ANSWER_GENERATION_EXPLANATORY = """你是一个专业的知识解释专家。请根据检索到的信息，为用户提供详细的解释。

## 用户问题：
{query}

## 检索到的相关信息：
{context}

## 回答要求：
1. 提供详细、全面的解释
2. 综合多个来源的信息，形成连贯的解释
3. 使用清晰的逻辑结构组织内容
4. 适当使用例子或类比帮助理解
5. 如果存在不同观点或方法，请进行比较说明
6. 标注信息来源

## 请回答："""

    ANSWER_GENERATION_REASONING = """你是一个专业的推理分析专家。请根据检索到的信息，进行逻辑推理并回答问题。

## 用户问题：
{query}

## 检索到的相关信息：
{context}

## 回答要求：
1. 展示清晰的推理步骤
2. 每个推理步骤都要有依据
3. 如果涉及假设，请明确说明
4. 考虑可能的反例或限制条件
5. 给出结论并说明置信度
6. 标注关键信息来源

## 推理过程：
请按以下格式回答：
1. 问题分析：[分析问题的核心]
2. 已知信息：[列出相关事实]
3. 推理步骤：
   - 步骤1：[推理内容]
   - 步骤2：[推理内容]
   ...
4. 结论：[最终答案]
5. 置信度：[高/中/低]"""

    ANSWER_GENERATION_WITH_HISTORY = """你是一个专业的知识问答助手。请根据对话历史和检索到的信息，回答用户的问题。

## 对话历史：
{history}

## 当前问题：
{query}

## 检索到的相关信息：
{context}

## 回答要求：
1. 考虑对话上下文，保持回答的连贯性
2. 直接回答问题，提供准确信息
3. 如果需要引用之前的内容，请明确指出
4. 标注信息来源

## 请回答："""

    # ==================== 打分评估提示词 ====================
    
    RELEVANCE_SCORING = """请评估检索内容与用户问题的相关性。

## 用户问题：
{query}

## 检索内容：
{content}

## 评分标准：
- 5分：完全相关，直接回答问题
- 4分：高度相关，包含大部分所需信息
- 3分：部分相关，包含一些有用信息
- 2分：低度相关，信息价值有限
- 1分：不相关，没有有用信息

## 请返回：
只返回一个数字（1-5），表示相关性分数。"""

    ANSWER_QUALITY_SCORING = """请评估答案的质量。

## 用户问题：
{query}

## 生成的答案：
{answer}

## 参考信息：
{reference}

## 评分维度：
1. 准确性（0-10）：答案是否准确无误
2. 完整性（0-10）：答案是否完整回答了问题
3. 相关性（0-10）：答案是否与问题高度相关
4. 清晰度（0-10）：答案是否清晰易懂

## 请以 JSON 格式返回：
{{
    "accuracy": 分数,
    "completeness": 分数,
    "relevance": 分数,
    "clarity": 分数,
    "overall": 总分,
    "feedback": "简要评价"
}}"""

    # ==================== 查询改写提示词 ====================
    
    QUERY_REWRITE = """请改写用户的问题，使其更适合检索。

## 原始问题：
{query}

## 改写要求：
1. 保持原意不变
2. 补充必要的上下文信息
3. 使用更精确的关键词
4. 去除无关的修饰词

## 请返回改写后的问题："""

    QUERY_EXPANSION = """请扩展用户的问题，生成多个相关的检索查询。

## 原始问题：
{query}

## 请生成：
1. 3-5 个不同角度的查询变体
2. 每个变体应该关注问题的不同方面
3. 变体应该有助于检索到更全面的信息

## 输出格式：
请以 JSON 数组格式返回：
["查询1", "查询2", "查询3"]"""

    QUERY_DECOMPOSITION = """请将复杂问题分解为多个简单的子问题。

## 原始问题：
{query}

## 分解要求：
1. 每个子问题应该独立可回答
2. 子问题的答案组合起来应该能回答原问题
3. 保持子问题之间的逻辑关系

## 输出格式：
请以 JSON 数组格式返回子问题列表：
["子问题1", "子问题2", "子问题3"]"""

    # ==================== 摘要生成提示词 ====================
    
    CONTEXT_SUMMARIZATION = """请对检索到的多个文档片段进行摘要整合。

## 检索到的文档片段：
{documents}

## 摘要要求：
1. 提取关键信息，去除冗余
2. 保持信息的准确性和完整性
3. 按逻辑顺序组织内容
4. 控制摘要长度在合理范围内

## 请生成摘要："""

    # ==================== 系统提示词 ====================
    
    SYSTEM_PROMPT_RAG = """你是一个专业的 RAG（检索增强生成）助手。你的任务是：
1. 理解用户的问题
2. 基于检索到的信息生成准确、有帮助的回答
3. 如果检索信息不足，诚实说明
4. 始终标注信息来源

你应该：
- 保持客观、专业
- 提供准确的信息
- 承认知识的局限性
- 引导用户提出更好的问题"""

    SYSTEM_PROMPT_AGENT = """你是一个智能问答 Agent。你可以使用以下工具来帮助回答问题：
{tools_description}

请根据用户的问题选择合适的工具，并提供准确的回答。"""


def get_template(name: str) -> str:
    """获取指定名称的提示词模板
    
    Args:
        name: 模板名称
        
    Returns:
        提示词模板字符串
        
    Raises:
        KeyError: 模板名称不存在
    """
    templates = {
        "intent_classification": PromptTemplates.INTENT_CLASSIFICATION,
        "intent_classification_with_context": PromptTemplates.INTENT_CLASSIFICATION_WITH_CONTEXT,
        "entity_extraction": PromptTemplates.ENTITY_EXTRACTION,
        "entity_extraction_for_query": PromptTemplates.ENTITY_EXTRACTION_FOR_QUERY,
        "answer_factual": PromptTemplates.ANSWER_GENERATION_FACTUAL,
        "answer_explanatory": PromptTemplates.ANSWER_GENERATION_EXPLANATORY,
        "answer_reasoning": PromptTemplates.ANSWER_GENERATION_REASONING,
        "answer_with_history": PromptTemplates.ANSWER_GENERATION_WITH_HISTORY,
        "relevance_scoring": PromptTemplates.RELEVANCE_SCORING,
        "answer_quality_scoring": PromptTemplates.ANSWER_QUALITY_SCORING,
        "query_rewrite": PromptTemplates.QUERY_REWRITE,
        "query_expansion": PromptTemplates.QUERY_EXPANSION,
        "query_decomposition": PromptTemplates.QUERY_DECOMPOSITION,
        "context_summarization": PromptTemplates.CONTEXT_SUMMARIZATION,
        "system_rag": PromptTemplates.SYSTEM_PROMPT_RAG,
        "system_agent": PromptTemplates.SYSTEM_PROMPT_AGENT,
    }
    
    if name not in templates:
        raise KeyError(f"模板 '{name}' 不存在。可用模板: {list(templates.keys())}")
    
    return templates[name]


def list_templates() -> Dict[str, str]:
    """列出所有可用的提示词模板
    
    Returns:
        模板名称和描述的字典
    """
    return {
        "intent_classification": "意图分类提示词",
        "intent_classification_with_context": "带上下文的意图分类提示词",
        "entity_extraction": "实体提取提示词",
        "entity_extraction_for_query": "查询实体提取提示词",
        "answer_factual": "事实型答案生成提示词",
        "answer_explanatory": "解释型答案生成提示词",
        "answer_reasoning": "推理型答案生成提示词",
        "answer_with_history": "带对话历史的答案生成提示词",
        "relevance_scoring": "相关性评分提示词",
        "answer_quality_scoring": "答案质量评分提示词",
        "query_rewrite": "查询改写提示词",
        "query_expansion": "查询扩展提示词",
        "query_decomposition": "问题分解提示词",
        "context_summarization": "上下文摘要提示词",
        "system_rag": "RAG 系统提示词",
        "system_agent": "Agent 系统提示词",
    }


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Prompt Templates 模块")
    print("=" * 50)
    
    # 测试 1: 获取模板
    templates = PromptTemplates()
    print(f"✓ 创建模板实例")
    
    # 测试 2: 获取意图分类模板
    intent_template = templates.get_template("intent_classification")
    print(f"✓ 获取意图分类模板: 长度={len(intent_template)}")
    print(f"  前50字符: {intent_template[:50]}...")
    
    # 测试 3: 获取实体提取模板
    entity_template = templates.get_template("entity_extraction")
    print(f"✓ 获取实体提取模板: 长度={len(entity_template)}")
    
    # 测试 4: 获取答案生成模板
    factual_template = templates.get_template("answer_factual")
    print(f"✓ 获取事实型答案模板: 长度={len(factual_template)}")
    
    explanatory_template = templates.get_template("answer_explanatory")
    print(f"✓ 获取解释型答案模板: 长度={len(explanatory_template)}")
    
    reasoning_template = templates.get_template("answer_reasoning")
    print(f"✓ 获取推理型答案模板: 长度={len(reasoning_template)}")
    
    # 测试 5: 获取评分模板
    relevance_template = templates.get_template("relevance_scoring")
    print(f"✓ 获取相关性评分模板: 长度={len(relevance_template)}")
    
    # 测试 6: 获取查询处理模板
    rewrite_template = templates.get_template("query_rewrite")
    print(f"✓ 获取查询改写模板: 长度={len(rewrite_template)}")
    
    expansion_template = templates.get_template("query_expansion")
    print(f"✓ 获取查询扩展模板: 长度={len(expansion_template)}")
    
    # 测试 7: 获取系统提示词
    system_rag = templates.get_template("system_rag")
    print(f"✓ 获取 RAG 系统提示词: 长度={len(system_rag)}")
    
    # 测试 8: 获取所有模板名称
    all_names = templates.get_all_template_names()
    print(f"✓ 所有模板名称: {len(all_names)} 个")
    
    # 测试 9: 获取模板描述
    descriptions = templates.get_template_descriptions()
    print(f"✓ 模板描述: {len(descriptions)} 个")
    
    print("\n所有测试通过!")
