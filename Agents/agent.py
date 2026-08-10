"""
Agents/agent.py
可执行的Agent类 - 支持LLM调用和RAG检索
"""

from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from config.settings import get_settings
from Agents.state import CritiqueResult, REAgentState, get_tool_registry
import json


async def summarize_messages(messages: List, max_summary_length: int = 2000) -> str:
    """
    使用 LLM 对消息历史进行智能总结，保留关键信息但大幅减少 tokens
    
    Args:
        messages: 要总结的消息列表
        max_summary_length: 总结的最大长度（字符数）
    
    Returns:
        总结后的文本
    """
    if not messages:
        return ""
    
    # 构建消息内容摘要（提取关键信息）
    message_contents = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue  # 跳过系统消息
        elif isinstance(msg, HumanMessage):
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            message_contents.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            message_contents.append(f"Assistant: {content}")
        elif isinstance(msg, ToolMessage):
            # 工具消息只保留工具名称和简要结果
            content = msg.content[:200] if len(msg.content) > 200 else msg.content
            message_contents.append(f"Tool Result: {content}")
    
    if not message_contents:
        return "No significant messages to summarize."
    
    # 使用 LLM 进行总结
    try:
        settings = get_settings()
        summarizer_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.3,  # 较低温度，更聚焦的总结
            openai_api_key=settings.openai_api_key,
            max_tokens=1000  # 限制总结长度
        )
        
        summary_prompt = f"""Please intelligently summarize the following conversation history, preserving all key information and decisions while significantly compressing the content.

Requirements:
1. Preserve all important task information, design decisions, and parameter settings
2. Preserve key conclusions and scores from expert analyses
3. Remove redundant and repetitive information
4. Compress detailed tool execution results, keeping only key information
5. Keep the summary within {max_summary_length} characters

Conversation history:
{chr(10).join(message_contents[:20])}  # Process at most the first 20 messages

Please provide a concise but complete summary:"""

        response = await summarizer_llm.ainvoke([HumanMessage(content=summary_prompt)])
        summary = response.content if response.content else ""
        
        # 确保总结不超过最大长度
        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length] + "..."
        
        return summary
    except Exception as e:
        # 如果总结失败，返回简单的文本摘要
        print(f"  ⚠️ 消息总结失败: {e}，使用简单摘要", flush=True)
        return f"对话历史摘要（{len(messages)}条消息）: " + "; ".join([msg.content[:100] for msg in messages[:5] if hasattr(msg, 'content')])


class Agent:
    """
    可执行的Agent类
    
    功能：
    1. LLM调用能力
    2. RAG知识检索
    3. 分析实验方案并生成CritiqueResult
    """

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str,
        rag_interface: Optional[Any] = None
    ) -> None:
        """
        初始化Agent
        
        Args:
            title: Agent标题
            expertise: 专业领域
            goal: 目标
            role: 角色描述
            model: LLM模型名称
            rag_interface: RAG接口（可选，延迟初始化）
        """
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self._rag_interface = rag_interface
        self._llm = None
        self._llm_with_tools = None
        self._tools = []
        self.settings = get_settings()

    def prompt(self) -> str:
        """生成系统提示"""
        # 检查是否有RAG工具可用
        has_rag_tool = any(tool.name == "rag_search" for tool in self._tools)
        
        tool_instructions = ""
        if has_rag_tool:
            tool_instructions = """
            
CRITICAL: You MUST use the `rag_search` tool to retrieve relevant knowledge from the knowledge base BEFORE providing your analysis.
- Call `rag_search` with a detailed query related to your expertise area and the task at hand
- Use the retrieved knowledge to inform your design decisions
- Cite specific knowledge sources in your analysis
- If the initial search doesn't return enough relevant results, refine your query and search again

You also have access to file reading/writing tools (`read_file`, `write_file`) if you need to examine data files or save intermediate results.
"""
        
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
            + tool_instructions
        )

    def message(self) -> dict[str, str]:
        """转换为消息格式"""
        return {
            "role": "system",
            "content": self.prompt(),
        }

    def _is_reasoning_model(self, model_name: str) -> bool:
        """判断是否是推理模型（需要更大的 max_tokens 配额）"""
        model_lower = model_name.lower()
        # 检查是否是 o1 系列或其他推理模型
        # gpt-5.1 和 gpt-5.2 都是推理模型，会使用 reasoning tokens
        reasoning_indicators = ["o1", "reasoning", "gpt-5.1", "gpt-5.2"]
        return any(indicator in model_lower for indicator in reasoning_indicators)
    
    def _get_max_tokens(self, model_name: str) -> int:
        """根据模型类型获取合适的 max_tokens 值"""
        if self._is_reasoning_model(model_name):
            # 推理模型：需要更大的配额（reasoning tokens + content tokens）
            # o1 系列通常需要 16000-32000
            # gpt-5.1/gpt-5.2 推理模型：如果使用了 16000 reasoning tokens，
            # 需要至少 32000-64000 的总配额来为内容 tokens 留出空间
            # 设置为 64000 以确保有足够空间生成完整响应
            return 64000  # 为推理 tokens 和内容 tokens 留出足够空间
        else:
            # 非推理模型：标准配额
            return 16000  # 增大标准配额，支持更详细的代码生成

    @property
    def llm(self) -> ChatOpenAI:
        """延迟初始化LLM"""
        if self._llm is None:
            max_tokens = self._get_max_tokens(self.model)
            temp = (
                self.settings.code_generator_temperature
                if self.role == "code_generator"
                else self.settings.default_temperature
            )
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=temp,
                openai_api_key=self.settings.openai_api_key,
                max_tokens=max_tokens  # 根据模型类型动态设置
            )
            if self._is_reasoning_model(self.model):
                print(f"   ℹ️ 推理模型检测到 ({self.model})，设置 max_tokens={max_tokens} 以支持推理 tokens", flush=True)
        return self._llm

    def set_tools(self, tools: List[BaseTool]):
        """设置工具列表"""
        self._tools = tools
        self._llm_with_tools = None  # 重置，下次访问时重新绑定

    @property
    def llm_with_tools(self) -> ChatOpenAI:
        """获取带工具绑定的LLM"""
        if self._llm_with_tools is None:
            if self._tools:
                self._llm_with_tools = self.llm.bind_tools(self._tools)
            else:
                self._llm_with_tools = self.llm
        return self._llm_with_tools

    def set_rag_interface(self, rag_interface: Any):
        """设置RAG接口"""
        self._rag_interface = rag_interface

    @property
    def rag_interface(self) -> Optional[Any]:
        """获取RAG接口"""
        return self._rag_interface

    async def search_knowledge(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        使用RAG检索相关知识（强制通过RAG工具链）
        
        优先通过 ToolRegistry 中的 `rag_search` 工具执行，
        这样可以在终端看到工具调用状态；如果工具不可用，
        则回退到直接使用底层 RAG 接口。
        消融实验：若Agent未分配rag_search工具，直接返回空（不检索）。
        """
        # 消融实验：Agent未分配rag_search时直接返回空
        if not any(getattr(t, "name", "") == "rag_search" for t in (self._tools or [])):
            return {"shared_knowledge": [], "specialized_knowledge": [], "total_results": 0}

        # 优先使用 RAGSearchTool（走工具链）
        try:
            tool_registry = get_tool_registry()
            rag_tool = tool_registry.get_tool("rag_search")
        except Exception:
            rag_tool = None

        if rag_tool is not None:
            try:
                tool_result = await rag_tool._arun(
                    query=query,
                    agent_role=self.role,
                    top_k=top_k,
                )
                if not tool_result.get("success", False):
                    print(f"⚠️ RAG工具检索失败 ({self.title}): {tool_result.get('error')}", flush=True)
                    return {
                        "shared_knowledge": [],
                        "specialized_knowledge": [],
                        "total_results": 0,
                    }

                data = tool_result.get("data", {}) or {}
                results = data.get("results", []) or []

                shared = [
                    {
                        "id": item.get("metadata", {}).get("doc_id", ""),  # 知识库条目ID
                        "content": str(item.get("content", "")),  # 完整内容，不截断
                        "source": item.get("metadata", {}).get("source", ""),
                        "title": item.get("metadata", {}).get("title", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                    for item in results
                ]

                return {
                    "shared_knowledge": shared,
                    "specialized_knowledge": [],
                    "total_results": int(data.get("total_results", len(shared))),
                }
            except Exception as e:
                print(f"⚠️ 通过RAG工具检索失败 ({self.title}): {e}", flush=True)
                # 继续回退到底层接口

        # 回退：直接使用底层 RAG 接口（兼容旧实现）
        if not self._rag_interface:
            return {
                "shared_knowledge": [],
                "specialized_knowledge": [],
                "total_results": 0,
            }

        try:
            results = await self._rag_interface.query(
                query=query,
                strategy=strategy,
                top_k=top_k,
            )

            return {
                "shared_knowledge": [
                    {
                        "id": r.document_id,  # 知识库条目ID
                        "content": r.content,  # 完整内容，不截断
                        "source": getattr(r, "source", ""),
                        "title": r.metadata.get("title", "") if hasattr(r, "metadata") and isinstance(r.metadata, dict) else "",
                        "score": getattr(r, "score", 0.0),
                    }
                    for r in results.shared_results
                ],
                "specialized_knowledge": [
                    {
                        "id": r.document_id,
                        "content": r.content,
                        "source": getattr(r, "source", ""),
                        "title": r.metadata.get("title", "") if hasattr(r, "metadata") and isinstance(r.metadata, dict) else "",
                        "score": getattr(r, "score", 0.0),
                    }
                    for r in getattr(results, "specialized_results", []) or []
                ],
                "total_results": len(results.shared_results)
                + len(getattr(results, "specialized_results", []) or []),
            }
        except Exception as e:
            print(f"⚠️ RAG检索失败 ({self.title}): {e}", flush=True)
            return {
                "shared_knowledge": [],
                "specialized_knowledge": [],
                "total_results": 0,
            }

    async def analyze(
        self,
        task_plan: Dict[str, Any],
        state: Optional[REAgentState] = None
    ) -> CritiqueResult:
        """
        分析任务方案并生成CritiqueResult
        
        Args:
            task_plan: 任务计划字典，包含：
                - title: 任务标题
                - description: 任务描述/背景
                - data_source: 数据源信息
                - methodology: 方法描述
                - model_architecture: 模型架构
                - evaluation_metrics: 评估指标
            state: 当前状态（可选）
        
        Returns:
            CritiqueResult对象
        """
        # 0. 检查并总结过长的消息历史（如果从 state 传入）
        if state and state.get("messages"):
            state_messages = state.get("messages", [])
            # 检查消息历史长度（粗略估算：1 token ≈ 4 字符）
            total_chars = sum(len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0 for msg in state_messages)
            estimated_tokens = total_chars / 4
            
            # 如果估计超过 200000 tokens，总结旧消息
            if estimated_tokens > 200000:
                print(f"  ⚠️ 消息历史过长，进行智能总结...", flush=True)
                # 保留最近的 10 条消息，总结之前的消息
                recent_messages = state_messages[-10:]
                old_messages = state_messages[:-10]
                
                if old_messages:
                    summary = await summarize_messages(old_messages, max_summary_length=2000)
                    # 创建总结消息
                    summary_message = HumanMessage(
                        content=f"[Previous Conversation Summary]\n{summary}\n\n[Continuing with recent messages...]"
                    )
                    # 用总结消息替换旧消息
                    state["messages"] = [summary_message] + recent_messages
                    print(f"  ✓ 消息历史已总结：{len(old_messages)} 条旧消息 -> 1 条总结消息", flush=True)
        
        # 1. 构建分析查询（包含具体的任务信息和场景细节）
        # 截断过长的字段以避免输入过大
        MAX_FIELD_LENGTH = 2000  # 每个字段最多2000字符
        
        def truncate_field(text, max_len=MAX_FIELD_LENGTH):
            if not text:
                return ""
            if len(text) <= max_len:
                return text
            return text[:max_len] + f"\n[... Content truncated, original length: {len(text)} characters ...]"
        
        title = truncate_field(task_plan.get("title", ""), 500)
        description = truncate_field(task_plan.get("description", ""), MAX_FIELD_LENGTH)
        data_source = truncate_field(task_plan.get("data_source", ""), MAX_FIELD_LENGTH)
        methodology = truncate_field(task_plan.get("methodology", ""), MAX_FIELD_LENGTH)
        model_arch = truncate_field(task_plan.get("model_architecture", ""), MAX_FIELD_LENGTH)
        eval_metrics = truncate_field(task_plan.get("evaluation_metrics", ""), MAX_FIELD_LENGTH)
        code_instructions = truncate_field(task_plan.get("code_instructions", ""), MAX_FIELD_LENGTH)
        
        # 提取数据集的关键信息（从data_source中）
        dataset_details = ""
        if data_source:
            # 尝试提取文件路径、数据类型、字段等信息
            if "File path:" in data_source:
                dataset_details = data_source.split("File path:")[1].split(";")[0].strip()
            if "Data type:" in data_source:
                data_type = data_source.split("Data type:")[1].split(";")[0].strip()
                dataset_details += f" {data_type}"
            if "Input features:" in data_source:
                features = data_source.split("Input features:")[1].split(";")[0].strip()
                dataset_details += f" features: {features}"
            if "Target variable:" in data_source:
                target = data_source.split("Target variable:")[1].split(";")[0].strip()
                dataset_details += f" target: {target}"
        
        # 根据Agent角色构建精简且聚焦的查询（突出角色特点）
        if self.role == "data_management":
            query = f"""Data preprocessing and management: dataset type analysis (MPRA/RNA-seq/ChIP-seq/ATAC-seq), 
dataset size-based preprocessing strategies (large datasets: quality control and cleaning, 
small datasets: data augmentation), train/validation/test split, quality control procedures."""
        elif self.role == "methodology":
            query = f"""Training methodology: loss function design, optimization algorithms, regularization strategies, 
prior knowledge integration (motifs, PWMs), data augmentation, training pipeline workflow."""
        elif self.role == "model_architect":
            query = f"""Neural network architecture: dataset size-based complexity control (small datasets: compact architectures, 
large datasets: expressive architectures), innovative architecture design (attention, residual connections, multi-scale), 
parameter count estimation, robustness and generalization."""
        elif self.role == "result_analyst":
            query = f"""Evaluation and analysis: evaluation metrics, statistical testing, validation strategy 
(cross-validation, held-out test, external validation), biological validation methods, result interpretation."""
        else:
            query = f"""Experimental design: {title}"""

        # 2. 使用RAG检索相关知识
        knowledge = await self.search_knowledge(query, strategy="hybrid", top_k=5)

        # 3. 构建分析提示
        knowledge_context = self._format_knowledge_context(knowledge)
        
        # 检查是否有Supervisor提供的专门任务prompt
        supervisor_task_prompt = task_plan.get("task_prompt", "")
        
        # 根据Agent角色构建不同的设计提示
        if self.role == "data_management":
            design_focus = "data usage plan"
            design_sections = [
                "1. Dataset characteristic analysis (dataset type: MPRA/RNA-seq/ChIP-seq/ATAC-seq/etc., sequence length distribution, dataset size and sample count, feature dimensions)",
                "2. Data source selection and justification",
                "3. Data preprocessing pipeline tailored to dataset characteristics (e.g., aggressive cleaning for large datasets, augmentation strategies for small datasets)",
                "4. Train/validation/test split strategy considering dataset size",
                "5. Data augmentation methods appropriate for dataset type and size",
                "6. Quality control procedures",
                "7. Bias mitigation strategies"
            ]
        elif self.role == "methodology":
            design_focus = "training methodology"
            design_sections = [
                "1. Loss function design and rationale",
                "2. Optimization algorithm and hyperparameters",
                "3. Regularization strategies",
                "4. Prior knowledge integration (motifs, PWMs, etc.)",
                "5. Data augmentation approaches",
                "6. Training pipeline workflow"
            ]
        elif self.role == "model_architect":
            design_focus = "model architecture"
            design_sections = [
                "1. Dataset size analysis and model complexity strategy (evaluate dataset size to determine appropriate parameter count and architecture complexity)",
                "2. Architecture type selection and rationale (encourage innovative and effective designs: attention mechanisms, residual connections, multi-scale convolutions, etc.)",
                "3. Detailed layer-by-layer design with parameter count justification",
                "4. Parameter count estimation and complexity control (flexibly adjust based on data volume: smaller models for limited data, larger models for abundant data)",
                "5. Long-range dependency modeling mechanisms",
                "6. Robustness considerations (generalization strategies, regularization integration, overfitting prevention)",
                "7. Interpretability features",
                "8. Computational efficiency considerations"
            ]
        elif self.role == "result_analyst":
            design_focus = "result analysis plan"
            design_sections = [
                "1. Evaluation metric suite selection",
                "2. Statistical testing design",
                "3. Validation strategy (cross-validation, held-out test, external validation)",
                "4. Biological validation methods",
                "5. Result interpretation framework",
                "6. Summary and reporting format"
            ]
        else:
            design_focus = "experimental design"
            design_sections = ["1. Design recommendations"]
        
        # 为 detailed_design 明确要求结构化输出，避免为空
        detailed_keys = [
            section.split(". ")[1].lower().replace(" ", "_")
            for section in design_sections
        ]
        detailed_keys_str = ", ".join(f'"{k}"' for k in detailed_keys)

        # 检查是否有RAG工具，如果有则强制要求使用
        has_rag_tool = any(tool.name == "rag_search" for tool in self._tools)
        rag_requirement = ""
        if has_rag_tool:
            # 如果预检索的知识很少或为空，强制要求使用工具
            total_knowledge = knowledge.get("total_results", 0)
            if total_knowledge < 3:
                rag_requirement = f"""
            
⚠️ CRITICAL: Pre-retrieved knowledge is insufficient ({total_knowledge} results). You MUST use the `rag_search` tool to search for relevant knowledge BEFORE providing your analysis.
- Call `rag_search` with query: "{query}" or a more specific query related to your expertise
- Use the retrieved knowledge to inform every aspect of your design
- Reference specific methods, techniques, or findings from the knowledge base in your detailed_design sections
- If the retrieved knowledge is still insufficient, refine your query and search again with different keywords
- DO NOT proceed with analysis without first retrieving sufficient knowledge from the RAG system
"""
            else:
                rag_requirement = f"""
            
⚠️ IMPORTANT: While some knowledge has been pre-retrieved, you SHOULD also use the `rag_search` tool to search for additional relevant knowledge if needed.
- You can call `rag_search` with query: "{query}" or refine it with more specific terms
- Use ALL retrieved knowledge (pre-retrieved + tool-retrieved) to inform your design
- Reference specific methods, techniques, or findings from the knowledge base in your detailed_design sections
"""
        
        # 获取数据集统计信息（如果Supervisor已读取）
        dataset_stats = task_plan.get("dataset_statistics", {})
        dataset_stats_text = ""
        if dataset_stats:
            dataset_stats_text = "\n\n📊 Dataset Statistics (read by Supervisor):\n"
            for file_path, stats in dataset_stats.items():
                dataset_stats_text += f"File: {file_path}\n"
                dataset_stats_text += f"  - Number of rows (samples): {stats.get('num_rows', 'N/A')}\n"
                dataset_stats_text += f"  - Number of columns (features): {stats.get('num_cols', 'N/A')}\n"
                col_names = stats.get('column_names', [])
                if col_names:
                    # 限制列名显示：最多显示前20个，避免过长
                    MAX_COL_NAMES = 20
                    if len(col_names) > MAX_COL_NAMES:
                        col_names_str = ', '.join(col_names[:MAX_COL_NAMES]) + f" ... (and {len(col_names) - MAX_COL_NAMES} more columns)"
                    else:
                        col_names_str = ', '.join(col_names)
                    dataset_stats_text += f"  - Column names ({len(col_names)}): {col_names_str}\n"
            
            # 添加数据集特点分析提示（仅对data_management角色）
            if self.role == "data_management":
                dataset_stats_text += "\n⚠️ CRITICAL DATASET CHARACTERISTIC ANALYSIS REQUIRED:\n"
                dataset_stats_text += "You MUST analyze the following dataset characteristics:\n"
                dataset_stats_text += "1. Dataset Type: Identify from column names and task description (MPRA, RNA-seq, ChIP-seq, ATAC-seq, STARR-seq, etc.)\n"
                dataset_stats_text += "2. Sequence Length: Analyze sequence length distribution if sequence columns exist (mean, median, range, outliers)\n"
                dataset_stats_text += "3. Dataset Size: Evaluate number of samples to determine if dataset is small (<1K), medium (1K-10K), or large (>10K)\n"
                dataset_stats_text += "4. Data Volume: Assess feature dimensions and data sparsity\n"
                dataset_stats_text += "5. Preprocessing Strategy Selection:\n"
                dataset_stats_text += "   - Large datasets (>10K samples): Focus on aggressive quality control, cleaning, filtering, outlier removal\n"
                dataset_stats_text += "   - Small datasets (<1K samples): Prioritize data augmentation techniques (sequence augmentation, synthetic data generation)\n"
                dataset_stats_text += "   - Medium datasets (1K-10K): Balance between quality control and augmentation\n"
                dataset_stats_text += "6. Link dataset characteristics to specific preprocessing parameters (quality thresholds, augmentation ratios, filtering criteria)\n"
            elif self.role == "model_architect":
                dataset_stats_text += "\n⚠️ CRITICAL MODEL COMPLEXITY CONTROL REQUIRED:\n"
                dataset_stats_text += "You MUST analyze dataset size and flexibly control model parameters and complexity:\n"
                dataset_stats_text += "1. Dataset Size Analysis: Evaluate number of samples to determine appropriate model complexity\n"
                dataset_stats_text += "2. Model Complexity Strategy:\n"
                dataset_stats_text += "   - Small datasets (<1K samples): Design compact architectures with fewer parameters, strong regularization, innovative architectural choices\n"
                dataset_stats_text += "   - Medium datasets (1K-10K samples): Design moderate complexity architectures with balanced parameter counts\n"
                dataset_stats_text += "   - Large datasets (>10K samples): Design more expressive architectures with higher capacity while maintaining efficiency\n"
                dataset_stats_text += "3. Parameter Count Justification: Provide clear rationale linking dataset size to parameter count decisions\n"
                dataset_stats_text += "4. Innovative Architecture Design: Encourage effective architectural choices (attention mechanisms, residual connections, multi-scale features, etc.)\n"
                dataset_stats_text += "5. Robustness: Ensure generalization strategies, overfitting prevention, and regularization integration\n"
        
        # 截断 supervisor_task_prompt 如果过长
        if supervisor_task_prompt and supervisor_task_prompt.strip():
            supervisor_task_prompt = truncate_field(supervisor_task_prompt, MAX_FIELD_LENGTH)
        
        # 如果Supervisor提供了专门的任务prompt，优先使用它
        if supervisor_task_prompt and supervisor_task_prompt.strip():
            task_instruction = f"""
Supervisor's Specific Task Assignment for You:
{supervisor_task_prompt}

Task Context:
Title: {title}
Background/Description: {description}
Data Source: {data_source}
Methodology: {methodology or "To be designed"}
Model Architecture: {model_arch or "To be designed"}
Evaluation Metrics: {eval_metrics or "To be designed"}
{dataset_stats_text}
"""
        else:
            # 使用通用的任务描述
            task_instruction = f"""
Task Information:
Title: {title}
Background/Description: {description}
Data Source: {data_source}
Methodology: {methodology or "To be designed"}
Model Architecture: {model_arch or "To be designed"}
Evaluation Metrics: {eval_metrics or "To be designed"}
{dataset_stats_text}

Your task is to design a comprehensive {design_focus} based on the task information above.
"""
        
        analysis_prompt = f"""You are designing an experimental plan for gene regulatory element design.

⚠️ CRITICAL LANGUAGE REQUIREMENT:
- You MUST write ALL responses in English (EN). Do NOT use Chinese, Japanese, or any other language.
- All text in "design_summary", "detailed_design", "strengths", "potential_issues", "recommendations" MUST be in English.
- This is a strict requirement for international publication standards.

{task_instruction}

Relevant Knowledge from Literature (pre-retrieved):
{knowledge_context}
{rag_requirement}

⚠️ DETAILED ANALYSIS REQUIREMENTS:
1. You MUST provide EXTENSIVE and DETAILED design plan - focus on comprehensive design specifications, not code implementation
2. For DATA MANAGEMENT role, you MUST FIRST conduct comprehensive dataset characteristic analysis:
   - Identify dataset type (MPRA, RNA-seq, ChIP-seq, ATAC-seq, STARR-seq, etc.) and explain its implications for preprocessing
   - Analyze sequence length distribution (mean, median, range, outliers) and its impact on preprocessing choices
   - Evaluate dataset size (number of samples) and determine if it's small (<1K), medium (1K-10K), or large (>10K)
   - Assess data volume and feature dimensions to guide preprocessing strategy
   - CRITICALLY link dataset characteristics to preprocessing strategy selection:
     * Large datasets (>10K samples): Focus on aggressive quality control, cleaning, filtering, outlier removal
     * Small datasets (<1K samples): Prioritize data augmentation techniques (sequence augmentation, synthetic data generation, etc.)
     * Medium datasets (1K-10K): Balance between quality control and augmentation
   - Provide specific preprocessing parameters based on dataset type and size (e.g., quality thresholds, augmentation ratios, filtering criteria)
3. Each section in "detailed_design" MUST contain:
   - At least 5-7 detailed sentences explaining the design approach and rationale
   - SPECIFIC parameter values, hyperparameter settings, and configuration details (this is CRITICAL)
   - Detailed model architecture specifications (for model architect: layer dimensions, activation functions, dropout rates, batch normalization settings, etc.)
   - Algorithm choices with detailed justification and parameter settings
   - Step-by-step design specifications and workflow
   - References to relevant methods from the knowledge base (if available)
   - Concrete design examples or use cases
4. For MODEL ARCHITECT role, you MUST provide EXTENSIVE parameter design details:
   - Layer-by-layer architecture with exact dimensions (input/output sizes, kernel sizes, stride, padding)
   - All hyperparameters: learning rate, batch size, dropout rates, weight decay, optimizer parameters
   - Activation functions and their parameters (e.g., LeakyReLU negative_slope, ELU alpha)
   - Regularization parameters (L1/L2 coefficients, dropout probabilities, batch norm momentum)
   - Initialization strategies and their parameters (e.g., Xavier/Glorot initialization parameters)
   - Training configuration (epochs, early stopping criteria, learning rate schedule parameters)
   - Model capacity estimation (total parameter count, FLOPs if applicable)
4. "strengths" and "potential_issues" should be BRIEF (1-2 items each, focus on critical points only)
5. "recommendations" MUST be concrete, actionable design improvements (at least 3-5 items) with specific parameter suggestions or configuration details

⚠️ STRICT SCORING CRITERIA (You MUST be strict and critical):
- Score 9.0-10.0: Design is EXCELLENT - comprehensive design specifications with detailed parameter values, all requirements met, minimal issues
- Score 8.5-8.9: Design is GOOD but has some gaps - mostly complete with most parameters specified, minor improvements needed
- Score 8.0-8.4: Design is ACCEPTABLE but needs refinement - missing some parameter details, moderate improvements needed
- Score < 8.0: Design needs SIGNIFICANT improvement - major gaps in parameter specifications, substantial design details missing

You MUST return a valid JSON object with ALL of the following top-level fields:
- "score": a float in [0,10] evaluating the feasibility and quality of your design (BE STRICT - only give 8.5+ if design is truly comprehensive with detailed parameters)
- "design_summary": a comprehensive summary (at least 5-7 sentences) of your design approach
- "detailed_design": an object containing detailed design specifications with parameter values (THIS IS THE MOST IMPORTANT PART)
- "strengths": a brief list of 1-2 key strengths (keep it concise)
- "potential_issues": a brief list of 1-2 critical potential issues (keep it concise)
- "recommendations": a list of at least 3-5 concrete, actionable design recommendations with specific parameter suggestions or configuration details
- "confidence": a float in [0,1] representing your confidence

IMPORTANT: Only give a score >= 8.5 if:
1. All design sections include detailed parameter specifications and values
2. All required design sections are fully detailed with comprehensive specifications
3. Design plan is comprehensive and actionable with clear parameter settings
4. For model architect: architecture specifications include exact dimensions, hyperparameters, and all configuration details
5. Design addresses all aspects of the task requirements with specific parameter values

If any of these are missing or incomplete, score should be < 8.5.

The "detailed_design" field MUST be a JSON object with ALL of these keys (do NOT leave them empty or omit them):
- {detailed_keys_str}

For each key in "detailed_design", write 5-7 DETAILED sentences describing your design for that aspect. Include:
- Technical rationale and justification
- SPECIFIC parameter values, hyperparameter settings, and configuration details (this is CRITICAL)
- Detailed design specifications (for model architect: exact layer dimensions, activation parameters, regularization settings, etc.)
- Step-by-step design workflow and configuration
- References to relevant knowledge from the literature (if available)
- Design examples or use cases with parameter values

Return ONLY the JSON object, no extra text.

Focus on your area of expertise: {self.expertise}
Your goal: {self.goal}
"""

        # 4. 调用LLM进行分析（支持工具调用）
        try:
            messages = [
                SystemMessage(content=self.prompt()),
                HumanMessage(content=analysis_prompt)
            ]
            
            # 使用支持工具调用的LLM，使Agent在需要时可以调用RAG、文件读写等工具
            response = await self.llm_with_tools.ainvoke(messages)
            
            # 处理工具调用：如果LLM返回工具调用，执行工具并继续对话
            max_tool_iterations = 3  # 最多允许3轮工具调用
            tool_iteration = 0
            
            # 检查是否有工具调用（兼容不同版本的LangChain）
            has_tool_calls = False
            if hasattr(response, "tool_calls") and response.tool_calls:
                has_tool_calls = True
            elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
                has_tool_calls = True
            
            while tool_iteration < max_tool_iterations and has_tool_calls:
                tool_iteration += 1
                print(f"  🔧 {self.title} 正在使用工具 (第 {tool_iteration} 轮)...", flush=True)
                
                # 检查消息长度，如果过长则总结
                total_chars = sum(len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0 for msg in messages)
                estimated_tokens = total_chars / 4
                if estimated_tokens > 200000:
                    print(f"  ⚠️ 消息历史过长，进行智能总结...", flush=True)
                    # 保留系统消息、最近的响应和最近的工具结果
                    system_msg = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
                    recent_messages = messages[-5:] if len(messages) > 5 else messages[1:]  # 保留最近5条（排除系统消息）
                    old_messages = messages[1:-5] if len(messages) > 6 else []
                    
                    if old_messages:
                        summary = await summarize_messages(old_messages, max_summary_length=1500)
                        summary_msg = HumanMessage(
                            content=f"[Previous Tool Call History Summary]\n{summary}\n\n[Continuing with recent messages...]"
                        )
                        # 重建消息列表
                        if system_msg:
                            messages = [system_msg, summary_msg] + recent_messages
                        else:
                            messages = [summary_msg] + recent_messages
                        print(f"  ✓ 消息历史已总结：{len(old_messages)} 条旧消息 -> 1 条总结消息", flush=True)
                
                # 添加LLM响应到消息历史
                messages.append(response)
                
                # 执行所有工具调用
                tool_results = []
                tool_calls_list = []
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_calls_list = response.tool_calls
                elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
                    tool_calls_list = response.additional_kwargs["tool_calls"]
                
                # 逐个工具调用并检查：一旦前序失败，后续全部跳过（但仍返回“skipped”结果，保证 tool_call_id 配对完整）
                should_continue_tools = True

                # 工具结果统一用 JSON 字符串承载（避免 python dict 字符串不稳定/不可解析）
                def _to_json_text(obj: Any, max_chars: int = 20000) -> str:
                    try:
                        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
                    except Exception:
                        text = str(obj)
                    if len(text) > max_chars:
                        return text[:max_chars] + f"\n\n[TRUNCATED: original_length={len(text)}]"
                    return text

                for tool_call in tool_calls_list:
                    # 处理不同格式的工具调用
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "") or tool_call.get("function", {}).get("name", "")
                        tool_args = tool_call.get("args", {}) or tool_call.get("function", {}).get("arguments", {})
                        tool_call_id = tool_call.get("id", "") or tool_call.get("function", {}).get("id", "")
                        # 如果args是字符串（JSON格式），解析它
                        if isinstance(tool_args, str):
                            import json
                            try:
                                tool_args = json.loads(tool_args)
                            except:
                                tool_args = {}
                    else:
                        # 如果是对象，尝试获取属性
                        tool_name = getattr(tool_call, "name", "") or (getattr(tool_call, "function", None) and getattr(tool_call.function, "name", ""))
                        tool_args = getattr(tool_call, "args", {}) or (getattr(tool_call, "function", None) and getattr(tool_call.function, "arguments", {}))
                        tool_call_id = getattr(tool_call, "id", "")
                        if isinstance(tool_args, str):
                            import json
                            try:
                                tool_args = json.loads(tool_args)
                            except:
                                tool_args = {}
                    
                    # 确保 tool_call_id 不为空，如果为空则生成一个
                    if not tool_call_id:
                        import uuid
                        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
                        print(f"    ⚠️ Tool call missing ID, generated: {tool_call_id}", flush=True)

                    # 若前序工具失败，则不再实际调用后续工具，直接返回 skipped
                    if not should_continue_tools:
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "name": tool_name or "unknown",
                            "content": _to_json_text({
                                "tool_name": tool_name or "unknown",
                                "success": False,
                                "skipped": True,
                                "reason": "Skipped because a previous tool call failed in the same tool_calls batch.",
                            })
                        })
                        continue
                    
                    # 如果 tool_name 为空，仍然需要创建 ToolMessage 以避免错误
                    if not tool_name:
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "name": "unknown",
                            "content": f"Tool call has no name. Original tool_call: {str(tool_call)[:200]}"
                        })
                        should_continue_tools = False
                        continue
                    
                    # 查找对应的工具
                    tool = next((t for t in self._tools if t.name == tool_name), None)
                    if tool:
                        try:
                            # 执行工具（支持异步）
                            if hasattr(tool, "_arun"):
                                result = await tool._arun(**tool_args)
                            else:
                                result = tool._run(**tool_args)
                            
                            if isinstance(result, dict) and result.get("success"):
                                tool_results.append({
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": _to_json_text({
                                        "tool_name": tool_name,
                                        "success": True,
                                        "data": result.get("data"),
                                        "metadata": result.get("metadata"),
                                    })
                                })
                                # 如果是RAG工具，更新knowledge_context
                                if tool_name == "rag_search" and isinstance(result.get("data"), dict):
                                    rag_results = result.get("data", {})
                                    # 合并到现有knowledge中
                                    existing_shared = knowledge.get("shared_knowledge", [])
                                    new_shared = rag_results.get("results", [])
                                    # 去重并合并
                                    all_sources = {item.get("metadata", {}).get("source", "") for item in existing_shared}
                                    for item in new_shared:
                                        item_source = item.get("metadata", {}).get("source", "")
                                        if item_source not in all_sources:
                                            existing_shared.append({
                                                "content": str(item.get("content", ""))[:500],
                                                "source": item_source,
                                                "score": float(item.get("score", 0.0)),
                                            })
                                            all_sources.add(item_source)
                                    knowledge["shared_knowledge"] = existing_shared[:10]  # 最多保留10条
                                    knowledge["total_results"] = len(existing_shared)
                                    knowledge_context = self._format_knowledge_context(knowledge)
                            else:
                                tool_results.append({
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": _to_json_text({
                                        "tool_name": tool_name,
                                        "success": (result.get("success") if isinstance(result, dict) and "success" in result else None),
                                        "result": result,
                                    })
                                })
                                # 标记失败：后续工具将被跳过（逐个检查调用）
                                should_continue_tools = False
                        except Exception as e:
                            print(f"    ⚠️ 工具 '{tool_name}' 执行失败: {e}", flush=True)
                            tool_results.append({
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": _to_json_text({
                                    "tool_name": tool_name,
                                    "success": False,
                                    "error": str(e),
                                })
                            })
                            should_continue_tools = False
                    else:
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": _to_json_text({
                                "tool_name": tool_name,
                                "success": False,
                                "error": "Tool not found",
                            })
                        })
                        should_continue_tools = False
                
                # 添加工具结果到消息历史（LangChain格式）
                # 确保每个 tool_call 都有对应的 ToolMessage
                processed_tool_call_ids = set()
                for tool_result in tool_results:
                    tool_call_id = tool_result.get("tool_call_id", "")
                    if tool_call_id:
                        messages.append(ToolMessage(
                            content=tool_result["content"],
                            tool_call_id=tool_call_id
                        ))
                        processed_tool_call_ids.add(tool_call_id)
                
                # 检查是否有遗漏的 tool_call_id（从 response 中获取所有 tool_call_ids）
                all_tool_call_ids = set()
                for tool_call in tool_calls_list:
                    if isinstance(tool_call, dict):
                        call_id = tool_call.get("id", "") or tool_call.get("function", {}).get("id", "")
                    else:
                        call_id = getattr(tool_call, "id", "")
                    if call_id:
                        all_tool_call_ids.add(call_id)
                
                # 为任何遗漏的 tool_call_id 创建空的 ToolMessage
                missing_ids = all_tool_call_ids - processed_tool_call_ids
                if missing_ids:
                    print(f"    ⚠️ 发现遗漏的 tool_call_ids: {missing_ids}，创建空响应", flush=True)
                    for missing_id in missing_ids:
                        messages.append(ToolMessage(
                            content="Tool execution was skipped or failed to generate response.",
                            tool_call_id=missing_id
                    ))
                
                # 如果knowledge_context已更新，更新prompt
                if knowledge_context != self._format_knowledge_context(knowledge):
                    # 重新构建prompt（简化版，只添加更新的knowledge）
                    update_message = HumanMessage(
                        content=f"Updated knowledge context:\n{knowledge_context}\n\nPlease continue your analysis based on this updated knowledge."
                    )
                    messages.append(update_message)
                
                # 继续调用LLM
                response = await self.llm_with_tools.ainvoke(messages)
                
                # 更新has_tool_calls标志
                has_tool_calls = False
                if hasattr(response, "tool_calls") and response.tool_calls:
                    has_tool_calls = True
                elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
                    has_tool_calls = True
            
            # 获取最终的分析文本
            analysis_text = response.content
            if not analysis_text and (hasattr(response, "tool_calls") and response.tool_calls or 
                                      (hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"))):
                # 如果只有工具调用没有文本，提示LLM生成分析
                messages.append(response)
                messages.append(HumanMessage(content="Please provide your analysis based on the tool results above. Return the JSON object as required."))
                response = await self.llm_with_tools.ainvoke(messages)
                analysis_text = response.content

            # 5. 解析LLM响应
            critique_data = self._parse_analysis_response(analysis_text)

            # 6. 构建CritiqueResult（包含设计详情和检索到的知识条目）
            # 提取检索到的知识条目（id + 内容），用于可解释性
            retrieved_knowledge_items = []
            shared_knowledge = knowledge.get("shared_knowledge", [])
            specialized_knowledge = knowledge.get("specialized_knowledge", [])
            
            for item in shared_knowledge + specialized_knowledge:
                retrieved_knowledge_items.append({
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),  # 完整内容
                    "source": item.get("source", ""),
                    "relevance_score": item.get("score", 0.0)
                })
            
            return CritiqueResult(
                agent_role=self.role,
                score=critique_data.get("score", 5.0),
                strengths=critique_data.get("strengths", []),
                weaknesses=critique_data.get("potential_issues", critique_data.get("weaknesses", [])),
                recommendations=critique_data.get("recommendations", []),
                confidence=critique_data.get("confidence", 0.5),
                metadata={
                    "query": query,
                    "knowledge_results": knowledge.get("total_results", 0),
                    "retrieved_knowledge": retrieved_knowledge_items,  # 检索到的知识条目（id + 内容）
                    "model": self.model,
                    "design_summary": critique_data.get("design_summary", ""),
                    "detailed_design": critique_data.get("detailed_design", {})
                }
            )

        except Exception as e:
            print(f"⚠️ Agent分析失败 ({self.title}): {e}")
            # 返回默认结果
            return CritiqueResult(
                agent_role=self.role,
                score=0.0,
                strengths=[],
                weaknesses=[f"分析过程中出错: {str(e)}"],
                recommendations=["请检查输入数据格式"],
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _format_knowledge_context(self, knowledge: Dict[str, Any]) -> str:
        """格式化知识上下文"""
        context_parts = []
        
        # 限制每条知识内容的长度，避免过长
        MAX_CONTENT_LENGTH = 3000  # 每条知识最多3000字符

        shared = knowledge.get("shared_knowledge", [])
        if shared:
            context_parts.append("Shared Knowledge (from RAG retrieval):")
            for i, item in enumerate(shared[:5], 1):  # 最多5条
                content = item.get('content', '')
                # 截断过长的内容
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + f"\n[... Content truncated, original length: {len(content)} characters ...]"
                source = item.get('source', 'Unknown')
                score = item.get('score', 0.0)
                context_parts.append(f"{i}. [{source}] (relevance score: {score:.3f})\n   {content}")

        specialized = knowledge.get("specialized_knowledge", [])
        if specialized:
            context_parts.append("\nSpecialized Knowledge:")
            for i, item in enumerate(specialized[:5], 1):  # 最多5条
                content = item.get('content', '')
                # 截断过长的内容
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + f"\n[... Content truncated, original length: {len(content)} characters ...]"
                source = item.get('source', 'Unknown')
                score = item.get('score', 0.0)
                context_parts.append(f"{i}. [{source}] (relevance score: {score:.3f})\n   {content}")

        if not context_parts:
            return "⚠️ No relevant knowledge found in the knowledge base. You MUST use the rag_search tool to retrieve knowledge before proceeding."

        return "\n".join(context_parts)

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """解析LLM的分析响应"""
        import json
        import re

        # 尝试提取JSON
        try:
            # 查找JSON代码块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # 查找普通JSON对象
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

        except json.JSONDecodeError:
            pass

        # 降级：从文本中提取信息
        result = {
            "score": 5.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "confidence": 0.5,
            "design_summary": "",
            "detailed_design": {}
        }

        # 尝试提取分数
        score_match = re.search(r'score["\']?\s*[:=]\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
        if score_match:
            try:
                result["score"] = float(score_match.group(1))
            except:
                pass

        # 提取列表项
        for key in ["strengths", "weaknesses", "recommendations"]:
            pattern = rf'{key}["\']?\s*[:=]\s*\[(.*?)\]'
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                items = re.findall(r'"([^"]+)"', match.group(1))
                result[key] = items

        # 尝试提取design_summary
        summary_match = re.search(r'"design_summary"\s*:\s*"([^"]+)"', response_text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            result["design_summary"] = summary_match.group(1)
        
        # 尝试提取detailed_design（这是一个对象，比较复杂）
        # 先尝试找到detailed_design的开始和结束
        detailed_match = re.search(r'"detailed_design"\s*:\s*(\{.*?\})', response_text, re.DOTALL | re.IGNORECASE)
        if detailed_match:
            try:
                # 尝试解析detailed_design对象
                detailed_str = detailed_match.group(1)
                result["detailed_design"] = json.loads(detailed_str)
            except:
                # 如果解析失败，至少保留原始文本
                result["detailed_design"] = {"raw_text": detailed_match.group(1)}

        return result

    def __hash__(self) -> int:
        return hash(self.title)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return False

        return (
            self.title == other.title
            and self.expertise == other.expertise
            and self.goal == other.goal
            and self.role == other.role
            and self.model == other.model
        )

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return self.title
