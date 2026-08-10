from pathlib import Path
from typing import Dict, Optional, List
from Agents.agent import Agent
from RAG.rag import HybridRAGSystem, AgentRAGInterface
from Agents.state import get_tool_registry
from config.settings import get_settings

settings = get_settings()
DEFAULT_MODEL = settings.llm_model  # 使用 settings.py 中的配置
CODE_MODEL = settings.code_model  # 代码生成专用模型


# ==================== Supervisor Agent ====================

SUPERVISOR_AGENT = Agent(
    title="Supervisor Agent",
    expertise=(
        "Read and understand human-provided task background, requirements, and dataset descriptions. "
        "Provide critical questions and evaluations. Synthesize expert agent analyses into a "
        "comprehensive experimental design report covering data usage, method design, model design, "
        "and result summary."
    ),
    goal=(
        "Read task background and dataset information provided by humans, identify unclear or "
        "missing information and raise critical questions, evaluate the completeness and feasibility "
        "of the task, coordinate four expert agents to provide design suggestions from their "
        "respective perspectives, and finally generate a detailed experimental design report that "
        "includes: (1) Data Usage Plan, (2) Method Design, (3) Model Design, and (4) Result Summary."
    ),
    role=(
        "Act as the supervisor and report generator: carefully read and understand human-provided "
        "task information, identify ambiguities and raise questions, evaluate task feasibility, "
        "coordinate expert agents, synthesize their design suggestions, and generate a comprehensive "
        "experimental design report. The report should be detailed, actionable, and cover all aspects "
        "of the experimental design."
    ),
    model=DEFAULT_MODEL,
)


# ==================== Expert Agents ====================

DATA_MANAGEMENT_AGENT = Agent(
    title="Data Management Expert",
    expertise=(
        "Design data usage plans for gene regulatory element design experiments, including: "
        "comprehensive dataset characteristic analysis (dataset type: MPRA/RNA-seq/ChIP-seq/ATAC-seq/etc., "
        "sequence length distribution, dataset size and sample count), data source selection and evaluation, "
        "data preprocessing strategies tailored to dataset characteristics (e.g., aggressive cleaning for "
        "large datasets, augmentation for small datasets), train/validation/test split design, "
        "data augmentation approaches, quality control measures, and handling of potential biases "
        "(species, experimental conditions, sequencing platforms, etc.)."
    ),
    goal=(
        "Based on the task background and dataset information provided, FIRST thoroughly analyze dataset "
        "characteristics including: (1) dataset type identification (MPRA, RNA-seq, ChIP-seq, ATAC-seq, "
        "etc.) and its implications, (2) sequence length distribution and range, (3) dataset size "
        "(number of samples) and data volume, (4) feature dimensions and data sparsity. THEN design a "
        "comprehensive data usage plan that includes: (1) detailed data source selection and justification, "
        "(2) data preprocessing pipeline tailored to dataset characteristics (e.g., aggressive quality "
        "control and cleaning for large datasets, data augmentation strategies for small datasets), "
        "(3) data split strategy considering dataset size, (4) data augmentation methods appropriate "
        "for dataset type and size, (5) quality control procedures, and (6) bias mitigation strategies. "
        "CRITICALLY: Choose preprocessing strategies based on dataset characteristics - larger datasets "
        "may require aggressive cleaning and filtering, while smaller datasets may benefit from "
        "augmentation techniques. Provide specific, actionable recommendations with clear rationale "
        "linking dataset characteristics to chosen strategies."
    ),
    role=(
        "Act as the data usage design expert: FIRST conduct comprehensive dataset characteristic analysis "
        "(dataset type, sequence length, size, volume), THEN design a complete data usage plan from data "
        "acquisition to preprocessing to training data preparation. CRITICALLY evaluate dataset "
        "characteristics and select appropriate preprocessing strategies: for large datasets (>10K samples), "
        "focus on quality control, cleaning, and filtering; for small datasets (<1K samples), prioritize "
        "data augmentation techniques. Identify potential data quality issues, and provide detailed, "
        "implementable data management strategies with clear justification linking dataset characteristics "
        "to chosen preprocessing approaches."
        "\n\n"
        "=== Data EDA Tools (CONDITIONAL USE) ===\n"
        "You have two specialized tabular EDA tools:\n"
        "1) describe_table(file_path, max_rows=200000, sequence_columns?, activity_columns?, output_dir='outputs/eda', save_json=True)\n"
        "   - Returns JSON summary and also saves a full JSON report to the project outputs directory.\n"
        "   - For MPRA-like tables, explicitly set sequence_columns=['sequence'/'seq'] and activity_columns=['expr'/'*_log2FC'] when known.\n"
        "2) plot_table_distributions(file_path, output_dir='outputs/eda', max_rows=200000, sequence_columns?, activity_column?)\n"
        "   - Saves PNG artifacts (histograms, scatter plots) under the project outputs directory and returns artifact paths.\n"
        "\n"
       ),
    model=DEFAULT_MODEL,
)

METHODOLOGY_AGENT = Agent(
    title="Methodology Expert",
    expertise=(
        "Design training methodologies for gene regulatory element prediction models, including: "
        "loss function design, optimization strategies, regularization techniques, integration "
        "of biological prior knowledge (motifs, PWMs, sequence constraints), data augmentation "
        "schemes, and training pipeline design."
    ),
    goal=(
        "Based on the task requirements, design a comprehensive training methodology that includes: "
        "(1) loss function selection and design rationale, (2) optimization algorithm and hyperparameter "
        "settings, (3) regularization strategies, (4) prior knowledge integration methods (motifs, "
        "PWMs, etc.), (5) data augmentation approaches, and (6) training pipeline workflow. "
        "Ensure the methodology is biologically sound and statistically robust."
    ),
    role=(
        "Act as the methodology design expert: design a complete training methodology from loss "
        "function to optimization to prior knowledge integration, ensure biological validity and "
        "statistical robustness, and provide detailed, implementable method designs."
    ),
    model=DEFAULT_MODEL,
)

MODEL_ARCHITECT_AGENT = Agent(
    title="Model Architect",
    expertise=(
        "Design neural network architectures for gene regulatory element prediction, including: "
        "flexible architecture selection based on dataset size and complexity, adaptive parameter "
        "count control (smaller models for limited data, larger models for abundant data), "
        "innovative and effective architecture designs, robust model structures, layer design, "
        "long-range dependency modeling, multi-scale feature extraction, interpretability mechanisms, "
        "and computational efficiency optimization. Expertise in balancing model capacity with "
        "data availability to prevent overfitting while maintaining model expressiveness."
    ),
    goal=(
        "Based on the task requirements and dataset characteristics (especially dataset size and "
        "data volume), FIRST evaluate the appropriate model complexity and parameter count: "
        "(1) For small datasets (<1K samples): Design compact, efficient architectures with fewer "
        "parameters, strong regularization, and innovative architectural choices to maximize "
        "performance with limited data. (2) For medium datasets (1K-10K samples): Design moderate "
        "complexity architectures with balanced parameter counts. (3) For large datasets (>10K samples): "
        "Design more expressive architectures with higher capacity while maintaining efficiency. "
        "THEN design a detailed model architecture that includes: (1) architecture type selection "
        "and rationale (encouraging innovative and effective designs), (2) detailed layer-by-layer "
        "design with parameter count justification based on data size, (3) parameter count estimation "
        "and complexity control strategy, (4) mechanisms for modeling long-range dependencies and "
        "multi-scale information, (5) interpretability features, (6) robustness considerations "
        "(generalization, regularization integration), and (7) computational efficiency considerations. "
        "CRITICALLY: Flexibly control model parameters and complexity based on data volume, and "
        "encourage innovative, effective, and robust architecture designs that balance expressiveness "
        "with generalization."
    ),
    role=(
        "Act as the model architecture design expert: FIRST evaluate dataset size and data volume "
        "to determine appropriate model complexity and parameter count. Design a complete neural "
        "network architecture from input layer to output layer, flexibly controlling model parameters "
        "and complexity based on data availability: smaller, more regularized models for limited data, "
        "more expressive models for abundant data. Encourage innovative and effective architectural "
        "choices (e.g., attention mechanisms, residual connections, multi-scale convolutions) while "
        "ensuring robustness and generalization. Consider task-specific requirements (sequence modeling, "
        "regulatory element prediction), ensure appropriate complexity matching data volume, maintain "
        "interpretability, and provide detailed, implementable architecture designs with clear "
        "rationale linking data characteristics to architectural choices."
    ),
    model=DEFAULT_MODEL,
)

RESULT_ANALYST_AGENT = Agent(
    title="Result Analyst",
    expertise=(
        "Design evaluation and result analysis strategies for gene regulatory element prediction "
        "experiments, including: evaluation metric selection, statistical testing design, "
        "validation strategy (cross-validation, held-out test, external validation), biological "
        "validation methods, and result interpretation frameworks."
    ),
    goal=(
        "Based on the task requirements, design a comprehensive result analysis plan that includes: "
        "(1) evaluation metric suite selection and rationale, (2) statistical testing design, "
        "(3) validation strategy (train/validation/test split, cross-validation, external validation), "
        "(4) biological validation methods, (5) result interpretation framework, and (6) summary "
        "and reporting format. Ensure the analysis plan is comprehensive and biologically meaningful."
    ),
    role=(
        "Act as the result analysis design expert: design a complete evaluation and analysis "
        "framework from metric selection to statistical testing to biological validation, ensure "
        "comprehensive and rigorous evaluation, and provide detailed, implementable analysis plans."
    ),
    model=DEFAULT_MODEL,
)


OPTIMIZER_AGENT = Agent(
    title="Training Optimizer",
    expertise=(
        "Analyze training logs, metric summaries, and code implementation for deep learning experiments. "
        "Identify issues in data pipeline, model architecture, optimization, regularization, early stopping, "
        "and evaluation. Propose concrete, implementable optimization strategies that balance performance, "
        "stability, and generalization for promoter activity prediction on short DNA sequences."
    ),
    goal=(
        "Given: (1) training logs with per-epoch losses and validation PCC, (2) final training_summary.json "
        "containing best epochs and test metrics for each ensemble member, (3) current code implementation "
        "for config, dataset, model, training loop, and utilities, and (4) the original experimental design "
        "task description section used for code generation, carefully analyze the current setup. "
        "Identify failure modes, bottlenecks, over/underfitting signals, and mismatches between design "
        "intent and implementation. Then produce a structured optimization plan that includes prioritized "
        "issues, root-cause hypotheses, and concrete code-level and configuration-level changes."
    ),
    role=(
        "Act as an experiment optimization expert. Read the provided logs, metric summaries, and code snippets "
        "to understand how the current E. coli 50bp promoter model is trained and evaluated. "
        "Diagnose where performance is limited (e.g., low PCC/R2, unstable validation curves, weak generalization) "
        "and how training dynamics behave across stages and ensemble members. "
        "Synthesize a detailed optimization plan with: (1) a concise high-level assessment, "
        "(2) a list of concrete problems grouped by component (data, model, optimization, regularization, "
        "evaluation, implementation robustness), (3) an ordered set of recommended changes with estimated impact "
        "and risk, and (4) optional ablation or follow-up experiments. "
        "The output must be directly usable by engineers as a roadmap for the next iteration."
    ),
    model=DEFAULT_MODEL,
)

CODE_GENERATOR_AGENT = Agent(
    title="Code Generator",
    expertise=(
        "Generate production-ready, accurate, and complete Python code for gene regulatory element "
        "design experiments, including: data loading and preprocessing pipelines, model architecture "
        "implementation, training scripts with proper hyperparameter configurations, evaluation and "
        "analysis modules, result visualization, and comprehensive documentation. Expertise in PyTorch, "
        "TensorFlow, NumPy, Pandas, scikit-learn, and bioinformatics libraries (BioPython, etc.). "
        "Ability to translate experimental design specifications into executable, well-structured code "
        "with proper error handling, logging, and modular design."
    ),
    goal=(
        "Based on the provided task description and experimental design report, generate accurate, "
        "complete, and production-ready Python code that implements the entire experimental pipeline. "
        "The code should include: (1) data loading and preprocessing modules with all specified "
        "transformations and augmentation strategies, (2) model architecture implementation matching "
        "the exact specifications (layer dimensions, hyperparameters, activation functions, etc.), "
        "(3) training script with proper loss functions, optimizers, learning rate schedules, and "
        "regularization as specified, (4) evaluation and analysis modules with all required metrics "
        "and statistical tests, (5) result visualization and reporting utilities, and (6) "
        "comprehensive documentation including code comments, docstrings, and usage instructions. "
        "All code must be accurate, executable, follow best practices, and include detailed explanations."
    ),
    role=(
        "Act as an expert code generator for bioinformatics and machine learning experiments: carefully "
        "analyze the task description and experimental design specifications, translate all design "
        "details (data preprocessing steps, model architecture parameters, training hyperparameters, "
        "evaluation metrics, etc.) into accurate and complete Python code. Generate modular, "
        "well-documented code with proper error handling, type hints where appropriate, and clear "
        "explanations. Ensure code accuracy by matching every parameter, hyperparameter, and design "
        "specification from the experimental design. Provide detailed comments explaining the "
        "biological and computational rationale behind code decisions. Generate code that is "
        "production-ready, maintainable, and follows Python best practices."
    ),
    model=CODE_MODEL,  # 代码生成使用专用模型
)

# ==================== Agent初始化函数 ====================

# Agent角色到配置的映射（包含所有Agent）
AGENT_ROLE_MAP = {
    "supervisor": SUPERVISOR_AGENT,
    "data_management": DATA_MANAGEMENT_AGENT,
    "methodology": METHODOLOGY_AGENT,
    "model_architect": MODEL_ARCHITECT_AGENT,
    "result_analyst": RESULT_ANALYST_AGENT,
    "code_generator": CODE_GENERATOR_AGENT,
    "optimizer": OPTIMIZER_AGENT,
}

# Agent角色到RAG角色的映射（现在所有Agent都使用共享知识库）
AGENT_TO_RAG_ROLE = {
    "supervisor": "shared_knowledge_base",
    "data_management": "shared_knowledge_base",
    "methodology": "shared_knowledge_base",
    "model_architect": "shared_knowledge_base",
    "result_analyst": "shared_knowledge_base",
}

# Agent角色到工具分配的映射
AGENT_TO_TOOLS = {
    "supervisor": ["read_file", "write_file"],
    "data_management": ["rag_search", "read_file", "write_file", "describe_table", "plot_table_distributions"],
    "methodology": ["rag_search", "read_file", "write_file"],
    "model_architect": ["rag_search", "read_file", "write_file"],
    "result_analyst": ["rag_search", "read_file", "write_file"],
    "code_generator": ["read_file", "write_file"],
    # optimizer 主要分析当前项目产出的日志与代码，不强制依赖 RAG
    "optimizer": ["read_file", "write_file"],
}


def _create_agent_instance(
    role: str,
    agent_config: Agent,
    rag_system: Optional[HybridRAGSystem],
    tool_registry
) -> Agent:
    """
    创建单个Agent实例的辅助函数（统一初始化逻辑）
    
    Args:
        role: Agent角色
        agent_config: Agent配置
        rag_system: RAG系统实例（如果为None则表示该Agent不需要RAG）
        tool_registry: 工具注册表
    
    Returns:
        Agent实例
    """
    # 创建Agent实例
    agent = Agent(
        title=agent_config.title,
        expertise=agent_config.expertise,
        goal=agent_config.goal,
        role=agent_config.role,
        model=agent_config.model
    )
    
    # 只有在需要RAG且提供了rag_system时才设置RAG接口
    if rag_system is not None and role in AGENT_TO_RAG_ROLE:
        rag_role = AGENT_TO_RAG_ROLE.get(role, "shared_knowledge_base")
        rag_interface = AgentRAGInterface(rag_system, rag_role)
        agent.set_rag_interface(rag_interface)
    
    # 分配工具（消融实验：rag_system=None 时不分配 rag_search）
    tool_names = AGENT_TO_TOOLS.get(role, [])
    if rag_system is None and "rag_search" in tool_names:
        tool_names = [t for t in tool_names if t != "rag_search"]
    tools = [tool_registry.get_tool(name) for name in tool_names if tool_registry.get_tool(name)]
    agent.set_tools(tools)
    
    return agent, tool_names


def initialize_agents_with_rag(
    rag_system: Optional[HybridRAGSystem] = None,
    include_supervisor: bool = False
) -> Dict[str, Agent]:
    """
    初始化所有Agent并设置RAG接口和工具
    
    Args:
        rag_system: RAG系统实例，如果为None则创建新实例
        include_supervisor: 是否包含supervisor（默认False，只初始化专家Agent）
    
    Returns:
        角色名到Agent实例的字典
    """
    # 消融实验：rag_system=None 表示不启用RAG，不在此处创建
    if rag_system is None and get_settings().use_rag:
        rag_system = HybridRAGSystem()
    
    # 获取工具注册表
    tool_registry = get_tool_registry()
    
    agents = {}
    
    # 确定要初始化的角色
    roles_to_init = list(AGENT_ROLE_MAP.keys())
    if not include_supervisor:
        roles_to_init = [r for r in roles_to_init if r != "supervisor"]
    
    for role in roles_to_init:
        agent_config = AGENT_ROLE_MAP[role]
        agent, tool_names = _create_agent_instance(role, agent_config, rag_system, tool_registry)
        agents[role] = agent
        rag_status = "RAG: " + AGENT_TO_RAG_ROLE.get(role, "shared_knowledge_base") if rag_system else "RAG: 禁用"
        print(f"✓ 初始化Agent: {agent.title} ({rag_status}, 工具: {', '.join(tool_names)})")
    
    return agents


def initialize_supervisor(
    rag_system: Optional[HybridRAGSystem] = None
) -> Agent:
    """
    初始化Supervisor Agent并设置RAG接口和工具
    
    Args:
        rag_system: RAG系统实例，如果为None则创建新实例
    
    Returns:
        Supervisor Agent实例
    """
    if rag_system is None and get_settings().use_rag:
        rag_system = HybridRAGSystem()
    
    tool_registry = get_tool_registry()
    agent_config = AGENT_ROLE_MAP["supervisor"]
    agent, tool_names = _create_agent_instance("supervisor", agent_config, rag_system, tool_registry)
    
    rag_status = "RAG: " + AGENT_TO_RAG_ROLE.get("supervisor", "shared_knowledge_base") if rag_system else "RAG: 禁用"
    print(f"✓ 初始化Supervisor: {agent.title} ({rag_status}, 工具: {', '.join(tool_names)})")
    
    return agent


def get_agent_by_role(role: str, rag_system: Optional[HybridRAGSystem] = None) -> Agent:
    """
    根据角色获取Agent实例（带RAG接口和工具）
    
    Args:
        role: Agent角色（supervisor, data_management, methodology, model_architect, result_analyst, code_generator）
        rag_system: RAG系统实例，如果为None且角色需要RAG则创建新实例
    
    Returns:
        Agent实例
    """
    if role not in AGENT_ROLE_MAP:
        raise ValueError(f"未知的Agent角色: {role}。可用角色: {list(AGENT_ROLE_MAP.keys())}")
    
    # 检查该角色是否需要RAG
    needs_rag = role in AGENT_TO_RAG_ROLE
    
    # 只有在需要RAG且未提供rag_system时才创建
    if needs_rag and rag_system is None:
        rag_system = HybridRAGSystem()
    elif not needs_rag:
        # 如果不需要RAG，设置为None
        rag_system = None
    
    tool_registry = get_tool_registry()
    agent_config = AGENT_ROLE_MAP[role]
    agent, _ = _create_agent_instance(role, agent_config, rag_system, tool_registry)
    
    return agent

