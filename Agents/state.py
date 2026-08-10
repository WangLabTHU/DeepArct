"""
Agents/state.py
定义Agent系统的状态和数据结构
"""

from typing import Dict, Any, List, Optional, TypedDict, Literal
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage


# ==================== Agent角色枚举 ====================

class AgentRole:
    """Agent角色定义"""
    DATA_MANAGEMENT = "data_management"
    METHODOLOGY = "methodology"
    MODEL_ARCHITECT = "model_architect"
    RESULT_ANALYST = "result_analyst"
    
    ALL_ROLES = [DATA_MANAGEMENT, METHODOLOGY, MODEL_ARCHITECT, RESULT_ANALYST]


# ==================== 状态定义 ====================

class REAgentState(TypedDict, total=False):
    """RE-Agent系统的状态"""
    messages: List[BaseMessage]
    
    # 任务信息（由用户提供）
    task_description: Optional[str]  # 任务描述
    background: Optional[str]  # 背景要求
    dataset_info: Optional[str]  # 数据集信息
    methodology: Optional[str]  # 方法描述（可选）
    model_architecture: Optional[str]  # 模型架构（可选）
    evaluation_metrics: Optional[str]  # 评估指标（可选）
    additional_info: Optional[Dict[str, Any]]  # 其他附加信息
    agent_task_plans: Optional[Dict[str, Dict[str, Any]]]  # 由Supervisor按角色分配的任务计划
    dataset_statistics: Optional[Dict[str, Dict[str, Any]]]  # 数据集统计信息（文件路径 -> {行数, 列数, 列名}）
    
    # 工作流控制
    next_action: Literal["request_info", "analyze", "discuss", "iterate", "report", "end"]
    
    # 专家分析结果
    data_critique: Optional[Dict[str, Any]]
    methodology_critique: Optional[Dict[str, Any]]
    model_critique: Optional[Dict[str, Any]]
    results_critique: Optional[Dict[str, Any]]
    
    # 迭代控制
    iteration_count: int
    max_iterations: int
    
    # 最终报告
    final_report: Optional[Dict[str, Any]]


# ==================== 分析结果模型 ====================

@dataclass
class CritiqueResult:
    """专家分析结果"""
    agent_role: str
    score: float  # 0-10分
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationReport:
    """优化报告"""
    title: str
    summary: str
    critiques: Dict[str, CritiqueResult]
    overall_score: float
    priority_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 工具注册表 ====================
# ==================== 工具基类 ====================

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


ToolResult = Dict[str, Any]


class BioAgentToolInput(BaseModel):
    """Bio-Agent工具的通用输入模型"""
    pass


class BioAgentTool(BaseTool):
    """
    Bio-Agent工具基类
    
    所有工具都应继承此类
    """
    
    def _run(self, *args, **kwargs) -> ToolResult:
        """同步执行（必须实现）"""
        raise NotImplementedError("工具必须实现_run方法")
    
    async def _arun(self, *args, **kwargs) -> ToolResult:
        """异步执行（可选）"""
        return self._run(*args, **kwargs)
    
    def format_success(self, data: Any, metadata: Dict[str, Any] = None) -> ToolResult:
        """格式化成功结果"""
        # 打印工具调用状态，便于在终端观察工具使用情况
        try:
            tool_name = getattr(self, "name", self.__class__.__name__)
            print(f"🛠️ Tool '{tool_name}' SUCCESS", flush=True)
        except Exception:
            # 日志失败不影响正常返回
            pass

        return {
            "status": "success",
            "success": True,
            "data": data,
            "error": None,
            "metadata": metadata or {}
        }
    
    def format_error(self, error: str, metadata: Dict[str, Any] = None) -> ToolResult:
        """格式化错误结果"""
        # 打印工具调用失败状态
        try:
            tool_name = getattr(self, "name", self.__class__.__name__)
            print(f"❌ Tool '{tool_name}' ERROR: {error}", flush=True)
        except Exception:
            pass

        return {
            "status": "error",
            "success": False,
            "data": None,
            "error": error,
            "metadata": metadata or {}
        }


# ==================== 工具1：RAG检索工具 ====================

from RAG.rag import HybridRAGSystem, AgentRAGInterface


class RAGSearchInput(BioAgentToolInput):
    """RAG检索工具输入"""
    query: str = Field(description="检索查询文本")
    agent_role: str = Field(
        default="data_management",
        description="智能体角色（data_management/methodology/model_architect/result_analyst），用于上下文理解"
    )
    top_k: int = Field(default=5, description="返回结果数量")


class RAGSearchTool(BioAgentTool):
    """RAG知识检索工具 - 从共享知识库检索相关文献和专业知识"""
    
    name: str = "rag_search"
    description: str = """
    从本地知识库检索基因调控元件设计相关的文献和专业知识。
    知识库包含：PubMed文献、arXiv预印本、bioRxiv预印本、PMC开放获取文章、GitHub代码库等。
    可以搜索：数据管理、训练方法、模型架构、评估指标等专业知识。
    """
    args_schema: type[BaseModel] = RAGSearchInput

    # 使用PrivateAttr存储底层RAG系统，避免pydantic字段校验错误
    _rag_system: HybridRAGSystem = PrivateAttr()

    def __init__(self, rag_system: Optional[HybridRAGSystem] = None, **data: Any):
        super().__init__(**data)
        self._rag_system = rag_system or HybridRAGSystem()
    
    async def _arun(
        self,
        query: str,
        agent_role: str = "data_management",
        top_k: int = 5
    ) -> ToolResult:
        """执行RAG检索"""
        try:
            # 创建Agent接口
            rag_interface = AgentRAGInterface(self._rag_system, agent_role)
            
            # 执行检索（现在只有共享库）
            results = await rag_interface.query(
                query=query,
                strategy="hybrid",  # 保留参数以兼容，但实际只使用共享库
                top_k=top_k
            )
            
            # 统计总结果数（现在只有shared_results）
            total_results = len(results.shared_results)
            
            # 格式化返回
            formatted_results = {
                "query": query,
                "results": [
                    {
                        "content": r.content,
                        "score": round(r.score, 4),
                        "metadata": {
                            "doc_id": r.metadata.get("doc_id", ""),
                            "title": r.metadata.get("title", ""),
                            "source": r.metadata.get("source", ""),
                            "authors": r.metadata.get("authors", ""),
                            "journal": r.metadata.get("journal", ""),
                            "date": r.metadata.get("date", ""),
                            "doi": r.metadata.get("doi", "")
                        }
                    }
                    for r in results.shared_results
                ],
                "total_results": total_results,
                "retrieval_time_ms": round(results.retrieval_time_ms, 2)
            }
            
            return self.format_success(
                data=formatted_results,
                metadata={
                    "agent_role": agent_role,
                    "top_k": top_k
                }
            )
        
        except Exception as e:
            return self.format_error(
                error=f"RAG检索失败: {str(e)}",
                metadata={"query": query, "agent_role": agent_role}
            )


# ==================== 工具2：文件读写工具 ====================

import json
from pathlib import Path


class FileReadInput(BioAgentToolInput):
    """文件读取工具输入"""
    file_path: str = Field(description="要读取的文件路径（相对或绝对路径）")
    encoding: str = Field(default="utf-8", description="文件编码")


class FileReadTool(BioAgentTool):
    """文件读取工具 - 读取文本文件或JSON文件"""
    
    name: str = "read_file"
    description: str = """
    读取文件内容。支持文本文件（.txt, .md等）和JSON文件（.json）。
    返回文件内容，如果是JSON文件则自动解析为字典。
    """
    args_schema: type[BaseModel] = FileReadInput
    
    async def _arun(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> ToolResult:
        """读取文件"""
        try:
            path = Path(file_path)
            project_root = Path(__file__).parent.parent  # Agents/state.py -> RE-Agent/
            
            # 首先尝试从 task_description.json 读取数据集路径
            task_desc_path = project_root / "task" / "task_description.json"
            dataset_path_from_json = None
            if task_desc_path.exists():
                try:
                    with open(task_desc_path, 'r', encoding='utf-8') as f:
                        task_desc = json.load(f)
                    dataset_path_from_json = task_desc.get("task_dataset", {}).get("file_path", "")
                    if dataset_path_from_json:
                        # 处理绝对路径或相对路径
                        if Path(dataset_path_from_json).is_absolute():
                            dataset_full_path = Path(dataset_path_from_json)
                        else:
                            dataset_full_path = project_root / dataset_path_from_json
                        
                        # 如果指定的文件路径不存在，但数据集路径存在，自动使用数据集路径
                        if (not path.exists() or not path.is_file()) and dataset_full_path.exists() and dataset_full_path.is_file():
                            print(f"  ℹ️ 文件 '{file_path}' 不存在，自动使用任务描述文件中的数据集路径: {dataset_path_from_json}", flush=True)
                            path = dataset_full_path
                except Exception as e:
                    # 如果读取 task_description.json 失败，继续原有逻辑
                    pass
            
            # 如果文件仍然不存在，尝试智能查找
            if not path.exists() or not path.is_file():
                # 如果路径是相对路径（不包含路径分隔符），尝试在项目根目录查找
                if not any(sep in file_path for sep in ['/', '\\']):
                    # 尝试直接在项目根目录
                    candidate = project_root / file_path
                    if candidate.exists() and candidate.is_file():
                        path = candidate
                    else:
                        # 尝试在 task/data/ 目录下递归查找
                        task_data_dir = project_root / "task" / "data"
                        if task_data_dir.exists():
                            found_files = list(task_data_dir.rglob(file_path))
                            if found_files:
                                path = found_files[0]  # 使用第一个找到的文件
                            else:
                                error_msg = f"文件不存在: {file_path}。已尝试在项目根目录和 task/data/ 目录下查找，未找到。"
                                if dataset_path_from_json:
                                    error_msg += f" 提示：任务描述文件中指定的数据集路径为: {dataset_path_from_json}"
                                else:
                                    error_msg += " 提示：请先使用 read_file 读取 task/task_description.json 获取正确的数据集文件路径。"
                                
                                return self.format_error(
                                    error=error_msg,
                                    metadata={"file_path": file_path, "searched_locations": [str(project_root), str(task_data_dir)]}
                                )
                else:
                    # 如果是相对路径但包含分隔符，尝试从项目根目录解析
                    if not path.is_absolute():
                        candidate = project_root / file_path
                        if candidate.exists() and candidate.is_file():
                            path = candidate
                        else:
                            return self.format_error(
                                error=f"文件不存在: {file_path}",
                                metadata={"file_path": file_path, "tried_path": str(candidate)}
                            )
                    else:
                        return self.format_error(
                            error=f"文件不存在: {file_path}",
                            metadata={"file_path": file_path}
                        )
            
            # 再次检查是否为文件
            if not path.is_file():
                return self.format_error(
                    error=f"路径不是文件: {file_path}",
                    metadata={"file_path": file_path, "resolved_path": str(path)}
                )
            
            # 读取文件（只读取前几行和统计信息，避免过大）
            MAX_PREVIEW_LINES = 10  # 最多显示前10行
            
            with open(path, 'r', encoding=encoding) as f:
                # 先读取所有行以获取统计信息
                all_lines = f.readlines()
                total_lines = len(all_lines)
                total_size = sum(len(line.encode(encoding)) for line in all_lines)
                
                # 只保留前几行作为预览
                preview_lines = all_lines[:MAX_PREVIEW_LINES]
                preview_content = ''.join(preview_lines)
            
            # 如果是JSON文件，尝试解析
            if path.suffix.lower() == '.json':
                try:
                    # 对于JSON文件，尝试解析完整内容以获取结构信息
                    full_content = ''.join(all_lines)
                    data = json.loads(full_content)
                    
                    # 如果是字典，只返回键和部分值预览
                    if isinstance(data, dict):
                        # 只返回前几个键值对作为预览
                        preview_data = dict(list(data.items())[:5])
                        if len(data) > 5:
                            preview_data["_note"] = f"... (and {len(data) - 5} more keys, total: {len(data)} keys)"
                    elif isinstance(data, list):
                        # 只返回前几个元素作为预览
                        preview_data = data[:5]
                        if len(data) > 5:
                            preview_data.append(f"... (and {len(data) - 5} more items, total: {len(data)} items)")
                    else:
                        preview_data = data
                    
                    return self.format_success(
                        data={
                            "file_path": str(path),
                            "file_type": "json",
                            "content": preview_data,
                            "preview_only": True,
                            "total_keys" if isinstance(data, dict) else "total_items": len(data) if isinstance(data, (dict, list)) else 1,
                            "size_bytes": total_size,
                            "line_count": total_lines
                        },
                        metadata={"encoding": encoding}
                    )
                except json.JSONDecodeError as e:
                    return self.format_error(
                        error=f"JSON解析失败: {str(e)}",
                        metadata={"file_path": file_path}
                    )
            
            # 判断文件类型
            file_type = "csv" if path.suffix.lower() == '.csv' else "text"
            
            # 构建预览内容说明
            preview_note = ""
            if total_lines > MAX_PREVIEW_LINES:
                preview_note = f"\n\n[Note: Showing first {MAX_PREVIEW_LINES} lines only. Total lines: {total_lines}]"
            
            return self.format_success(
                data={
                    "file_path": str(path),
                    "file_type": file_type,
                    "content": preview_content + preview_note,  # 只返回前几行预览
                    "preview_only": True,
                    "total_lines": total_lines,
                    "size_bytes": total_size,
                    "line_count": total_lines
                    },
                    metadata={"encoding": encoding}
                )
        
        except Exception as e:
            return self.format_error(
                error=f"读取文件失败: {str(e)}",
                metadata={"file_path": file_path}
            )


class FileWriteInput(BioAgentToolInput):
    """文件写入工具输入"""
    file_path: str = Field(description="要写入的文件路径（相对或绝对路径）")
    content: str = Field(description="要写入的内容（文本或JSON字符串）")
    encoding: str = Field(default="utf-8", description="文件编码")
    create_dirs: bool = Field(default=True, description="如果目录不存在，是否创建")


class FileWriteTool(BioAgentTool):
    """文件写入工具 - 写入文本文件或JSON文件"""
    
    name: str = "write_file"
    description: str = """
    写入文件内容。支持文本文件和JSON文件。
    如果content是JSON字符串，会自动格式化保存。
    如果目录不存在且create_dirs=True，会自动创建目录。
    """
    args_schema: type[BaseModel] = FileWriteInput
    
    async def _arun(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> ToolResult:
        """写入文件"""
        try:
            path = Path(file_path)
            
            # 创建目录（如果需要）
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果是JSON文件，尝试格式化
            if path.suffix.lower() == '.json':
                try:
                    # 尝试解析JSON以验证格式
                    data = json.loads(content)
                    # 格式化写入
                    with open(path, 'w', encoding=encoding) as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except json.JSONDecodeError as e:
                    return self.format_error(
                        error=f"JSON格式无效: {str(e)}",
                        metadata={"file_path": file_path}
                    )
            else:
                # 普通文本文件
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
            
            # 获取文件信息
            file_size = path.stat().st_size
            
            return self.format_success(
                data={
                    "file_path": str(path),
                    "file_size_bytes": file_size,
                    "message": "文件写入成功"
                },
                metadata={"encoding": encoding, "create_dirs": create_dirs}
            )
        
        except Exception as e:
            return self.format_error(
                error=f"写入文件失败: {str(e)}",
                metadata={"file_path": file_path}
            )


# ==================== 工具3：表格数据分布分析（describe） ====================

import pandas as pd
import numpy as np
import re


def _resolve_project_file(file_path: str) -> Path:
    """
    解析用户提供的文件路径到实际存在的文件。
    尽量复用 FileReadTool 的“智能定位”策略，但保持最小实现。
    """
    path = Path(file_path)
    project_root = Path(__file__).parent.parent

    if path.exists() and path.is_file():
        return path

    # 优先：任务描述里指定的数据集路径
    task_desc_path = project_root / "task" / "task_description.json"
    if task_desc_path.exists():
        try:
            with task_desc_path.open("r", encoding="utf-8") as f:
                task_desc = json.load(f)
            dataset_path_from_json = task_desc.get("task_dataset", {}).get("file_path", "")
            if dataset_path_from_json:
                candidate = Path(dataset_path_from_json)
                if not candidate.is_absolute():
                    candidate = project_root / dataset_path_from_json
                if candidate.exists() and candidate.is_file():
                    return candidate
        except Exception:
            pass

    # 相对路径：从项目根解析
    if not path.is_absolute():
        candidate = project_root / file_path
        if candidate.exists() and candidate.is_file():
            return candidate

    # 仅文件名：在 task/data 下递归找
    if not any(sep in file_path for sep in ["/", "\\"]):
        task_data_dir = project_root / "task" / "data"
        if task_data_dir.exists():
            found = list(task_data_dir.rglob(file_path))
            for p in found:
                if p.exists() and p.is_file():
                    return p

    # 兜底：原样返回，让调用方报错信息更准确
    return path


def _read_table(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=max_rows)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=max_rows)
    raise ValueError(f"不支持的表格格式: {suffix} (仅支持 .csv/.tsv/.xlsx/.xls)")


def _resolve_output_dir(output_dir: str) -> Path:
    """
    将输出目录解析到项目根目录下的 outputs/...
    - 绝对路径：原样使用
    - 相对路径：基于项目根目录拼接
    """
    out = Path(output_dir)
    if out.is_absolute():
        return out
    project_root = Path(__file__).parent.parent
    return project_root / out


def _infer_sequence_columns(df: pd.DataFrame, max_candidates: int = 6) -> List[str]:
    """
    启发式识别“生物序列”列（DNA/RNA/蛋白等，常见于 MPRA 表格）。
    目标是：默认尽量准、但不“误伤”普通类别列。
    """
    candidates: List[tuple[str, float]] = []
    for col in df.columns:
        s = df[col]
        if s.dtype.kind in ("i", "u", "f", "b", "M"):
            continue
        ss = s.dropna()
        if ss.empty:
            continue
        # 统一为字符串，并做轻量过滤
        txt = ss.astype("string")
        # 采样避免过慢
        sample = txt.sample(n=min(2000, len(txt)), random_state=0)
        lens = sample.str.len()
        if lens.isna().all():
            continue
        mean_len = float(lens.dropna().mean())
        # 序列通常不是很短的标签，也不会长到成段文本（MPRA 常见 50~300）
        if mean_len < 15 or mean_len > 20000:
            continue
        joined = sample.fillna("").str.upper()
        # 统计“像序列”的比例：由字母/.- 组成，且非空
        is_seq_like = joined.str.fullmatch(r"[A-Z\.\-\*]+").fillna(False)
        seq_like_rate = float(is_seq_like.mean())
        if seq_like_rate < 0.8:
            continue
        # 对 DNA/RNA 更友好：ACGTN/U 占比越高越可能是序列列
        letters = "".join(joined[is_seq_like].tolist())[:200000]
        if not letters:
            continue
        dna_like = sum(ch in "ACGTNU" for ch in letters) / max(1, len(letters))
        score = 0.6 * seq_like_rate + 0.4 * dna_like
        candidates.append((col, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates[:max_candidates]]


def _infer_activity_columns(df: pd.DataFrame, max_candidates: int = 3) -> List[str]:
    """
    启发式识别“活性/表达/标签”数值列（MPRA activity / expression / target）。
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return []
    scored: List[tuple[str, float]] = []
    for col in num_cols:
        name = str(col).lower()
        score = 0.0
        if any(k in name for k in ["activity", "mpra", "expr", "expression", "target", "label", "y", "signal", "log2"]):
            score += 2.0
        s = df[col].dropna()
        if s.empty:
            continue
        # 变化足够大才像标签；常数列降低评分
        uniq = s.nunique(dropna=True)
        score += 1.0 if uniq >= 20 else (0.4 if uniq >= 5 else 0.0)
        scored.append((col, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_candidates] if _ > 0.0]


def _sequence_stats(series: pd.Series, max_rows: int = 800000) -> Dict[str, Any]:
    s = series.dropna().astype("string")
    if s.empty:
        return {"n": 0}
    if len(s) > max_rows:
        s = s.sample(n=max_rows, random_state=0)
    u = s.str.upper()
    lens = u.str.len().dropna().astype(int)
    if lens.empty:
        return {"n": int(len(s))}

    # 允许字符：DNA/RNA/蛋白都常见的字母 + .-*
    allowed_re = re.compile(r"^[A-Z\.\-\*]+$")
    is_allowed = u.apply(lambda x: bool(allowed_re.match(str(x))) if x is not None else False)
    invalid_rate = float((~is_allowed).mean()) if len(is_allowed) else 0.0

    # DNA/RNA：GC 与字母频率（对蛋白列也能给出频率，GC 可能接近 0）
    def _gc(x: str) -> float:
        x = str(x)
        if not x:
            return float("nan")
        x2 = re.sub(r"[^A-Z]", "", x)
        if not x2:
            return float("nan")
        g = x2.count("G")
        c = x2.count("C")
        return (g + c) / len(x2)

    gc = u.apply(_gc)
    # 频率基于采样字符，避免超大字符串
    letters = "".join(u.head(5000).tolist())[:300000]
    letters = re.sub(r"[^A-Z]", "", letters)
    freq: Dict[str, float] = {}
    if letters:
        total = len(letters)
        for ch in ["A", "C", "G", "T", "U", "N"]:
            freq[ch] = round(letters.count(ch) / total, 6)

    return {
        "n": int(len(s)),
        "length": {
            "mean": float(lens.mean()),
            "std": float(lens.std(ddof=0)) if len(lens) > 1 else 0.0,
            "min": int(lens.min()),
            "max": int(lens.max()),
        },
        "invalid_rate": round(invalid_rate, 6),
        "gc": {
            "mean": (None if gc.dropna().empty else float(gc.dropna().mean())),
            "std": (None if gc.dropna().empty else float(gc.dropna().std(ddof=0))) if len(gc.dropna()) > 1 else (0.0 if not gc.dropna().empty else None),
        },
        "dna_freq": freq,
    }


class DescribeTableInput(BioAgentToolInput):
    file_path: str = Field(description="表格路径（csv/tsv/xlsx/xls，相对或绝对路径）")
    max_rows: int = Field(default=800000, description="最多读取行数（用于大表采样/预览）")
    top_k: int = Field(default=20, description="类别列频数 TopK")
    quantiles: List[float] = Field(default=[0.01, 0.05, 0.5, 0.95, 0.99], description="数值列分位数")
    sequence_columns: Optional[List[str]] = Field(default=None, description="指定序列列名（不填则自动识别）")
    activity_columns: Optional[List[str]] = Field(default=None, description="指定活性/标签列名（不填则自动识别）")
    output_dir: str = Field(default="outputs/eda", description="分析结果输出目录（相对路径会落在项目根目录下）")
    save_json: bool = Field(default=True, description="是否将分析结果保存为JSON文件到 output_dir")


class DescribeTableTool(BioAgentTool):
    name: str = "describe_table"
    description: str = """
    对表格数据做分布与缺失分析，返回结构化摘要（JSON）。
    支持 csv/tsv/xlsx/xls。默认最多读取 max_rows 行用于快速分析。
    """
    args_schema: type[BaseModel] = DescribeTableInput

    async def _arun(
        self,
        file_path: str,
        max_rows: int = 800000,
        top_k: int = 20,
        quantiles: List[float] = None,
        sequence_columns: Optional[List[str]] = None,
        activity_columns: Optional[List[str]] = None,
        output_dir: str = "outputs/eda",
        save_json: bool = True,
    ) -> ToolResult:
        try:
            q = quantiles if quantiles is not None else [0.01, 0.05, 0.5, 0.95, 0.99]
            resolved = _resolve_project_file(file_path)
            if not resolved.exists() or not resolved.is_file():
                return self.format_error(
                    error=f"文件不存在: {file_path}",
                    metadata={"file_path": file_path, "resolved_path": str(resolved)},
                )

            df = _read_table(resolved, max_rows=max_rows)
            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

            dtypes = {c: str(df[c].dtype) for c in df.columns}
            missing = (df.isna().mean() * 100.0).round(4).to_dict()

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.columns if c not in num_cols]

            num_summary: Dict[str, Any] = {}
            if num_cols:
                desc = df[num_cols].describe(percentiles=q).T
                # 转为普通 dict，避免 numpy 类型无法序列化
                num_summary = {
                    col: {k: (None if pd.isna(v) else float(v)) for k, v in desc.loc[col].to_dict().items()}
                    for col in desc.index.tolist()
                }

            cat_summary: Dict[str, Any] = {}
            for col in cat_cols[: min(len(cat_cols), 30)]:
                vc = df[col].astype("string").value_counts(dropna=False).head(top_k)
                cat_summary[col] = {
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "top_values": {str(k): int(v) for k, v in vc.to_dict().items()},
                }

            result = {
                "file_path": str(resolved),
                "preview_rows": n_rows,
                "columns": list(df.columns),
                "dtypes": dtypes,
                "missing_percent": missing,
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
                "numeric_summary": num_summary,
                "categorical_summary": cat_summary,
            }

            # ===== 生物序列 + MPRA 活性增强 =====
            seq_cols = sequence_columns if sequence_columns is not None else _infer_sequence_columns(df)
            act_cols = activity_columns if activity_columns is not None else _infer_activity_columns(df)

            seq_details: Dict[str, Any] = {}
            for col in seq_cols[:6]:
                if col in df.columns:
                    seq_details[col] = _sequence_stats(df[col], max_rows=max_rows)

            act_details: Dict[str, Any] = {}
            for col in act_cols[:3]:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    s = df[col].dropna()
                    if not s.empty:
                        act_details[col] = {
                            "n": int(len(s)),
                            "mean": float(s.mean()),
                            "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
                            "min": float(s.min()),
                            "max": float(s.max()),
                        }

            # 简单相关：长度/GC vs 活性（只针对第一个序列列与第一个活性列）
            corr: Dict[str, Any] = {}
            if seq_cols and act_cols and (seq_cols[0] in df.columns) and (act_cols[0] in df.columns):
                seq0 = df[seq_cols[0]].astype("string")
                y0 = df[act_cols[0]]
                lens = seq0.str.len()
                # GC 计算用轻量函数
                gc = seq0.str.upper().apply(lambda x: np.nan if x is None else _sequence_stats(pd.Series([x])).get("gc", {}).get("mean"))
                tmp = pd.DataFrame({"len": lens, "gc": gc, "y": y0})
                tmp = tmp.dropna()
                if len(tmp) >= 20:
                    corr = {
                        "sequence_column": seq_cols[0],
                        "activity_column": act_cols[0],
                        "pearson_len_y": float(tmp["len"].corr(tmp["y"])),
                        "pearson_gc_y": float(tmp["gc"].corr(tmp["y"])),
                        "n": int(len(tmp)),
                    }

            result["bio_sequence"] = {
                "sequence_columns": seq_cols,
                "details": seq_details,
            }
            result["mpra_activity"] = {
                "activity_columns": act_cols,
                "details": act_details,
                "correlation_with_seq_features": corr,
            }

            saved_path = None
            if save_json:
                out_dir = _resolve_output_dir(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                saved_path = out_dir / f"{resolved.stem}_describe.json"
                with saved_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                result["saved_json"] = str(saved_path)

            return self.format_success(
                data=result,
                metadata={"max_rows": max_rows, "top_k": top_k, "quantiles": q, "output_dir": output_dir, "saved_json": str(saved_path) if saved_path else None},
            )
        except Exception as e:
            return self.format_error(
                error=f"表格分布分析失败: {str(e)}",
                metadata={"file_path": file_path},
            )


# ==================== 工具4：表格分布可视化（plot） ====================

import matplotlib
matplotlib.use("Agg")  # 避免无显示环境报错
import matplotlib.pyplot as plt


class PlotTableDistributionsInput(BioAgentToolInput):
    file_path: str = Field(description="表格路径（csv/tsv/xlsx/xls）")
    output_dir: str = Field(default="outputs/eda", description="图片输出目录")
    max_rows: int = Field(default=800000, description="最多读取行数（用于大表采样/预览）")
    max_numeric_cols: int = Field(default=12, description="最多绘制的数值列数量")
    max_categorical_cols: int = Field(default=6, description="最多绘制的类别列数量")
    top_k: int = Field(default=20, description="类别列TopK展示")
    sequence_columns: Optional[List[str]] = Field(default=None, description="指定序列列名（不填则自动识别）")
    activity_column: Optional[str] = Field(default=None, description="指定活性/标签列名（不填则自动识别第一个）")


class PlotTableDistributionsTool(BioAgentTool):
    name: str = "plot_table_distributions"
    description: str = """
    对表格数据绘制基础分布图（数值列直方图、类别列TopK柱状图），输出PNG文件并返回路径列表。
    """
    args_schema: type[BaseModel] = PlotTableDistributionsInput

    async def _arun(
        self,
        file_path: str,
        output_dir: str = "outputs/eda",
        max_rows: int = 800000,
        max_numeric_cols: int = 12,
        max_categorical_cols: int = 6,
        top_k: int = 20,
        sequence_columns: Optional[List[str]] = None,
        activity_column: Optional[str] = None,
    ) -> ToolResult:
        try:
            resolved = _resolve_project_file(file_path)
            if not resolved.exists() or not resolved.is_file():
                return self.format_error(
                    error=f"文件不存在: {file_path}",
                    metadata={"file_path": file_path, "resolved_path": str(resolved)},
                )

            df = _read_table(resolved, max_rows=max_rows)
            out_dir = _resolve_output_dir(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            artifacts: List[Dict[str, Any]] = []

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_numeric_cols]
            cat_cols = [c for c in df.columns if c not in num_cols][:max_categorical_cols]

            # 数值列直方图（合成一张图）
            if num_cols:
                n = len(num_cols)
                ncols = 3
                nrows = int(np.ceil(n / ncols))
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.2 * nrows))
                axes = np.array(axes).reshape(-1)
                for i, col in enumerate(num_cols):
                    ax = axes[i]
                    series = df[col].dropna()
                    ax.hist(series.values, bins=50, color="#4C78A8", alpha=0.85)
                    ax.set_title(col)
                for j in range(len(num_cols), len(axes)):
                    axes[j].axis("off")
                fig.tight_layout()
                png = out_dir / f"{resolved.stem}_numeric_hist.png"
                fig.savefig(png, dpi=150)
                plt.close(fig)
                artifacts.append({"kind": "numeric_hist", "path": str(png), "columns": num_cols})

            # 类别列TopK柱状图（每列一张，避免太挤）
            for col in cat_cols:
                vc = df[col].astype("string").value_counts(dropna=False).head(top_k)
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                ax.bar([str(x) for x in vc.index.tolist()], vc.values.tolist(), color="#F58518", alpha=0.9)
                ax.set_title(f"{col} (Top {top_k})")
                ax.tick_params(axis="x", rotation=45, labelsize=8)
                fig.tight_layout()
                png = out_dir / f"{resolved.stem}_cat_{col}.png"
                fig.savefig(png, dpi=150)
                plt.close(fig)
                artifacts.append({"kind": "categorical_bar", "path": str(png), "column": col, "top_k": top_k})

            # ===== 生物序列 + MPRA 活性增强图 =====
            seq_cols = sequence_columns if sequence_columns is not None else _infer_sequence_columns(df)
            act_col = activity_column
            if act_col is None:
                acts = _infer_activity_columns(df)
                act_col = acts[0] if acts else None

            if seq_cols:
                seq0 = df[seq_cols[0]].astype("string")
                lens = seq0.str.len()
                # 计算 GC（向量化够用）
                seq_up = seq0.str.upper().fillna("")
                letters = seq_up.str.replace(r"[^A-Z]", "", regex=True)
                denom = letters.str.len().replace(0, np.nan)
                gc = (letters.str.count("G") + letters.str.count("C")) / denom

                fig, ax = plt.subplots(figsize=(7.0, 4.0))
                ax.hist(lens.dropna().values, bins=50, color="#54A24B", alpha=0.85)
                ax.set_title(f"{seq_cols[0]} length distribution")
                fig.tight_layout()
                png = out_dir / f"{resolved.stem}_seq_len_hist.png"
                fig.savefig(png, dpi=150)
                plt.close(fig)
                artifacts.append({"kind": "seq_length_hist", "path": str(png), "sequence_column": seq_cols[0]})

                fig, ax = plt.subplots(figsize=(7.0, 4.0))
                ax.hist(gc.dropna().values, bins=50, color="#E45756", alpha=0.85)
                ax.set_title(f"{seq_cols[0]} GC distribution")
                fig.tight_layout()
                png = out_dir / f"{resolved.stem}_seq_gc_hist.png"
                fig.savefig(png, dpi=150)
                plt.close(fig)
                artifacts.append({"kind": "seq_gc_hist", "path": str(png), "sequence_column": seq_cols[0]})

                if act_col is not None and act_col in df.columns and pd.api.types.is_numeric_dtype(df[act_col]):
                    y = df[act_col]
                    tmp = pd.DataFrame({"len": lens, "gc": gc, "y": y}).dropna()
                    if len(tmp) >= 50:
                        fig, ax = plt.subplots(figsize=(6.5, 4.5))
                        ax.scatter(tmp["len"].values, tmp["y"].values, s=8, alpha=0.35, color="#4C78A8")
                        ax.set_xlabel("sequence length")
                        ax.set_ylabel(act_col)
                        ax.set_title(f"length vs {act_col}")
                        fig.tight_layout()
                        png = out_dir / f"{resolved.stem}_len_vs_{act_col}.png"
                        fig.savefig(png, dpi=150)
                        plt.close(fig)
                        artifacts.append({"kind": "len_vs_activity", "path": str(png), "activity_column": act_col, "sequence_column": seq_cols[0]})

                        fig, ax = plt.subplots(figsize=(6.5, 4.5))
                        ax.scatter(tmp["gc"].values, tmp["y"].values, s=8, alpha=0.35, color="#F58518")
                        ax.set_xlabel("GC fraction")
                        ax.set_ylabel(act_col)
                        ax.set_title(f"GC vs {act_col}")
                        fig.tight_layout()
                        png = out_dir / f"{resolved.stem}_gc_vs_{act_col}.png"
                        fig.savefig(png, dpi=150)
                        plt.close(fig)
                        artifacts.append({"kind": "gc_vs_activity", "path": str(png), "activity_column": act_col, "sequence_column": seq_cols[0]})

            return self.format_success(
                data={
                    "file_path": str(resolved),
                    "artifacts": artifacts,
                    "preview_rows": int(df.shape[0]),
                },
                metadata={
                    "output_dir": str(out_dir),
                    "max_rows": max_rows,
                    "max_numeric_cols": max_numeric_cols,
                    "max_categorical_cols": max_categorical_cols,
                },
            )
        except Exception as e:
            return self.format_error(
                error=f"表格可视化失败: {str(e)}",
                metadata={"file_path": file_path, "output_dir": output_dir},
            )


# ==================== 工具注册表 ====================

class ToolRegistry:
    """工具注册表 - 管理所有可用工具"""
    
    def __init__(self):
        self.tools: Dict[str, BioAgentTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        # RAG检索工具
        try:
            self.register_tool("rag_search", RAGSearchTool())
        except Exception as e:
            print(f"⚠️ 注册RAG工具失败: {e}")
        
        # 文件读取工具
        try:
            self.register_tool("read_file", FileReadTool())
        except Exception as e:
            print(f"⚠️ 注册文件读取工具失败: {e}")
        
        # 文件写入工具
        try:
            self.register_tool("write_file", FileWriteTool())
        except Exception as e:
            print(f"⚠️ 注册文件写入工具失败: {e}")

        # 表格分布分析工具
        try:
            self.register_tool("describe_table", DescribeTableTool())
        except Exception as e:
            print(f"⚠️ 注册表格分布分析工具失败: {e}")

        # 表格分布可视化工具
        try:
            self.register_tool("plot_table_distributions", PlotTableDistributionsTool())
        except Exception as e:
            print(f"⚠️ 注册表格可视化工具失败: {e}")
    
    def register_tool(self, name: str, tool: BioAgentTool):
        """注册工具"""
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> Optional[BioAgentTool]:
        """获取工具"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[BioAgentTool]:
        """获取所有工具列表（用于LangChain）"""
        return list(self.tools.values())


# 全局工具注册表实例
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


# ==================== 导出 ====================

__all__ = [
    "AgentRole",
    "REAgentState",
    "CritiqueResult",
    "OptimizationReport",
    "BioAgentTool",
    "BioAgentToolInput",
    "ToolResult",
    "RAGSearchTool",
    "RAGSearchInput",
    "FileReadTool",
    "FileReadInput",
    "FileWriteTool",
    "FileWriteInput",
    "DescribeTableTool",
    "DescribeTableInput",
    "PlotTableDistributionsTool",
    "PlotTableDistributionsInput",
    "ToolRegistry",
    "get_tool_registry"
]
