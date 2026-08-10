"""
coding_multi.py
分段多轮代码生成 - 严格按照实验报告生成生产级代码
"""

import asyncio
import json
import re
from pathlib import Path
import ast
import shutil
from langchain_core.messages import SystemMessage, HumanMessage

from Agents.prompt import get_agent_by_role
from config.settings import get_settings


SYSTEM_PROMPT = """
You are a senior ML engineer writing production-grade Python code.

Global rules:
- Follow all specs exactly; never simplify any step
- Be deterministic and decisive (temperature=0 mindset)
- Prefer explicit, robust error handling and logging
- Use clear type hints and readable, maintainable structure
- When choices are under-specified, choose the most standard and robust solution in modern PyTorch practice
- Output ONLY in the requested machine-readable format, no extra prose
"""


async def generate_code_from_report(report_path: str | None = None, output_dir: str = "code_generated_multi"):
    """
    根据实验方案报告分段生成完整代码
    
    Args:
        report_path: 实验方案报告JSON文件路径（默认使用 outputs/final_report.json）
        output_dir: 输出目录
    """
    print("="*80)
    print("🚀 Multi-Stage Code Generator - 分段代码生成")
    print("="*80)
    
    # 1. 读取报告文件（默认使用 outputs/final_report.json）
    if report_path is None:
        report_path = "outputs/final_report.json"
    
    print(f"\n📖 读取实验方案报告: {report_path}")
    report_path_obj = Path(report_path)
    if not report_path_obj.exists():
        raise FileNotFoundError(f"报告文件不存在: {report_path}")
    
    with open(report_path_obj, "r", encoding="utf-8") as f:
        full_report = json.load(f)
    
    # 只提取 expert_analyses 之前的内容
    report = {}
    for key in ["title", "summary", "priority_recommendations", "task_information", "experimental_design"]:
        if key in full_report:
            report[key] = full_report[key]
    
    print("✓ 报告文件读取成功（仅使用 expert_analyses 之前的内容）")
    
    # 2. 提取任务信息和实验方案
    task_info = report.get("task_information", {})
    exp_design = report.get("experimental_design", {})
    priority_recs = report.get("priority_recommendations", [])
    
    task_description = task_info.get("description", "")
    background = task_info.get("background", "")
    dataset_info = task_info.get("dataset_info", "")
    
    # 3. 初始化 Code Generator Agent
    print("\n🔧 初始化 Code Generator Agent...")
    code_agent = get_agent_by_role("code_generator")
    print("✓ Code Generator Agent 初始化完成")
    
    # 4. 准备输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # （保留最少状态即可；当前三阶段架构不再需要分阶段接口摘要缓存）
    architecture_plan: str = ""
    files_to_generate: list[str] = []

    def _require_files(code_data: dict, required: list[str], stage_name: str) -> None:
        """确保 LLM 输出中包含所需文件块，否则立即失败，避免后续阶段接口漂移。"""
        files = code_data.get("files", []) or []
        present = {
            Path(f.get("path", "")).name
            for f in files
            if isinstance(f, dict) and (f.get("path", "") or "").strip()
        }
        missing = [name for name in required if name not in present]
        if missing:
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_missing_files.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write(f"Required: {required}\n")
                f.write(f"Present: {sorted(present)}\n")
                f.write("\nRaw parsed file paths:\n")
                for item in files:
                    if isinstance(item, dict):
                        f.write(f"- {item.get('path')}\n")
            raise ValueError(
                f"{stage_name} missing required files: {missing}. "
                f"See {debug_file} for details."
            )

    def _parse_files_from_architecture_plan(plan_text: str) -> list[str]:
        """
        从 ARCHITECTURE_PLAN.md 中解析需要生成的文件列表。

        约定：Stage 0 必须在文档中包含以下段落（严格大小写）：

        ## FILES_TO_GENERATE
        - config.py
        - dataset.py
        - subdir/other.py
        """
        if not plan_text:
            return []

        lines = plan_text.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "## FILES_TO_GENERATE":
                start_idx = i + 1
                break
        if start_idx is None:
            return []

        out: list[str] = []
        for line in lines[start_idx:]:
            s = line.strip()
            if not s:
                continue
            if s.startswith("## "):
                break
            if not s.startswith("-"):
                continue
            item = s.lstrip("-").strip().strip("`").strip()
            if item:
                out.append(item)

        seen: set[str] = set()
        uniq: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _assert_files_not_placeholder(stage_dir: Path, required: list[str], stage_name: str) -> None:
        """
        防止“伪空文件/占位符文件”落盘（例如 'omitted for brevity' / 'Existing content preserved'）。
        这些会导致后续阶段接口完全断裂。
        """
        forbidden_substrings = [
            "omitted for brevity",
            "Existing content preserved",
            "no interface changes required",
            "omitted for brevity in this snippet",
        ]
        too_short_threshold = 80  # 过短的 .py 视为占位风险（2~10 行）

        problems: list[str] = []
        for name in required:
            path = stage_dir / name
            if not path.exists():
                problems.append(f"{name}: missing on disk")
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                problems.append(f"{name}: read error: {e}")
                continue

            # 只对代码文件做占位符检查（md/txt 不强制长度）
            if path.suffix == ".py":
                if len(text.strip()) < too_short_threshold:
                    problems.append(f"{name}: too short ({len(text.strip())} chars)")
                lower = text.lower()
                if any(s.lower() in lower for s in forbidden_substrings):
                    problems.append(f"{name}: contains placeholder phrase")

        if problems:
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_placeholders.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write(f"Stage dir: {stage_dir}\n")
                f.write(f"Required: {required}\n\n")
                f.write("Problems:\n")
                for p in problems:
                    f.write(f"- {p}\n")
            raise ValueError(
                f"{stage_name} produced placeholder/empty-like files. "
                f"Refuse to proceed. See {debug_file}."
            )

    async def _generate_with_retry(
        stage_name: str,
        prompt: str,
        required_files: list[str],
        stage_num: int,
        post_save_check: bool = True,
        max_attempts: int = 2,
        error_feedback: bool = True,
        require_report_alignment_audit: bool = False,
        post_validate_fn=None,
    ) -> dict:
        """
        生成 + 保存 + 校验封装：
        - 第1轮：全量生成
        - 后续轮次：仅定向修补，不再整包重生
        """
        last_err: Exception | None = None
        last_audit: dict | None = None

        async def _repair_files_from_error(
            stage_dir: Path,
            required: list[str],
            error_text: str,
            stage_name_local: str,
        ) -> bool:
            """基于错误信息做定向修补（不全量重生）。"""
            py_required = [f for f in required if f.endswith(".py")]
            if not py_required:
                return False

            # 读取相关 debug 文件，补充错误上下文与命中文件
            debug_parts: list[str] = []
            for dbg in [
                Path("code_generated") / f"debug_{stage_name_local.lower().replace(' ', '_')}_legacy_fields.txt",
                Path("code_generated") / f"debug_{stage_name_local.lower().replace(' ', '_')}_syntax_gate.txt",
                Path("code_generated") / "debug_stage_2_interface_validation.txt",
            ]:
                if dbg.exists() and dbg.is_file():
                    try:
                        dbg_text = dbg.read_text(encoding="utf-8", errors="replace")
                        debug_parts.append(f"[{dbg.name}]\n{dbg_text[:4000]}")
                    except Exception:
                        pass
            full_error_context = (error_text or "") + ("\n\n" + "\n\n".join(debug_parts) if debug_parts else "")

            # 先从错误上下文中提取文件名；提取不到就修补全部 py
            mentioned = set(re.findall(r"([A-Za-z0-9_]+\.py)", full_error_context))
            target_files = [f for f in py_required if Path(f).name in mentioned] or py_required

            MAX_FILE_CHARS = 7000
            current_files: dict[str, str] = {}
            for fn in target_files:
                p = stage_dir / fn
                if p.exists() and p.is_file():
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    current_files[fn] = txt[:MAX_FILE_CHARS] + ("\n# ... truncated ..." if len(txt) > MAX_FILE_CHARS else "")
            if not current_files:
                return False

            repair_prompt = f"""You are fixing generated project files based on validation errors.

IMPORTANT:
- Do NOT regenerate the whole project.
- ONLY rewrite the listed target files.
- Ensure Python syntax is valid.
- Keep cross-file imports/interfaces compatible.

## error_to_fix
{full_error_context}

## target_files_current_content
{json.dumps(current_files, ensure_ascii=False)}

Output format strictly:
===FILE: <filename>===
<complete file content>
===END===
"""
            response = await code_agent.llm.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT.strip() + "\n\n" + code_agent.prompt().strip()), HumanMessage(content=repair_prompt)]
            )
            text = response.content if hasattr(response, "content") and response.content else ""
            data = _parse_json_response(text, f"{stage_name_local} error repair")
            files = data.get("files", []) or []
            allowed = set(target_files)
            wrote = False
            for item in files:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("path", "")).strip()
                code = str(item.get("code", ""))
                if rel not in allowed or not code:
                    continue
                p = stage_dir / rel
                try:
                    if not p.resolve().is_relative_to(stage_dir.resolve()):
                        continue
                except Exception:
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(code, encoding="utf-8")
                wrote = True
                print(f"   🔧 Error repair updated: {rel}")
            return wrote

        for attempt in range(1, max_attempts + 1):
            stage_dir = output_path / f"stage_{stage_num}"
            if attempt == 1:
                current_prompt = prompt
                code_data = await _generate_stage_code(
                    code_agent,
                    current_prompt,
                    stage_name,
                )
            else:
                # 后续轮次只做定向修补，不做全量重生
                repaired = False
                if last_audit and require_report_alignment_audit:
                    repaired = await _repair_files_by_audit_category(
                        stage_dir=stage_dir,
                        required_files=required_files,
                        priority_recommendations=priority_recs,
                        audit=last_audit.get("audit", {}) or {},
                        stage_name=stage_name,
                    )
                if not repaired and last_err is not None:
                    repaired = await _repair_files_from_error(
                        stage_dir=stage_dir,
                        required=required_files,
                        error_text=str(last_err),
                        stage_name_local=stage_name,
                    )
                if not repaired:
                    raise RuntimeError(f"{stage_name} retry {attempt}: targeted repair produced no changes")
                code_data = {"files": []}
            try:
                if attempt == 1:
                    _require_files(code_data, required_files, stage_name)
                    await _save_stage_files(output_path, code_data, stage_num)
                if post_save_check:
                    _assert_python_syntax_all(output_path / f"stage_{stage_num}", required_files, stage_name)
                    _assert_files_not_placeholder(output_path / f"stage_{stage_num}", required_files, stage_name)
                if post_validate_fn is not None:
                    post_validate_fn(output_path / f"stage_{stage_num}")
                if require_report_alignment_audit:
                    audit_result = await _audit_alignment_with_final_report(
                        stage_dir=output_path / f"stage_{stage_num}",
                        required_py_files=required_files,
                        priority_recommendations=priority_recs,
                        stage_name=stage_name,
                    )
                    last_audit = audit_result
                    # 审计不通过时，优先按 category 做定向修复，而非整包重生
                    if audit_result.get("should_fail", False):
                        raise ValueError(
                            f"{stage_name} missed key points from final_report "
                            f"(missing_key_points={audit_result.get('missing_key_points_count', '?')}, "
                            f"high_violations={audit_result.get('high_violations_count', '?')}). "
                            f"See {audit_result.get('debug_file') or 'debug audit file'}."
                        )
                return code_data
            except Exception as e:
                last_err = e
                print(f"   ⚠️ {stage_name} validation failed on attempt {attempt}: {e}")
        raise last_err if last_err else RuntimeError(f"{stage_name} generation failed")

    def _files_for_categories(categories: set[str], required_files: list[str]) -> list[str]:
        """按审计 category 选择需要修复的文件（仅在 required_files 范围内）。"""
        cat_map: dict[str, list[str]] = {
            "data": ["dataset.py", "config.py", "train.py", "evaluate.py"],
            "train": ["train.py", "config.py", "dataset.py"],
            "model": ["model.py", "config.py", "train.py"],
            "eval": ["evaluate.py", "train.py", "utils.py", "config.py"],
            "config": ["config.py", "train.py", "dataset.py", "model.py", "evaluate.py"],
            "method": ["train.py", "utils.py", "config.py", "model.py", "dataset.py"],
        }
        required_set = set(required_files)
        out: list[str] = []
        for c in categories:
            for fn in cat_map.get(c, []):
                if fn in required_set and fn not in out:
                    out.append(fn)
        return out

    async def _repair_files_by_audit_category(
        stage_dir: Path,
        required_files: list[str],
        priority_recommendations: list,
        audit: dict,
        stage_name: str,
    ) -> bool:
        """
        根据审计 violations.category 仅修复相关文件，避免整包重生。
        """
        violations = audit.get("violations", []) if isinstance(audit, dict) else []
        if not isinstance(violations, list) or not violations:
            return False

        # 先按 category 分组，后续每个 category 单独调用一次 LLM
        grouped: dict[str, list[dict]] = {}
        for v in violations:
            if not isinstance(v, dict):
                continue
            cat = str(v.get("category", "")).strip().lower()
            cat_key = cat or "unknown"
            grouped.setdefault(cat_key, []).append(
                {
                    "category": cat_key,
                    "report_requirement": str(v.get("report_requirement", ""))[:500],
                    "fix_hint": str(v.get("fix_hint", ""))[:500],
                    "severity": str(v.get("severity", "")),
                }
            )

        py_required = [f for f in required_files if f.endswith(".py")]
        if not py_required:
            return False

        MAX_FILE_CHARS = 6000
        wrote_any = False

        # 每个 category 单独修补一轮，减少不同类别约束相互干扰
        for cat, cat_violations in grouped.items():
            target_files = _files_for_categories({cat}, py_required)
            if not target_files:
                continue

            current_files: dict[str, str] = {}
            for fn in target_files:
                p = stage_dir / fn
                if p.exists() and p.is_file():
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    current_files[fn] = txt[:MAX_FILE_CHARS] + ("\n# ... truncated ..." if len(txt) > MAX_FILE_CHARS else "")

            if not current_files:
                continue

            repair_prompt = f"""You are fixing generated project files for ONE audit category.

IMPORTANT:
- Do NOT regenerate the whole project.
- ONLY rewrite the listed target files.
- Keep cross-file interfaces compatible.
- Ensure Python syntax is valid.
- Focus only on this category: {cat}

## priority_recommendations
{json.dumps(priority_recommendations, ensure_ascii=False)}

## current_category_violations
{json.dumps(cat_violations, ensure_ascii=False, indent=2)}

## target_files_current_content
{json.dumps(current_files, ensure_ascii=False)}

Output format strictly:
===FILE: <filename>===
<complete file content>
===END===
"""
            response = await code_agent.llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT.strip() + "\n\n" + code_agent.prompt().strip()),
                    HumanMessage(content=repair_prompt),
                ]
            )
            repair_text = response.content if hasattr(response, "content") and response.content else ""
            repair_data = _parse_json_response(repair_text, f"{stage_name} category repair [{cat}]")
            files = repair_data.get("files", []) or []
            if not files:
                continue

            allowed = set(target_files)
            for item in files:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("path", "")).strip()
                code = str(item.get("code", ""))
                if rel not in allowed or not code:
                    continue
                p = stage_dir / rel
                try:
                    if not p.resolve().is_relative_to(stage_dir.resolve()):
                        continue
                except Exception:
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(code, encoding="utf-8")
                wrote_any = True
                print(f"   🔧 Category repair [{cat}] updated: {rel}")

        return wrote_any

    def _validate_cross_file_imports(stage_dir: Path, files: list[str]) -> None:
        """
        Stage 2 校验：语法 + 跨文件 import/from-import 的符号存在性。
        目标是尽早发现“文件间接口依赖断裂”，而不是等运行时才爆炸。
        """
        module_names = {Path(f).stem for f in files if f.endswith(".py")}
        code_by_module: dict[str, str] = {}
        tree_by_module: dict[str, ast.AST] = {}
        public_defs: dict[str, set[str]] = {}

        # 1) 语法解析 + 收集顶层 def/class/模块级变量
        for f in files:
            p = stage_dir / f
            if not p.exists():
                raise FileNotFoundError(f"[Stage2] missing file on disk: {p}")
            code = p.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                raise SyntaxError(f"[Stage2] SyntaxError in {f}: line {e.lineno} — {e.msg}") from e
            mod = p.stem
            code_by_module[mod] = code
            tree_by_module[mod] = tree
            defs: set[str] = set()
            for node in getattr(tree, "body", []):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defs.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defs.add(target.id)
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        defs.add(node.target.id)
            public_defs[mod] = defs

        # 2) 校验 import / from-import：只检查本项目内部模块（config/dataset/utils/model/train/evaluate）
        problems: list[str] = []
        for mod, tree in tree_by_module.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name0 = (alias.name or "").split(".")[0]
                        if name0 in module_names and name0 not in code_by_module:
                            problems.append(f"{mod}.py imports missing module '{name0}'")
                elif isinstance(node, ast.ImportFrom):
                    module = (node.module or "").split(".")[0]
                    if not module or module not in module_names:
                        continue
                    # from module import *
                    if any(a.name == "*" for a in node.names):
                        continue
                    available = public_defs.get(module, set())
                    for alias in node.names:
                        imported = alias.name
                        if imported not in available:
                            problems.append(
                                f"{mod}.py: from {module} import {imported} (symbol not found in {module}.py top-level defs/classes)"
                            )

        if problems:
            debug_file = Path("code_generated") / "debug_stage_2_interface_validation.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            debug_content = "Stage 2 validation failed.\n\nProblems:\n" + "\n".join(f"- {p}" for p in problems)
            debug_content += "\n\nAvailable top-level symbols by module:\n"
            for m in sorted(public_defs.keys()):
                debug_content += f"- {m}: {sorted(public_defs[m])}\n"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(debug_content)
            raise ValueError(f"Stage 2 cross-file validation failed. See {debug_file}")

    def _assert_python_syntax_all(stage_dir: Path, required: list[str], stage_name: str) -> None:
        """
        对落盘后的所有 .py 文件做硬语法门禁。
        语法不过直接失败重试，避免无效代码进入后续语义审计。
        """
        problems: list[str] = []
        for name in required:
            if not name.endswith(".py"):
                continue
            path = stage_dir / name
            if not path.exists():
                problems.append(f"{name}: missing on disk")
                continue
            try:
                code = path.read_text(encoding="utf-8", errors="replace")
                ast.parse(code)
            except SyntaxError as e:
                problems.append(f"{name}: line {e.lineno} - {e.msg}")
            except Exception as e:
                problems.append(f"{name}: parse error - {e}")

        if problems:
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_syntax_gate.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write(f"Stage dir: {stage_dir}\n\n")
                f.write("Syntax problems:\n")
                for p in problems:
                    f.write(f"- {p}\n")
            raise ValueError(
                f"{stage_name} failed Python syntax gate (files={len(problems)}). "
                f"See {debug_file}."
            )

    async def _audit_alignment_with_final_report(
        stage_dir: Path,
        required_py_files: list[str],
        priority_recommendations: list,
        stage_name: str,
    ) -> dict:
        """
        使用 LLM 审计生成代码与 final_report 建议的一致性。
        失败则抛异常，交给重试机制处理。
        """
        py_files = [f for f in required_py_files if f.endswith(".py")]
        code_bundle: dict[str, str] = {}
        for rel in py_files:
            p = stage_dir / rel
            if p.exists() and p.is_file():
                code_bundle[rel] = p.read_text(encoding="utf-8", errors="replace")

        if not code_bundle:
            raise ValueError(f"{stage_name} audit failed: no python files found")

        # 审计仅需要“关键点覆盖”，无需全量代码，避免 token 过高。
        MAX_CODE_CHARS = 3000
        code_bundle = {
            rel: (
                code[:MAX_CODE_CHARS] + ("\n# ... truncated ..." if len(code) > MAX_CODE_CHARS else "")
            )
            for rel, code in code_bundle.items()
        }

        audit_system = (
            "You are a practical code auditor. "
            "Only check whether KEY POINTS from final_report are missing. "
            "Do NOT fail for minor numeric deviations or implementation style differences. "
            "Return ONLY JSON."
        )
        audit_prompt = f"""Audit generated code against final_report priority recommendations only.

## priority_recommendations (source of truth)
{json.dumps(priority_recommendations, ensure_ascii=False)}

## generated_code
{json.dumps(code_bundle, ensure_ascii=False)}

## rules
1) Focus on key-point coverage only (data QC/split, core model/training setup, evaluation metrics/reporting).
2) Only mark "high" severity when a key point is completely missing or clearly contradictory.
3) Minor numeric differences, naming differences, or equivalent implementations should be "low"/"medium", not fail conditions.

Return JSON exactly:
{{
  "pass": true/false,
  "score": 0-100,
  "key_points_checked": ["..."],
  "missing_key_points": ["..."],
  "violations": [
    {{
      "category": "data|method|model|train|eval|config",
      "report_requirement": "...",
      "code_evidence": "...",
      "severity": "high|medium|low",
      "fix_hint": "..."
    }}
  ]
}}
"""
        resp = await code_agent.llm.ainvoke(
            [SystemMessage(content=audit_system), HumanMessage(content=audit_prompt)]
        )
        text = resp.content if hasattr(resp, "content") and resp.content else ""
        if not text.strip():
            raise ValueError(f"{stage_name} audit failed: empty auditor response")

        try:
            audit = json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise ValueError(f"{stage_name} audit failed: non-JSON auditor response")
            audit = json.loads(m.group(0))

        violations = audit.get("violations", []) or []
        missing_key_points = audit.get("missing_key_points", []) or []
        high_violations = [
            v for v in violations
            if isinstance(v, dict) and str(v.get("severity", "")).lower() == "high"
        ]

        # 宽松策略：仅当“关键点缺失”或“高严重度问题”存在时才失败
        should_fail = bool(missing_key_points) or bool(high_violations)
        debug_path = None
        if should_fail:
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_report_alignment.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(audit, ensure_ascii=False, indent=2))
            debug_path = str(debug_file)
        return {
            "should_fail": should_fail,
            "audit": audit,
            "missing_key_points_count": len(missing_key_points),
            "high_violations_count": len(high_violations),
            "debug_file": debug_path,
        }
    
    # ==================== Stage 0: 精简约束文档（仅 task_information + priority_recommendations） ====================
    print("\n" + "="*80)
    print("🧭 Stage 0: Constraints Document (task_information + priority_recommendations)")
    print("="*80)

    stage0_prompt = f"""You are defining a CONCISE project contract. Output ONLY the constraint document.

## task_information
{json.dumps(task_info, ensure_ascii=False, indent=2)}

## priority_recommendations
{chr(10).join(f"- {rec}" for rec in priority_recs)}

## Stage 0 Task
Produce a COMPACT constraint document. Be CONCISE - this will be reused in every subsequent LLM call.

You MUST output exactly one file:

===FILE: CONSTRAINTS.md===
## FILES_TO_GENERATE
- config.py
- dataset.py
- utils.py
- model.py
- train.py
- evaluate.py

## config.py
- Requirements: (2-3 bullet points)
- Interfaces: class/function names and signatures (one line each)

## dataset.py
- Requirements: (2-3 bullet points)
- Interfaces: class/function names and signatures

## utils.py
- Requirements: (2-3 bullet points)
- Interfaces: class/function names and signatures

## model.py
- Requirements: (2-3 bullet points)
- Interfaces: class/function names and signatures

## train.py
- Requirements: (2-3 bullet points)
- Interfaces: class/function names and signatures

## evaluate.py
- Requirements: (2-3 bullet points)
- Interfaces: class/function names and signatures

Keep each section SHORT. No placeholders.
===END===
"""

    await _generate_with_retry(
        stage_name="Stage 0",
        prompt=stage0_prompt,
        required_files=["CONSTRAINTS.md"],
        stage_num=0,
        post_save_check=False,
        max_attempts=2,
    )
    stage_0_dir = output_path / "stage_0"
    constraints_path = stage_0_dir / "CONSTRAINTS.md"
    constraints_doc = constraints_path.read_text(encoding="utf-8", errors="replace") if constraints_path.exists() else ""
    files_to_generate = _parse_files_from_architecture_plan(constraints_doc)
    if not files_to_generate:
        files_to_generate = ["config.py", "dataset.py", "utils.py", "model.py", "train.py", "evaluate.py"]
    print(f"   ✓ Constraints generated. Files: {files_to_generate}")

    # Stage 1 各文件读取的报告片段（固定映射）
    FILE_REPORT_SECTIONS: dict[str, list[str]] = {
        "config.py": ["task_information", "priority_recommendations"],
        "utils.py": ["task_information", "priority_recommendations"],
        "dataset.py": ["task_information", "1_data_usage_plan", "priority_recommendations"],
        "model.py": ["task_information", "3_model_design"],
        "train.py": ["task_information", "2_method_design"],
        "evaluate.py": ["task_information", "4_result_summary"],
    }

    def _build_report_subset(section_keys: list[str]) -> str:
        parts: list[str] = []
        if "task_information" in section_keys:
            parts.append(f"## task_information\n{json.dumps(task_info, ensure_ascii=False, indent=2)}")
        if "priority_recommendations" in section_keys:
            parts.append(f"## priority_recommendations\n{chr(10).join(f'- {r}' for r in priority_recs)}")
        for k in ("1_data_usage_plan", "2_method_design", "3_model_design", "4_result_summary"):
            if k in section_keys and exp_design.get(k):
                parts.append(f"## {k}\n{json.dumps(exp_design[k], ensure_ascii=False, indent=2)}")
        return "\n\n".join(parts)

    # ==================== Stage 1: 6 个文件依次逐个生成（6 次 LLM 调用） ====================
    stage_1_dir = output_path / "stage_1"
    stage_1_dir.mkdir(parents=True, exist_ok=True)
    generation_order = ["config.py", "dataset.py", "utils.py", "model.py", "train.py", "evaluate.py"]
    ordered_files = [f for f in generation_order if f in files_to_generate]
    ordered_files += [f for f in files_to_generate if f not in ordered_files]

    async def _repair_single_file_from_error(stage_dir: Path, file_rel: str, error_text: str, stage_name: str) -> bool:
        """基于错误信息修补单个文件。"""
        p = stage_dir / file_rel
        if not p.exists() or not p.is_file():
            return False
        txt = p.read_text(encoding="utf-8", errors="replace")
        repair_prompt = f"""Fix the following file based on validation errors. Output ONLY the complete fixed file.

## error_to_fix
{error_text[:3000]}

## current_file_content
{txt[:6000]}{chr(10) + "# ... truncated ..." if len(txt) > 6000 else ""}

Output format:
===FILE: {file_rel}===
<complete file content>
===END===
"""
        response = await code_agent.llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT.strip() + "\n\n" + code_agent.prompt().strip()),
            HumanMessage(content=repair_prompt),
        ])
        text = response.content if hasattr(response, "content") and response.content else ""
        data = _parse_json_response(text, f"{stage_name} repair")
        for item in (data.get("files") or []):
            if isinstance(item, dict) and str(item.get("path", "")).strip() == file_rel and item.get("code"):
                p.write_text(str(item["code"]), encoding="utf-8")
                print(f"   🔧 Repaired: {file_rel}")
                return True
        return False

    for file_rel in ordered_files:
        stage_name = f"Stage 1 ({file_rel})"
        print("\n" + "="*80)
        print(f"📦 {stage_name}: Generate {file_rel}")
        print("="*80)
        section_keys = FILE_REPORT_SECTIONS.get(Path(file_rel).name, ["task_information", "priority_recommendations"])
        report_subset = _build_report_subset(section_keys)
        single_prompt = f"""You are generating PRODUCTION-GRADE code. Generate ONLY this file: **{file_rel}**

## CONSTRAINTS (MUST FOLLOW)
{constraints_doc}

## Report Sections for this file
{report_subset}

Generate complete contents for {file_rel}. No placeholders. Follow interfaces in CONSTRAINTS.

Output format:
===FILE: {file_rel}===
<complete file content>
===END===
"""
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                code_data = await _generate_stage_code(code_agent, single_prompt, stage_name)
                _require_files(code_data, [file_rel], stage_name)
                await _save_stage_files(output_path, code_data, 1)
                _assert_python_syntax_all(stage_1_dir, [file_rel], stage_name)
                _assert_files_not_placeholder(stage_1_dir, [file_rel], stage_name)
                print(f"   ✓ {file_rel} generated")
                break
            except Exception as e:
                last_err = e
                print(f"   ⚠️ Attempt {attempt} failed: {e}")
                if attempt < 3:
                    repaired = await _repair_single_file_from_error(stage_1_dir, file_rel, str(e), stage_name)
                    if repaired:
                        _assert_python_syntax_all(stage_1_dir, [file_rel], stage_name)
                        _assert_files_not_placeholder(stage_1_dir, [file_rel], stage_name)
                        print(f"   ✓ {file_rel} repaired")
                        break
        else:
            raise last_err if last_err else RuntimeError(f"{stage_name} failed")

    required_files = files_to_generate
    core_py = [p for p in required_files if p.endswith(".py")]

    # Stage 1 完成后：跨文件 import 校验
    if core_py:
        _validate_cross_file_imports(stage_1_dir, core_py)
        print("   ✓ Cross-file imports validated")

    # ==================== 合并所有代码到最终目录 ====================
    print("\n" + "="*80)
    print("📦 Merging All Stages")
    print("="*80)
    
    await _merge_all_stages(output_path)

    # ==================== 删除前五轮的分别文件夹 ====================
    print("\n" + "="*80)
    print("🗑️ Cleaning Up Stage Directories")
    print("="*80)
    
    for stage_num in [0, 1]:
        stage_dir = output_path / f"stage_{stage_num}"
        if stage_dir.exists():
            try:
                shutil.rmtree(stage_dir)
                print(f"   ✓ Deleted: stage_{stage_num}/")
            except Exception as e:
                print(f"   ⚠️ Failed to delete stage_{stage_num}/: {e}")
    
    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✅ Multi-Stage Code Generation Complete!")
    print("="*80)
    print(f"📁 Output directory: {output_path.resolve()}")
    
    # 列出最终生成的文件
    final_files = list((output_path / "final").glob("*")) if (output_path / "final").exists() else []
    if final_files:
        print(f"\n📄 Final generated files ({len(final_files)}):")
        for f in sorted(final_files):
            if f.is_file():
                size = f.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                print(f"   ✓ {f.name:<40} ({size_str})")


async def _generate_stage_code(code_agent, prompt: str, stage_name: str) -> dict:
    """生成单个阶段的代码"""
    print(f"\n💻 Generating {stage_name} code...")
    print("   (This may take several minutes, please wait...)")
    
    response = None
    code_content = ""
    try:
        settings = get_settings()
        code_model = settings.code_model
        
        # 使用标准的 chat API。合并系统提示，避免多个 system message 产生歧义。
        combined_system = SYSTEM_PROMPT.strip() + "\n\n" + code_agent.prompt().strip()
        messages = [
            SystemMessage(content=combined_system),
            HumanMessage(content=prompt),
        ]
        
        try:
            # 代码生成直接使用 llm，不使用工具绑定
            # 代码生成任务应该直接返回代码内容，不应该有工具调用
            response = await code_agent.llm.ainvoke(messages)
            code_content = response.content if hasattr(response, 'content') and response.content else ""
            
            # 检查响应元数据，判断是否因为长度限制被截断
            finish_reason = None
            reasoning_tokens = 0
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                finish_reason = metadata.get('finish_reason')
                token_usage = metadata.get('token_usage', {}) or metadata.get('tokeen_usage', {})
                if isinstance(token_usage, dict):
                    completion_details = token_usage.get('completion_tokens_details', {})
                    if isinstance(completion_details, dict):
                        reasoning_tokens = completion_details.get('reasoning_tokens', 0)
            
            # 调试信息：检查响应对象结构
            if not code_content or not code_content.strip():
                print(f"   ⚠️ Empty or whitespace-only response detected. Response type: {type(response)}")
                if hasattr(response, 'content'):
                    print(f"   ⚠️ response.content type: {type(response.content)}, length: {len(str(response.content))}")
                if finish_reason == 'length':
                    print(f"   ⚠️ Response was truncated due to length limit (finish_reason: length)")
                    if reasoning_tokens > 0:
                        print(f"   ⚠️ Model used {reasoning_tokens} reasoning tokens, but content was empty")
                        print(f"   💡 Suggestion: The prompt may be too long or max_tokens too small.")
                        print(f"   💡 Consider: Reducing prompt size or increasing max_tokens limit.")
                if hasattr(response, '__dict__'):
                    print(f"   ⚠️ Response dict keys: {list(response.__dict__.keys())}")
                # 尝试获取更多调试信息
                if response is not None and hasattr(response, 'additional_kwargs'):
                    print(f"   ⚠️ additional_kwargs keys: {list(response.additional_kwargs.keys()) if response.additional_kwargs else 'None'}")
                    # 检查是否有 refusal
                    if response.additional_kwargs and 'refusal' in response.additional_kwargs:
                        refusal_content = response.additional_kwargs.get('refusal', '')
                        print(f"   ⚠️ Model refusal detected: {refusal_content[:500] if refusal_content else 'No refusal message'}")
                        print(f"   💡 Suggestion: The prompt may be too long, contain problematic content, or request something the model refuses to do.")
        except Exception as e:
            if "chat model" in str(e).lower() or "not supported" in str(e).lower():
                print(f"   ⚠️ Model {code_model} doesn't support chat API, falling back...")
                from Agents.prompt import DEFAULT_MODEL
                code_agent.model = DEFAULT_MODEL
                code_agent._llm = None
                code_agent._llm_with_tools = None
                
                response = await code_agent.llm.ainvoke(messages)
                code_content = response.content if response.content else ""
                
                if not code_content:
                    print(f"   ⚠️ Empty response after fallback. Response type: {type(response)}")
                    if hasattr(response, '__dict__'):
                        print(f"   ⚠️ Response dict: {response.__dict__}")
                
                print(f"   ✓ Switched to model: {DEFAULT_MODEL}")
            else:
                raise
        
        # 检查响应是否为空
        if not code_content or not code_content.strip():
            # 获取 finish_reason 和 token 信息用于错误提示
            finish_reason = None
            reasoning_tokens = 0
            total_tokens = 0
            if response is not None and hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                finish_reason = metadata.get('finish_reason')
                token_usage = metadata.get('token_usage', {}) or metadata.get('tokeen_usage', {})
                if isinstance(token_usage, dict):
                    total_tokens = token_usage.get('total_tokens', 0)
                    completion_details = token_usage.get('completion_tokens_details', {})
                    if isinstance(completion_details, dict):
                        reasoning_tokens = completion_details.get('reasoning_tokens', 0)
            
            error_msg = f"   ❌ Empty response from API. Model: {code_model}"
            
            # 检查是否有 refusal
            if response is not None and hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                if 'refusal' in response.additional_kwargs:
                    refusal_content = response.additional_kwargs.get('refusal', '')
                    error_msg += f"\n   ⚠️ Model refusal detected: {str(refusal_content)[:200]}"
                    error_msg += f"\n   💡 Suggestion: The prompt may be too long or contain problematic content. Try reducing the prompt size."
            
            if finish_reason == 'length':
                error_msg += f"\n   ⚠️ Response was truncated due to length limit (finish_reason: length)"
                if reasoning_tokens > 0:
                    error_msg += f"\n   ⚠️ Model used {reasoning_tokens} reasoning tokens but content was empty"
                error_msg += f"\n   💡 Suggestion: Reduce prompt size or increase max_tokens limit"
            
            print(error_msg)
            
            # 保存调试信息
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_empty_response.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write(f"Model: {code_model}\n")
                f.write("="*80 + "\n")
                f.write("Empty response received from API.\n")
                f.write(f"Response length: {len(code_content)}\n")
                if finish_reason:
                    f.write(f"Finish reason: {finish_reason}\n")
                if reasoning_tokens > 0:
                    f.write(f"Reasoning tokens: {reasoning_tokens}\n")
                if total_tokens > 0:
                    f.write(f"Total tokens: {total_tokens}\n")
                f.write("\nResponse metadata:\n")
                if response is not None and hasattr(response, 'response_metadata'):
                    import json
                    f.write(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))
            
            error_detail = f"Empty response from API for {stage_name}"
            if finish_reason == 'length':
                error_detail += " (truncated due to length limit)"
            error_detail += f". Check {debug_file} for details."
            raise ValueError(error_detail)
        
        # 解析模型输出为结构化的文件列表
        print(f"   📝 Response length: {len(code_content)} characters")
        code_data = _parse_json_response(code_content, stage_name)
        
        # 如果解析失败，保存原始响应用于调试
        if not code_data.get("files"):
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_response.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write("="*80 + "\n")
                f.write(code_content)
            print(f"   ⚠️ Parsing failed, saved raw response to: {debug_file}")
            print(f"   📄 First 500 chars of response:\n{code_content[:500]}")
        
        return code_data
        
    except Exception as e:
        print(f"\n❌ {stage_name} code generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def _parse_json_response(content: str, stage_name: str = "") -> dict:
    """解析 LLM 返回的文件输出。
    
    首选新的分隔符协议：
    ===FILE: path===
    <code>
    ===FILE: other.py===
    <code>
    ===END===
    """
    if not content or not content.strip():
        print(f"   ⚠️ Empty response content")
        return {"files": [], "stage": 0, "description": "Empty response"}
    
    # 1) 按分隔符协议解析
    # 兼容模型偶发输出不严格的结束标记：
    # - 期望：===END===
    # - 偶发：===END  或  ===END==（缺少若干 '='）
    file_pattern = re.compile(
        r"===FILE:\s*(.+?)===\s*\n(.*?)(?=\n===FILE:|\n===END|$)",
        re.DOTALL,
    )
    files = []
    for match in file_pattern.finditer(content):
        path = match.group(1).strip()
        code = match.group(2)
        files.append({"path": path, "code": code.rstrip("\n")})
    
    if files:
        print(f"   ✓ Parsed {len(files)} files from delimiter format")
        return {
            "files": files,
            "stage": stage_name,
            "description": f"Parsed from delimiter format for {stage_name}",
        }
    
    # 2) 若未匹配到分隔符，返回空文件列表（保底）
    print("   ⚠️ No delimiter-based file blocks found; returning empty file list")
    return {"files": [], "stage": 0, "description": "No file blocks found"}


async def _save_stage_files(output_path: Path, code_data: dict, stage_num: int):
    """保存单个阶段的代码文件，并进行基本语法校验。"""
    stage_dir = output_path / f"stage_{stage_num}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    files = code_data.get("files", [])
    if not files:
        print(f"   ⚠️ No files found in {code_data.get('stage', stage_num)} response")
        return
    
    print(f"   📁 Saving {len(files)} files to stage_{stage_num}/")
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        
        file_path_str = (file_info.get("path") or 
                        file_info.get("file_path") or 
                        file_info.get("filename") or 
                        file_info.get("name") or "")
        
        if not file_path_str:
            continue
        
        file_path = stage_dir / file_path_str
        # 防止路径穿越（例如 ../../somewhere 或绝对路径）
        try:
            if not file_path.resolve().is_relative_to(stage_dir.resolve()):
                print(f"   ⚠️ Skipping suspicious path: {file_path_str}")
                continue
        except Exception:
            # 任何解析异常都保守跳过
            print(f"   ⚠️ Skipping suspicious path: {file_path_str}")
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        code = (file_info.get("code") or 
               file_info.get("content") or 
               file_info.get("source") or 
               file_info.get("source_code") or "")
        
        if code:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"   ✓ Saved: {file_path.name} ({len(code)} chars)")

            # 对 Python 文件做语法校验
            if file_path.suffix == ".py":
                ok = validate_python_syntax(code, filename=str(file_path))
                if not ok:
                    print(f"   ⚠️ {file_path.name} has syntax errors, please review.")
        else:
            print(f"   ⚠️ Skipped empty file: {file_path_str}")


async def _merge_all_stages(output_path: Path):
    """合并所有阶段的代码到最终目录（以磁盘落盘内容为准）"""
    final_dir = output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 Merging all stages into final directory...")
    
    # 收集 stage_1 的代码文件；可选复制 stage_0 的 CONSTRAINTS.md 供参考
    all_files: dict[str, Path] = {}
    for stage_num in [1]:
        stage_dir = output_path / f"stage_{stage_num}"
        if not stage_dir.exists():
            continue
        for src in sorted(stage_dir.iterdir()):
            if src.is_file():
                all_files[src.name] = src
    constraints_src = output_path / "stage_0" / "CONSTRAINTS.md"
    if constraints_src.exists():
        all_files["CONSTRAINTS.md"] = constraints_src

    for name, src in all_files.items():
        shutil.copy2(src, final_dir / name)
    
    print(f"   ✓ Merged {len(all_files)} files to final/")


def validate_python_syntax(code: str, filename: str = "") -> bool:
    """对 Python 代码做语法校验，返回 True 表示合法。"""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error in {filename}: line {e.lineno} — {e.msg}")
        return False


def main():
    """主函数"""
    import sys
    
    # 默认使用 outputs/final_report.json（在函数内部处理）
    report_path = None
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    
    output_dir = "code_generated_new"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    asyncio.run(generate_code_from_report(report_path, output_dir))


if __name__ == "__main__":
    main()
