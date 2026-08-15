"""调用 Dify Chatflow 工作流 API。

增强点：
1. API 调用重试机制（偶发网络波动自动重试，业务错误不重试）
2. 节点进度回调（实时通知调用方当前执行到哪个节点，用于 UI 进度反馈）
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Tuple

import requests

from config import (
    DIFY_API_KEY,
    DIFY_CHATFLOW_URL,
    DIFY_MAX_RETRIES,
    DIFY_RETRY_INTERVAL,
    DIFY_TIMEOUT,
)

# 节点标题 → 用户可读的进度描述
# 工作流执行到对应节点时，回调会传这段文字给 UI 显示
_NODE_TITLE_MAP: Dict[str, str] = {
    "1.1综合参数提取": "正在提取项目参数...",
    "1.2边界条件补充": "正在补充边界条件...",
    "1.3WBS一二级+粗颗粒逻辑": "正在生成工作分解结构（WBS）...",
    "1.4WBS三级生成": "正在细化 WBS 三级任务...",
    "1.5WBS审查": "正在审查 WBS 结构...",
    "2.1工序依赖关系": "正在计算工序依赖关系...",
    "3.1资源调优": "正在进行资源调优...",
    "3.2进度方案生成": "正在生成进度方案...",
    "4.1方案审核": "正在审核方案逻辑...",
    "4.2方案重生成": "正在重新生成方案...",
    "5.1人员配置": "正在计算人员配置...",
    "5.2资源荷载": "正在计算设备资源荷载...",
    "5.3横道图": "正在整理横道图数据...",
    "6.1扰动事件解析": "正在解析扰动事件...",
    "6.2生成更新方案": "正在生成更新方案...",
    "6.3进度方案生成": "正在生成优化后的进度方案...",
    "可视化": "正在生成方案摘要...",
    "LLM 19": "正在输出最终方案...",
    "输出整理": "正在整理输出...",
    "输出整理2": "正在整理输出...",
    "输出整理4": "正在整理输出...",
    "输出整理5": "正在整理输出...",
}


def call_dify_chatflow(
    query_text: str,
    *,
    files: list | None = None,
    conversation_id: str | None = None,
    current_plan_json: Dict[str, Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Tuple[str, Dict[str, Any], str | None]:
    """调用 Dify Chatflow API（流式），返回 (文字描述, structured_output, conversation_id)。

    - 文字描述：在聊天框显示给用户（可复制）
    - structured_output：如果有，就是 {overview, all_tasks_schedule, ...} 的 JSON 字典，
      直接交给 normalize_to_wrapped + validate_data_structure 后就可以渲染图表；
      若没有可用结构则返回空字典 {}。
    - conversation_id：Dify 会话 ID，续问时要传入。
    - on_progress：可选的进度回调函数，工作流每执行到一个节点就会调用一次，
      传入该节点的用户可读描述（如 "正在生成进度方案..."）。
      UI 层可以用这个来实时显示 AI 的执行进度。
    """
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    final_query = query_text
    if files:
        for file_obj in files:
            try:
                name = getattr(file_obj, "name", None) or "上传文件"
                if name.lower().endswith(".docx"):
                    try:
                        from docx import Document
                        from io import BytesIO

                        doc = Document(BytesIO(file_obj.read()))
                        parts: list[str] = []
                        for p in doc.paragraphs:
                            if p.text.strip():
                                parts.append(p.text)
                        for t in doc.tables:
                            for row in t.rows:
                                cells = [cell.text.strip() for cell in row.cells]
                                parts.append(" | ".join(cells))
                        content = "\n".join(parts)
                    except Exception:
                        content = f"(无法解析 .docx 文件：{name})"
                else:
                    raw = file_obj.read()
                    try:
                        content = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    except Exception:
                        content = f"(无法读取文件：{name})"
                final_query += f"\n\n===== 文件 {name} 内容 =====\n{content}\n"
            except Exception:
                final_query += f"\n\n(文件读取失败)\n"

    if current_plan_json:
        plan_str = json.dumps(current_plan_json, ensure_ascii=False, indent=2)
        final_query = (
            f"【用户要求】\n{final_query}\n\n"
            f"【当前已有进度计划数据】\n{plan_str}\n\n"
            f"请基于当前进度计划数据，结合用户要求进行生成/调整。"
        )

    payload: Dict[str, Any] = {
        "inputs": {},
        "query": final_query,
        "response_mode": "streaming",
        "user": "streamlit_web",
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id

    # ====== 带重试的调用 ======
    last_error: Exception | None = None
    for attempt in range(1, DIFY_MAX_RETRIES + 1):
        try:
            return _do_stream_request(
                headers=headers,
                payload=payload,
                on_progress=on_progress,
            )
        except _BusinessError as exc:
            # 业务错误（AI 返回 error 事件）不重试，直接抛出
            raise RuntimeError(str(exc)) from exc
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc
            if attempt < DIFY_MAX_RETRIES:
                # 通知 UI 正在重试
                if on_progress:
                    on_progress(f"网络异常，{DIFY_RETRY_INTERVAL} 秒后第 {attempt + 1} 次尝试...")
                time.sleep(DIFY_RETRY_INTERVAL)
                continue
            # 重试用完，抛出友好错误
            raise RuntimeError(
                f"AI 服务连接失败（已重试 {DIFY_MAX_RETRIES} 次）：{exc}。请检查网络后重试。"
            ) from exc
        except Exception as exc:
            # 其他未知错误不重试
            raise RuntimeError(f"调用 AI 服务时发生错误：{exc}") from exc

    # 理论上不会走到这里
    raise RuntimeError(f"AI 服务调用失败：{last_error}")


class _BusinessError(Exception):
    """业务错误（AI 返回 error 事件），不重试。"""


def _do_stream_request(
    *,
    headers: dict,
    payload: dict,
    on_progress: Callable[[str], None] | None,
) -> Tuple[str, Dict[str, Any], str | None]:
    """执行一次流式请求，返回 (文字描述, structured_output, conversation_id)。

    遇到网络错误/超时抛出对应异常（由上层重试）；遇到业务 error 事件抛出 _BusinessError。
    """
    answer_parts: list[str] = []
    structured_output: Dict[str, Any] = {}
    new_conversation_id: str | None = None

    with requests.post(
        DIFY_CHATFLOW_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=DIFY_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except Exception:
                continue
            if not line.startswith("data: "):
                continue
            body = line[len("data: "):]
            if body.strip() == "[DONE]":
                break
            try:
                evt = json.loads(body)
            except json.JSONDecodeError:
                continue
            event = evt.get("event")
            if event == "message":
                answer_parts.append(evt.get("answer", ""))
            elif event == "message_end":
                new_conversation_id = evt.get("conversation_id")
                metadata = evt.get("metadata") or {}
                # Dify 工作流 answer 节点中若有结构化输出，优先从 outputs 拿
                outputs = metadata.get("outputs") if isinstance(metadata, dict) else None
                if not outputs:
                    outputs = evt.get("outputs") or {}
                if isinstance(outputs, dict):
                    # 优先找 structured_output 字段；兼容某些节点命名 structured_output / plan
                    for key in ("structured_output", "plan", "output"):
                        if key in outputs and isinstance(outputs[key], dict):
                            structured_output = outputs[key]
                            break
                    if not structured_output and "text" in outputs:
                        candidate = _try_parse_structured_from_text(str(outputs["text"]))
                        if candidate:
                            structured_output = candidate
                if not structured_output and answer_parts:
                    candidate = _try_parse_structured_from_text("".join(answer_parts))
                    if candidate:
                        structured_output = candidate
            elif event == "node_started":
                # 节点开始事件：通知 UI 当前执行到哪个节点
                if on_progress:
                    node_data = evt.get("data") or {}
                    node_title = node_data.get("title") or ""
                    progress_text = _NODE_TITLE_MAP.get(node_title)
                    if progress_text:
                        on_progress(progress_text)
            elif event == "workflow_started":
                if on_progress:
                    on_progress("AI 开始处理您的请求...")
            elif event == "workflow_finished":
                if on_progress:
                    on_progress("处理完成，正在整理结果...")
            elif event == "error":
                raise _BusinessError(f"Dify 返回错误：{evt.get('message', '未知错误')}")

    answer_text = "".join(answer_parts).strip()
    return answer_text, structured_output, new_conversation_id


def _try_parse_structured_from_text(text: str) -> Dict[str, Any]:
    """尝试从 AI 回答文本里抓取符合绘图结构的 JSON。

    搜索顺序：
    1) ```json ... ``` 代码块
    2) 第一个以 { 开头 且 包含 overview 或 all_tasks_schedule 的 JSON
    """
    if not text:
        return {}

    import re

    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            if _looks_like_plan(data):
                return data
        except Exception:
            pass

    # 直接匹配包含关键字段的 JSON 对象
    for m in re.finditer(r"\{[^{}]*\"(overview|all_tasks_schedule)\"[^{}]*\}", text, re.DOTALL):
        pass  # 防止贪婪匹配；改用手动切分
    start = text.find("{")
    while start != -1:
        depth = 0
        end = -1
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                data = json.loads(text[start:end + 1])
                if _looks_like_plan(data):
                    return data
            except Exception:
                pass
            start = text.find("{", end + 1)
        else:
            break
    return {}


def _looks_like_plan(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    # 完整形式 {structured_output: {...}}
    if isinstance(data.get("structured_output"), dict):
        inner = data["structured_output"]
        return "overview" in inner and "all_tasks_schedule" in inner
    # 扁平形式 {...}
    return "overview" in data and "all_tasks_schedule" in data
