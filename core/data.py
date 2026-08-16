"""进度计划 JSON 的加载、校验和数据转换。"""

import json

import pandas as pd


def load_json_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_json_from_upload(file_content):
    return json.loads(file_content.decode("utf-8"))


def validate_data_structure(data):
    """校验完整或简化格式的进度计划 JSON。"""
    if "structured_output" in data and isinstance(data["structured_output"], dict):
        structured = data["structured_output"]
    elif "overview" in data and "all_tasks_schedule" in data:
        structured = data
    else:
        return False, "数据缺少 'overview' 或 'all_tasks_schedule' 字段"

    overview = structured.get("overview")
    if not isinstance(overview, dict):
        return False, "缺少 'overview' 字段"
    for field in ("project_name", "total_duration_days", "planned_start_date", "planned_end_date"):
        if field not in overview:
            return False, f"overview 缺少 '{field}' 字段"

    tasks = structured.get("all_tasks_schedule", [])
    if not isinstance(tasks, list) or not tasks:
        return False, "all_tasks_schedule 为空或不是列表"
    for index, task in enumerate(tasks, 1):
        for field in ("task_id", "task_name", "start_date", "finish_date", "duration_days"):
            if field not in task:
                return False, f"第 {index} 个任务缺少 '{field}' 字段"
    return True, "数据格式验证通过"


def normalize_to_wrapped(data):
    """统一为 ``{structured_output: ...}`` 结构。"""
    return data if isinstance(data.get("structured_output"), dict) else {"structured_output": data}


def _section_sort_key(code):
    """把 section_code 转成可排序的 hashable key（tuple），兼容纯数字（1, 2）和字母+数字（WP1-1）两种格式。"""
    import re
    code_str = str(code).strip()
    parts = []
    for token in re.split(r"(\d+)", code_str):
        if token == "":
            continue
        if token.isdigit():
            parts.append(("", int(token)))
        else:
            for sub in re.split(r"(\D+)", token):
                if sub == "":
                    continue
                if sub.isdigit():
                    parts.append(("", int(sub)))
                else:
                    parts.append((sub, 0))
    return tuple(parts) if parts else (("", 0),)


def extract_section_from_task_id(task_id):
    return str(task_id).split(".")[0]


def get_section_mapping(tasks):
    section_names = {
        "1": "施工准备", "2": "地基与基础", "3": "主体结构", "4": "建筑装饰装修",
        "5": "建筑屋面", "6": "建筑给水排水", "7": "建筑电气", "8": "智能建筑",
        "9": "建筑节能与消防", "10": "室外工程", "11": "竣工验收",
    }
    sections = {}
    for task in tasks:
        code = extract_section_from_task_id(task["task_id"])
        sections.setdefault(code, section_names.get(code, f"分部{code}"))
    return dict(sorted(sections.items(), key=lambda item: _section_sort_key(item[0])))


def get_critical_task_ids(critical_path_tasks):
    return {task["task_id"] for task in critical_path_tasks}


def tasks_to_dataframe(tasks, critical_task_ids):
    dataframe = pd.DataFrame(tasks)
    dataframe["is_critical"] = dataframe["task_id"].isin(critical_task_ids)
    dataframe["section_code"] = dataframe["task_id"].apply(extract_section_from_task_id)
    dataframe["start_date"] = pd.to_datetime(dataframe["start_date"])
    dataframe["finish_date"] = pd.to_datetime(dataframe["finish_date"])
    return dataframe.sort_values(["start_date", "task_id"]).reset_index(drop=True)


def calculate_daily_resources(tasks_df):
    date_range = pd.date_range(tasks_df["start_date"].min(), tasks_df["finish_date"].max(), freq="D")
    daily_manpower = pd.Series(0, index=date_range, dtype=float)
    daily_detail = {date: {} for date in date_range}
    for _, task in tasks_df.iterrows():
        resources = task.get("assigned_resources", {})
        if not isinstance(resources, dict):
            continue
        active_dates = date_range[(date_range >= task["start_date"]) & (date_range <= task["finish_date"])]
        for resource_name, count in resources.items():
            if isinstance(count, (int, float)):
                daily_manpower.loc[active_dates] += int(count)
                for date in active_dates:
                    daily_detail[date][resource_name] = daily_detail[date].get(resource_name, 0) + int(count)
    details = ["<br>".join(f"  {name}: {count}" for name, count in sorted(daily_detail[date].items())) or "无" for date in date_range]
    return date_range, daily_manpower.tolist(), details
