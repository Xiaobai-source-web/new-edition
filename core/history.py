"""本地上传历史的存取逻辑。"""

import os
from datetime import datetime
from pathlib import Path


def get_history_directory():
    """返回首个可写的历史目录；失败时返回 ``None``。"""
    candidates = [Path(__file__).resolve().parent.parent / "uploaded_history", Path.cwd() / "uploaded_history"]
    candidates.extend([Path.home() / "progress_plan_history", Path(os.getenv("TEMP") or os.getenv("TMP") or "/tmp") / "progress_plan_history"])
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_test"
            probe.write_text("test", encoding="utf-8")
            probe.unlink()
            return directory
        except OSError:
            continue
    return None


def _json_filename(file_name):
    return file_name if file_name.endswith(".json") else f"{file_name}.json"


def check_history_file_exists(file_name):
    directory = get_history_directory()
    return bool(directory and (directory / _json_filename(file_name)).exists())


def save_file_unique(file_name, file_content):
    directory = get_history_directory()
    if directory is None:
        return False, "历史文件目录不可用"
    destination = directory / _json_filename(file_name)
    if destination.exists():
        return False, f"历史中已存在同名文件 '{destination.name}'，按规则视为同一文件，不重复保存"
    try:
        destination.write_bytes(file_content)
        return True, str(destination)
    except OSError as error:
        return False, f"保存失败：{error}"


def get_history_json_files():
    directory = get_history_directory()
    if directory is None:
        return []
    try:
        return sorted(({
            "file_name": path.name,
            "file_path": str(path),
            "project_name": path.stem,
            "upload_time": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        } for path in directory.glob("*.json")), key=lambda item: item["upload_time"], reverse=True)
    except OSError:
        return []


def delete_history_file(file_path):
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
        return True
    except OSError:
        return False
