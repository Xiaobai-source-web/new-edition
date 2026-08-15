# 施工进度计划智能助手

基于 **Streamlit + Plotly + Pandas** 的施工进度计划可视化应用。右侧 Dify 对话框用于生成计划 JSON；应用负责导入、校验并展示甘特图、资源曲线、里程碑和风险信息。

## 功能

- 导入、校验和展示施工进度 JSON
- 甘特图：分部筛选、里程碑、悬停查看工序与资源
- 人力资源负荷曲线、工序详情、资源计划和风险清单
- 导出交互式 HTML 图表和 CSV 工序表
- 上传文件的本地历史记录与同名去重
- 内置“名创优品”示例项目

## 项目结构

```text
.
├── app.py              # Streamlit 入口与页面编排
├── config.py           # 路径、Dify IFRAME 等应用配置
├── core/
│   ├── data.py          # JSON 加载、校验、任务和资源数据处理
│   └── history.py       # 上传历史文件的本地存取
├── 名创优品.json         # 内置示例进度计划
└── requirements.txt     # Python 依赖
```

页面层与核心逻辑已分离：`app.py` 仅处理 Streamlit 界面、图表和用户交互；`core/data.py` 不依赖 Streamlit，可单独测试；`core/history.py` 负责本地文件持久化；`config.py` 集中管理可变配置。

## 安装与启动

需要 Python 3.9 或更高版本。

```bash
python -m venv venv
```

Windows：

```bash
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux：

```bash
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

默认打开 `http://localhost:8501`。如需局域网访问：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 使用流程

1. 在右侧 AI 对话框描述项目，获取 JSON 文本。
2. 将文本保存为 UTF-8 编码的 `.json` 文件。
3. 在左侧上传 JSON，或从历史文件加载。
4. 在图表页筛选分部、查看资源，并按需导出 HTML 或 CSV。

上传记录会优先保存在项目根目录的 `uploaded_history/`。在 Streamlit Community Cloud 等临时环境中，该目录可能会在服务重启后清空。

## JSON 格式

应用支持下面两种顶层结构：直接包含 `overview` 和 `all_tasks_schedule`，或将它们包在 `structured_output` 内。必填字段如下：

```json
{
  "structured_output": {
    "overview": {
      "project_name": "项目名称",
      "total_duration_days": 210,
      "planned_start_date": "2026-03-01",
      "planned_end_date": "2026-09-27"
    },
    "all_tasks_schedule": [
      {
        "task_id": "1.1.1",
        "task_name": "施工准备",
        "start_date": "2026-03-01",
        "finish_date": "2026-03-10",
        "duration_days": 10,
        "assigned_resources": {"管理人员": 5}
      }
    ]
  }
}
```

可选字段包括 `critical_path_tasks`、`key_milestones`、`resource_plan` 和 `risks`。`task_id` 的第一段数字用于识别施工分部，例如 `2.x.x` 表示地基与基础、`3.x.x` 表示主体结构。

## 配置

- Dify 对话框地址：修改 [config.py](config.py) 的 `IFRAME_URL`。
- 示例项目文件：修改 `DEMO_PLAN_PATH`，或替换 `名创优品.json`。
- 所有运行依赖都列在 `requirements.txt` 中。

## 开发检查

修改后可执行：

```bash
python -m compileall app.py core config.py
```
