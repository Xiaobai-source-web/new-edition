"""
进度计划智能助手（一体化版）
设计：
- 右侧为原生 Streamlit 聊天 UI，通过 Dify Chatflow API 直接调用工作流；
  工作流每次返回「文字描述 + structured_output JSON」双输出：
  · 文字描述 → 直接展示给用户（可复制）
  · structured_output → 网页侧自动校验 → 自动渲染甘特图/资源曲线 → 自动存入历史
- 左侧/下方保留 JSON 文件上传区 + 历史文件列表（方便回溯），仍可手动传入 JSON
- 历史文件以设备为单位保存；同名的文件视为同一文件，拒绝重复保存。
开发者：智建领航小组 · 华南理工大学
"""

import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from core.data import (
    calculate_daily_resources,
    get_critical_task_ids,
    get_section_mapping,
    load_json_from_file,
    load_json_from_upload,
    normalize_to_wrapped,
    tasks_to_dataframe,
    validate_data_structure,
)
from core.history import (
    check_history_file_exists,
    delete_history_file,
    get_history_json_files,
    save_file_unique,
)
from core.dify_client import call_dify_chatflow
from config import DEMO_PLAN_PATH

# ==================== 图表绘制 ====================

def _format_cn_date(dt):
    try:
        return f"{dt.year}年{dt.month}月{dt.day}日"
    except Exception:
        return str(dt)


def _format_short_date(dt):
    try:
        return f"{int(dt.month)}/{int(dt.day)}"
    except Exception:
        return str(dt)


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


def _build_gantt_data(tasks_df, section_filter=None):
    if section_filter and len(section_filter) > 0:
        filtered_df = tasks_df[tasks_df["section_code"].isin(section_filter)].copy()
    else:
        filtered_df = tasks_df.copy()
    if len(filtered_df) == 0:
        return None

    plot_df = filtered_df.copy()
    plot_df["Start"] = pd.to_datetime(plot_df["start_date"])
    plot_df["Finish"] = pd.to_datetime(plot_df["finish_date"])
    # 兼容 section_code 是纯数字或字符串（WP1-1）两种格式
    plot_df["_sec_code_raw"] = plot_df["section_code"].astype(str)
    plot_df["_sec_key"] = plot_df["_sec_code_raw"].apply(_section_sort_key)
    # task_id 也按自然排序，避免 1.10 排在 1.2 之前
    plot_df["_task_key"] = plot_df["task_id"].astype(str).apply(_section_sort_key)
    plot_df = plot_df.sort_values(["_sec_key", "_task_key"]).reset_index(drop=True)

    sections = []
    for code, grp in plot_df.groupby("_sec_code_raw", sort=False):
        sec_start = grp["Start"].min()
        sec_finish = grp["Finish"].max()
        sec_duration = (sec_finish - sec_start).days + 1
        sections.append({
            "code": str(code),
            "count": len(grp),
            "Start": sec_start,
            "Finish": sec_finish,
            "duration": sec_duration,
            "children": grp,
            "_sec_key": _section_sort_key(code),
        })
    sections.sort(key=lambda s: s["_sec_key"])

    ordered_rows = []
    for sec in sections:
        ordered_rows.append({
            "label": f"#{sec['code']}#（含{sec['count']}道工序）",
            "Start": sec["Start"],
            "Finish": sec["Finish"],
            "duration": sec["duration"],
            "row_type": "section",
            "bar_color": "#000000",
            "resources": None,
            "task_id": sec["code"],
            "task_name": f"分部{sec['code']}",
        })
        for _, t in sec["children"].iterrows():
            resources = t.get("assigned_resources", {})
            ordered_rows.append({
                "label": f"  {t['task_id']} {t['task_name']}",
                "Start": t["Start"],
                "Finish": t["Finish"],
                "duration": t["duration_days"],
                "row_type": "task",
                "bar_color": "#e74c3c",
                "resources": resources if isinstance(resources, dict) else {},
                "task_id": t["task_id"],
                "task_name": t["task_name"],
            })
    return pd.DataFrame(ordered_rows)


def _build_resource_hover_text(resources):
    if not resources or not isinstance(resources, dict):
        return "无资源配置"
    lines = [f"  {k}: {v}" for k, v in sorted(resources.items())]
    return "<br>".join(lines)


def create_gantt_chart(tasks_df, milestones, section_filter=None, show_milestones=True):
    rows_df = _build_gantt_data(tasks_df, section_filter=section_filter)
    if rows_df is None or len(rows_df) == 0:
        fig = go.Figure()
        fig.update_layout(title="施工进度甘特图（暂无数据）")
        return fig

    y_order = rows_df["label"].tolist()
    n_rows = len(rows_df)
    fig = go.Figure()

    black_rows = rows_df[rows_df["bar_color"] == "#000000"]
    red_rows = rows_df[rows_df["bar_color"] == "#e74c3c"]

    for label_set, color, name in [
        (black_rows, "#000000", "分部大类"),
        (red_rows, "#e74c3c", "分部小类"),
    ]:
        if len(label_set) == 0:
            continue
        x_durations = [
            (row["Finish"] - row["Start"]).total_seconds() * 1000
            for _, row in label_set.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=x_durations,
            y=label_set["label"].tolist(),
            base=[d.to_pydatetime() for d in label_set["Start"]],
            orientation="h",
            marker=dict(color=color, line=dict(color="#333", width=0.5)),
            name=name,
            showlegend=True,
            hoverinfo="skip",
            width=0.6,
        ))

    date_min = rows_df["Start"].min()
    date_max = rows_df["Finish"].max()
    all_dates = pd.date_range(start=date_min, end=date_max, freq="D")
    red_tasks_list = red_rows.to_dict("records")

    task_hover_cache = {}
    for t in red_tasks_list:
        res_text = _build_resource_hover_text(t["resources"])
        task_hover_cache[t["task_id"]] = (
            f"<b>{t['task_id']} {t['task_name']}</b><br>"
            f"工期：{t['duration']}天<br>"
            f"资源配置：{res_text}"
        )

    date_to_hover = {}
    for t in red_tasks_list:
        task_dates = pd.date_range(start=t["Start"], end=t["Finish"], freq="D")
        hover = task_hover_cache[t["task_id"]]
        for d in task_dates:
            d_key = d.strftime("%Y-%m-%d")
            if d_key not in date_to_hover:
                date_to_hover[d_key] = []
            date_to_hover[d_key].append(hover)

    daily_hover_texts = []
    for d in all_dates:
        d_key = d.strftime("%Y-%m-%d")
        if d_key in date_to_hover:
            daily_hover_texts.append("<br>".join(date_to_hover[d_key]))
        else:
            daily_hover_texts.append("当天无进行中的小类工序")

    fig.add_trace(go.Scatter(
        x=all_dates,
        y=[y_order[len(y_order) // 2]] * len(all_dates),
        mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=1),
        text=daily_hover_texts,
        hoverinfo="text",
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ))

    annotations = []
    for _, row in rows_df.iterrows():
        start_dt = row["Start"].to_pydatetime()
        finish_dt = row["Finish"].to_pydatetime()
        annotations.append(dict(
            x=start_dt, y=row["label"],
            text=_format_short_date(start_dt),
            showarrow=False, xanchor="right", yanchor="middle",
            xshift=-5,
            font=dict(size=9, color="#333", family="Microsoft YaHei"),
        ))
        annotations.append(dict(
            x=finish_dt, y=row["label"],
            text=_format_short_date(finish_dt),
            showarrow=False, xanchor="left", yanchor="middle",
            xshift=5,
            font=dict(size=9, color="#333", family="Microsoft YaHei"),
        ))

    tick_texts = []
    for _, row in rows_df.iterrows():
        if row["row_type"] == "section":
            tick_texts.append(
                f"<b>{row['label']}　{_format_cn_date(row['Start'])}–{_format_cn_date(row['Finish'])}　{row['duration']}d</b>"
            )
        else:
            tick_texts.append(
                f"{row['label']}　{_format_cn_date(row['Start'])}–{_format_cn_date(row['Finish'])}　{row['duration']}d"
            )

    if show_milestones and milestones:
        for milestone in milestones:
            md = pd.Timestamp(milestone["date"]).to_pydatetime()
            fig.add_trace(go.Scatter(
                x=[md], y=[y_order[0]], mode="markers",
                marker=dict(
                    symbol="diamond", size=14, color="#f39c12",
                    line=dict(color="#d68910", width=2)
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>里程碑：{milestone['name']}</b><br>"
                    f"日期：{milestone['date']}<extra></extra>"
                ),
            ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color="#f39c12", symbol="diamond"),
            name="里程碑", showlegend=True,
        ))

    height = max(600, n_rows * 28 + 200)
    fig.update_layout(
        title=dict(text="施工进度甘特图", font=dict(size=18, family="Microsoft YaHei"), x=0.5, xanchor="center"),
        barmode="overlay",
        height=height,
        margin=dict(l=80, r=80, t=100, b=100),
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Microsoft YaHei")),
        hoverlabel=dict(font=dict(family="Microsoft YaHei", size=12), bgcolor="white", bordercolor="#ddd"),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=y_order[::-1],
        ticktext=tick_texts[::-1],
        tickvals=y_order[::-1],
        tickfont=dict(size=10, family="Microsoft YaHei"),
        gridcolor="rgba(0,0,0,0.05)",
        showgrid=True, zeroline=False,
        side="left",
    )
    fig.update_xaxes(
        type="date",
        tickformat="%Y年%m月%d日",
        hoverformat="%Y年%m月%d日",
        tickangle=-45,
        gridcolor="rgba(0,0,0,0.1)",
        showgrid=True, zeroline=False,
        showticklabels=True,
        side="bottom",
        showspikes=True,
        spikecolor="#f1c40f",
        spikesnap="cursor",
        spikethickness=2,
        spikedash="solid",
    )
    fig.update_layout(
        xaxis2=dict(
            type="date",
            tickformat="%Y年%m月%d日",
            tickangle=-45,
            gridcolor="rgba(0,0,0,0)",
            showgrid=False, zeroline=False,
            side="top",
            overlaying="x",
            showticklabels=True,
            anchor="y",
            dtick="M1",
            showspikes=True,
            spikecolor="#f1c40f",
            spikesnap="cursor",
            spikethickness=2,
            spikedash="solid",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            font=dict(family="Microsoft YaHei", size=12),
            bgcolor="rgba(241, 196, 15, 0.95)",
            bordercolor="#f1c40f",
        ),
    )
    return fig


def create_manpower_curve(tasks_df):
    date_range, daily_manpower, daily_detail_texts = calculate_daily_resources(tasks_df)
    x_dates = [pd.Timestamp(d).to_pydatetime() for d in date_range]
    y_vals = [int(v) for v in daily_manpower]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_dates, y=y_vals, mode="lines", fill="tozeroy",
        line=dict(color="#27ae60", width=2),
        fillcolor="rgba(39, 174, 96, 0.3)",
        name="人力需求",
        customdata=daily_detail_texts,
        hovertemplate="<b>日期：%{x|%Y-%m-%d}</b><br>总人力：%{y}人<br><b>资源明细：</b><br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="资源负荷曲线（人力）", font=dict(size=16, family="Microsoft YaHei"), x=0.5),
        xaxis=dict(title="日期", type="date", tickformat="%Y-%m-%d", tickangle=-45, gridcolor="rgba(0,0,0,0.1)", showgrid=True),
        yaxis=dict(title="人力（人）", gridcolor="rgba(0,0,0,0.1)", rangemode="tozero"),
        height=400,
        margin=dict(l=60, r=30, t=60, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(font=dict(family="Microsoft YaHei")),
        showlegend=False,
    )
    return fig


# ==================== 界面展示 ====================

def render_project_overview(overview):
    st.markdown("### 📊 项目概览")
    st.markdown(
        f"<h2 style='font-weight: bold; color: #1e3a8a;'>{overview.get('project_name', '未知项目')}</h2>",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总工期", f"{overview.get('total_duration_days', 0)} 天")
    with col2:
        st.metric("计划开始", overview.get("planned_start_date", "暂无"))
    with col3:
        st.metric("计划完成", overview.get("planned_end_date", "暂无"))
    with col4:
        st.metric("关键路径工序数", f"{overview.get('critical_path_length', 0)} 项")


def _is_numeric_value(v):
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.replace(",", ""))
            return True
        except (ValueError, AttributeError):
            return False
    return False


def _render_centered_table(df):
    if df is None or len(df) == 0:
        st.info("暂无数据")
        return
    columns = df.columns.tolist()
    numeric_cols = set()
    for col in columns:
        col_values = df[col].dropna()
        if len(col_values) == 0:
            continue
        if all(_is_numeric_value(v) for v in col_values):
            numeric_cols.add(col)

    html_parts = ['<table class="centered-table">']
    html_parts.append("<thead><tr>")
    for col in columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead>")
    html_parts.append("<tbody>")
    for _, row in df.iterrows():
        html_parts.append("<tr>")
        for col in columns:
            value = row[col]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                value = ""
            if col in numeric_cols:
                if isinstance(value, float) and value == int(value):
                    value = f"{int(value):,}"
                elif isinstance(value, (int, float)):
                    value = f"{value:,}"
                html_parts.append(f'<td class="num-cell">{value}</td>')
            else:
                html_parts.append(f"<td>{value}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")
    css = """
    <style>
    .centered-table { width: 100%; border-collapse: collapse; font-family: "Microsoft YaHei", "微软雅黑", sans-serif; margin: 10px 0; }
    .centered-table th, .centered-table td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }
    .centered-table th { background-color: #f3f4f6; font-weight: 600; text-align: center; }
    .centered-table .num-cell { text-align: center; font-variant-numeric: tabular-nums; }
    .centered-table tbody tr:nth-child(even) { background-color: #fafafa; }
    .centered-table tbody tr:hover { background-color: #f0f9ff; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_resource_detail(task):
    st.markdown("### 🔧 工序资源配置详情")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**工序编号**\n\n{task.get('task_id', '暂无')}")
    with col2:
        st.info(f"**工序名称**\n\n{task.get('task_name', '暂无')}")
    with col3:
        st.info(f"**开始日期**\n\n{task.get('start_date', '暂无')}")
    with col4:
        st.info(f"**工期**\n\n{task.get('duration_days', 0)} 天")
    st.markdown("#### 资源配置")
    resources = task.get("assigned_resources", {})
    if isinstance(resources, dict) and resources:
        resource_data = [{"资源类型": k, "数量": v} for k, v in resources.items()]
        _render_centered_table(pd.DataFrame(resource_data))
    else:
        st.warning("该工序暂无资源配置信息")


def render_resource_plan(resource_plan):
    st.markdown("### 📦 资源计划概览")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总人工工日", f"{resource_plan.get('total_manpower_days', 0):,.0f} 工日")
    with col2:
        st.metric("峰值人力", f"{resource_plan.get('peak_manpower', 0)} 人")
    equipment_peak = resource_plan.get("equipment_peak", {})
    if equipment_peak:
        st.markdown("#### 主要设备峰值")
        _render_centered_table(pd.DataFrame([{"设备名称": k, "峰值数量": v} for k, v in equipment_peak.items()]))
    material_summary = resource_plan.get("material_summary", [])
    if material_summary:
        st.markdown("#### 主要材料汇总")
        df_materials = pd.DataFrame(material_summary)
        df_materials.columns = ["材料名称", "总数量", "单位"]
        _render_centered_table(df_materials)


def render_risks(risks):
    st.markdown("### ⚠️ 风险与应对措施")
    for i, risk in enumerate(risks, 1):
        with st.expander(f"风险 {i}：{risk.get('risk_name', '未知风险')}"):
            st.markdown(f"**应对措施**：{risk.get('mitigation', '暂无措施')}")


def render_milestones_table(milestones):
    st.markdown("### 🏁 关键里程碑")
    if milestones:
        df_milestones = pd.DataFrame(milestones)
        df_milestones["date"] = pd.to_datetime(df_milestones["date"])
        df_milestones = df_milestones.sort_values("date")
        df_milestones["date"] = df_milestones["date"].dt.strftime("%Y-%m-%d")
        df_milestones.columns = ["里程碑名称", "日期", "关联工序", "描述"]
        _render_centered_table(df_milestones)


# ==================== 导出 ====================

def export_combined_html(fig_gantt, fig_manpower, progress_bar=None):
    def update_progress(step, total, msg):
        if progress_bar:
            try:
                progress_bar.progress(step / total, text=msg)
            except TypeError:
                progress_bar.progress(step / total)
    try:
        update_progress(1, 3, "正在渲染甘特图...")
        gantt_html = pio.to_html(fig_gantt, full_html=False, include_plotlyjs=False)
        update_progress(2, 3, "正在渲染资源曲线...")
        manpower_html = pio.to_html(fig_manpower, full_html=False, include_plotlyjs=False)

        combined_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>施工进度计划图</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: #fff; font-family: Microsoft YaHei, sans-serif; }}
        .chart-container {{ width: 100%; max-width: 1600px; margin: 0 auto; }}
        .gantt-section {{ width: 100%; height: 1000px; }}
        .manpower-section {{ width: 100%; height: 500px; margin-top: 20px; }}
        h1 {{ text-align: center; color: #333; }}
    </style>
</head>
<body>
    <h1>施工进度计划图</h1>
    <div class="chart-container">
        <div class="gantt-section">{gantt_html}</div>
        <div class="manpower-section">{manpower_html}</div>
    </div>
</body>
</html>
"""
        update_progress(3, 3, "正在保存...")
        return combined_html.encode("utf-8")
    except Exception as e:
        if progress_bar:
            try:
                progress_bar.progress(0, text=f"失败：{str(e)}")
            except TypeError:
                progress_bar.progress(0)
        st.error(f"导出失败：{str(e)}")
        return None


def export_tasks_csv(tasks_df):
    export_df = tasks_df.copy()
    export_df["start_date"] = export_df["start_date"].dt.strftime("%Y-%m-%d")
    export_df["finish_date"] = export_df["finish_date"].dt.strftime("%Y-%m-%d")
    export_df["assigned_resources"] = export_df["assigned_resources"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
    )
    export_df.columns = ["工序编号", "工序名称", "开始日期", "完成日期", "工期(天)", "资源配置", "是否关键工序", "分部编码"]
    return export_df.to_csv(index=False, encoding="utf-8-sig")


# ==================== 完整渲染 ====================

def render_plan_full(data, current_version="default"):
    """完整渲染函数：使用标签页拆分。"""
    try:
        structured = data["structured_output"]
        overview = structured["overview"]
        all_tasks = structured["all_tasks_schedule"]
        critical_tasks = structured.get("critical_path_tasks", [])
        milestones = structured.get("key_milestones", [])
        resource_plan = structured.get("resource_plan", {})
        risks = structured.get("risks", [])

        critical_ids = get_critical_task_ids(critical_tasks)
        tasks_df = tasks_to_dataframe(all_tasks, critical_ids)
        section_mapping = get_section_mapping(all_tasks)

        render_project_overview(overview)
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 甘特图", "📈 资源曲线", "🔍 工序详情", "🏁 里程碑", "⚠️ 风险",
        ])

        with tab1:
            col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])
            with col_filter1:
                section_options = list(section_mapping.keys())
                selected_sections = st.multiselect(
                    "按分部工程筛选",
                    options=section_options,
                    format_func=lambda x: f"{x} - {section_mapping.get(x, '')}",
                    default=[],
                    help="功能：按施工分部工程（如施工准备、地基与基础、主体结构等）筛选甘特图中显示的工序。操作方法：点击下拉框选择一个或多个分部，图表将只显示所选分部的工序；不选则显示全部。",
                    key=f"section_filter_{current_version}",
                )
            with col_filter2:
                show_milestones = st.checkbox(
                    "显示里程碑",
                    value=True,
                    help="功能：在甘特图中用黄色菱形标记关键里程碑节点。操作方法：勾选后图表会自动刷新显示里程碑。",
                    key=f"show_milestones_{current_version}",
                )
            with col_filter3:
                show_resource_curve_tab = st.checkbox(
                    "同步显示资源曲线",
                    value=False,
                    help="功能：在甘特图标签页下方同步展示资源负荷曲线。操作方法：勾选后无需切换到资源曲线标签页即可查看。",
                    key=f"show_res_tab_{current_version}",
                )

            fig_gantt = create_gantt_chart(
                tasks_df,
                milestones,
                section_filter=selected_sections if selected_sections else None,
                show_milestones=show_milestones,
            )
            st.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{current_version}")

            col_export1, col_export2 = st.columns([1, 1])
            with col_export1:
                html_key = f"html_export_{current_version}"
                if html_key not in st.session_state:
                    st.session_state[html_key] = None
                if st.session_state[html_key] is None:
                    if st.button(
                        "🌐 生成网页",
                        help="功能：将当前甘特图和资源曲线打包生成一个独立的 HTML 网页文件。操作方法：点击后等待生成完成，出现下载按钮即可下载。",
                        key=f"btn_html_{current_version}",
                    ):
                        try:
                            progress_bar = st.progress(0, text="正在准备...")
                        except TypeError:
                            progress_bar = st.progress(0)
                        filtered_tasks = tasks_df[tasks_df["section_code"].isin(selected_sections)].copy() if selected_sections else tasks_df.copy()
                        fig_manpower_for_export = create_manpower_curve(filtered_tasks)
                        result = export_combined_html(fig_gantt, fig_manpower_for_export, progress_bar)
                        if result:
                            st.session_state[html_key] = result
                            st.success("网页生成成功！")
                            st.rerun()
                else:
                    st.download_button(
                        label="📥 下载网页",
                        data=st.session_state[html_key],
                        file_name=f"{overview.get('project_name', '进度计划')}_进度图.html",
                        mime="text/html",
                        help="功能：下载已生成的 HTML 网页文件。操作方法：点击后浏览器会自动下载 .html 文件，可用浏览器打开查看。",
                        key=f"dl_html_{current_version}",
                    )
            with col_export2:
                csv_data = export_tasks_csv(tasks_df)
                st.download_button(
                    label="📊 导出工序表",
                    data=csv_data,
                    file_name=f"{overview.get('project_name', '进度计划')}_工序表.csv",
                    mime="text/csv",
                    help="功能：导出当前进度计划的所有工序信息为 CSV 表格。操作方法：点击后浏览器会自动下载 .csv 文件，可用 Excel 打开。",
                    key=f"dl_csv_{current_version}",
                )

            if show_resource_curve_tab:
                st.markdown("---")
                st.subheader("📈 资源负荷曲线")
                filtered_tasks = tasks_df[tasks_df["section_code"].isin(selected_sections)].copy() if selected_sections else tasks_df.copy()
                fig_manpower = create_manpower_curve(filtered_tasks)
                st.plotly_chart(fig_manpower, use_container_width=True, key=f"manpower_sync_{current_version}")

        with tab2:
            st.info("显示每日人力需求变化趋势，悬停可查看每日资源分配详情")
            filtered_tasks = tasks_df.copy()
            fig_manpower = create_manpower_curve(filtered_tasks)
            st.plotly_chart(fig_manpower, use_container_width=True, key=f"manpower_{current_version}")

            st.markdown("---")
            render_resource_plan(resource_plan)

        with tab3:
            st.info("查询各工序的资源配置、工期等详细信息")
            task_options = [f"{t['task_id']} - {t['task_name']}" for t in all_tasks]
            selected_task_label = st.selectbox(
                "选择工序查看资源配置详情",
                options=task_options,
                index=0,
                help="功能：从所有工序中选择一个，查看其详细的资源配置（工种、人数、设备等）。操作方法：点击下拉框选择工序，下方会自动显示该工序的详细信息。",
                key=f"task_select_{current_version}",
            )
            selected_task_id = selected_task_label.split(" - ")[0]
            selected_task = next((t for t in all_tasks if t["task_id"] == selected_task_id), None)
            if selected_task:
                render_resource_detail(selected_task)

            st.markdown("---")
            st.subheader("📋 全部工序一览")
            display_df = tasks_df[["task_id", "task_name", "start_date", "finish_date", "duration_days", "is_critical"]].copy()
            display_df["start_date"] = display_df["start_date"].dt.strftime("%Y-%m-%d")
            display_df["finish_date"] = display_df["finish_date"].dt.strftime("%Y-%m-%d")
            display_df["is_critical"] = display_df["is_critical"].apply(lambda x: "✅ 是" if x else "")
            display_df.columns = ["工序编号", "工序名称", "开始日期", "完成日期", "工期(天)", "关键工序"]
            _render_centered_table(display_df)

        with tab4:
            st.info("关键节点和重要时间点的汇总")
            render_milestones_table(milestones)

        with tab5:
            st.info("显示项目可能面临的风险及对应的应对措施")
            if risks:
                render_risks(risks)
            else:
                st.info("当前项目暂无风险记录")

        return tasks_df, overview
    except Exception as e:
        st.error(f"渲染失败：{str(e)}")
        st.exception(e)
        return None, None


# ==================== AI 聊天模块（原生 UI + Dify API）====================


def _sanitize_plan_filename(text: str, fallback: str) -> str:
    """从 overview/project_name 或 AI 回复标题中提取一个干净的文件名。"""
    import re
    cleaned = re.sub(r"[\\/:*?\"<>|\r\n\t]", "", text).strip()
    cleaned = cleaned[:30].strip()
    if cleaned:
        return cleaned
    return fallback


def _consume_plan_and_save(wrapped_plan: dict, preferred_name: str) -> str | None:
    """把合法 plan 写入 session_state + 历史文件（去重）。返回最终使用的版本名，失败返回 None。"""
    overview = wrapped_plan.get("structured_output", {}).get("overview", {})
    project_name = str(overview.get("project_name") or preferred_name or "未命名项目")
    base_name = _sanitize_plan_filename(project_name, f"计划_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    version_name = base_name
    # 若已存在同名，追加后缀避免冲突（但历史中"完全同文件名"才拒绝；这里我们允许 session 内存中叠加）
    candidate = version_name
    counter = 1
    while candidate in st.session_state.data_versions:
        counter += 1
        candidate = f"{version_name}-{counter}"
    version_name = candidate

    file_bytes = json.dumps(wrapped_plan, ensure_ascii=False).encode("utf-8")
    file_name = f"{version_name}.json"
    if not check_history_file_exists(file_name):
        save_file_unique(file_name, file_bytes)
    st.session_state.data_versions[version_name] = wrapped_plan
    st.session_state.current_version = version_name
    st.session_state.history_page = 0
    return version_name


def render_chat_panel():
    """渲染右侧 AI 聊天 UI（原生 Streamlit）。

    - 发送用户消息 → 调用 Dify Chatflow → 流式返回文字 + structured_output
    - 文字：写入聊天记录，直接展示（可复制）
    - structured_output：自动校验 → 自动加入版本 → 页面自动展示图表
    - 快捷指令：一键填充常见请求模板
    - 进度反馈：实时显示 AI 当前执行到哪个节点
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 16px; border-radius: 8px; color: white; margin-bottom: 8px;">
        <h4 style="color: white; margin: 0;">🤖 AI 智能对话</h4>
        <p style="margin: 4px 0 0 0; opacity: 0.92; font-size: 0.85rem; line-height: 1.5;">
            描述项目需求或上传 Word/文档 → AI 返回<b>文字说明</b>，并自动在网页下方渲染甘特图与资源曲线。<br>
            💡 支持两种使用方式：① 直接生成新计划 ② 在已有计划基础上进行优化（如延期、压缩）。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 会话状态初始化
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # ====== 快捷指令按钮 ======
    st.markdown("""
    <div style="margin-bottom: 2px; color: #64748b; font-size: 0.8rem;">
        ⚡ 快捷指令（点击即可填充到输入框）：
    </div>
    """, unsafe_allow_html=True)
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        if st.button("📅 生成新计划", key="quick_new", help="功能：一键填入「生成新计划」的请求模板。\n操作方法：点击后模板文字会出现在输入框，您可在此基础上修改项目参数后发送。"):
            st.session_state.pending_input = (
                "请生成一份施工进度计划，项目信息如下：\n"
                "- 工程名称：XX项目\n"
                "- 总工期：240天\n"
                "- 计划开工日期：2026年9月1日\n"
                "- 建筑面积：约30000㎡\n"
                "- 结构形式：框架-剪力墙结构\n"
                "- 层数：地上28层/地下2层\n"
                "请生成完整的进度计划，包含所有工序、关键路径和资源分配。"
            )
            st.rerun()
    with quick_col2:
        has_current = bool(st.session_state.current_version)
        if st.button("⚡ 优化现有计划", key="quick_optimize", disabled=not has_current, help="功能：基于当前已加载的计划进行优化调整。\n操作方法：点击后填入优化请求模板，您可修改调整需求后发送。\n注意：需要先加载一个已有计划才能使用此功能。"):
            st.session_state.pending_input = (
                "请基于当前已加载的进度计划进行优化调整，调整需求如下：\n"
                "- 调整场景：XX原因导致工期延误\n"
                "- 延误天数：3天\n"
                "- 受影响工序：XX工序\n"
                "- 调整目标：在保证总工期不变的前提下，通过资源调整和工序优化追回延误工期\n"
                "请输出调整后的完整进度计划。"
            )
            st.rerun()
    with quick_col3:
        if st.button("🌧️ 模拟延误", key="quick_delay", disabled=not has_current, help="功能：模拟常见的工期延误场景，测试计划的抗风险能力。\n操作方法：点击后填入延误场景模板，您可修改延误原因和天数后发送。\n注意：需要先加载一个已有计划才能使用此功能。"):
            st.session_state.pending_input = (
                "模拟工期延误场景：\n"
                "- 延误原因：持续暴雨导致桩基施工无法进行\n"
                "- 受影响工序：桩基施工\n"
                "- 延误天数：3天\n"
                "请分析延误对总工期的影响，并生成调整后的进度计划，要求尽量追回延误工期。"
            )
            st.rerun()
    with quick_col4:
        if st.button("📊 查看当前计划", key="quick_view", disabled=not has_current, help="功能：显示当前已加载计划的概览信息。\n操作方法：点击后在对话中显示当前计划的项目名称、工期、关键路径等摘要。"):
            current = st.session_state.current_version
            if current and current in st.session_state.data_versions:
                wrapped = st.session_state.data_versions[current]
                inner = wrapped.get("structured_output", {})
                overview = inner.get("overview", {})
                tasks = inner.get("all_tasks_schedule", [])
                critical = inner.get("critical_path_tasks", [])
                milestones = inner.get("key_milestones", [])
                summary = (
                    f"📊 **当前计划概览：{current}**\n\n"
                    f"- 项目名称：{overview.get('project_name', '未知')}\n"
                    f"- 总工期：{overview.get('total_duration_days', '?')} 天\n"
                    f"- 计划工期：{overview.get('planned_start_date', '?')} → {overview.get('planned_end_date', '?')}\n"
                    f"- 工序总数：{len(tasks)} 道\n"
                    f"- 关键路径任务：{len(critical)} 道\n"
                    f"- 关键里程碑：{len(milestones)} 个\n"
                )
                st.session_state.chat_messages.append({"role": "assistant", "content": summary})
                st.rerun()

    # 处理快捷指令：点击后显示可编辑文本框，用户可修改模板后再发送
    pending_template = st.session_state.get("pending_input")
    if pending_template:
        # 编辑模式：显示预填充的文本框 + 发送/取消按钮（放前面，更靠近蓝框）
        edit_col, btn_col1, btn_col2 = st.columns([6, 1, 1])
        with edit_col:
            edited_text = st.text_area(
                "✏️ 编辑请求内容（可直接修改后发送）",
                value=pending_template,
                height=220,
                key="quick_edit_area",
                help="功能：快捷指令已填入模板，您可以在此基础上修改项目参数，然后点击「发送」。",
            )
        with btn_col1:
            send_clicked = st.button("📤 发送", key="quick_send", type="primary")
        with btn_col2:
            cancel_clicked = st.button("✖ 取消", key="quick_cancel")

        # 附件上传（编辑模式，放在编辑框下面，通栏全宽）
        attach_files = st.file_uploader(
            "📎 可选：附加项目文档（Word / TXT / JSON 等）",
            type=["docx", "txt", "json", "doc"],
            accept_multiple_files=True,
            key="chat_attachments_edit",
            help="功能：把项目参数 Word 文档或其它说明性文件传给 AI，辅助生成/调整进度计划。操作方法：点击选择文件，最多可上传 5 个；发送消息时会一并提交。",
        )

        if send_clicked and edited_text.strip():
            st.session_state.pop("pending_input", None)
            user_input = edited_text.strip()
        elif cancel_clicked:
            st.session_state.pop("pending_input", None)
            st.rerun()
        else:
            user_input = None
    else:
        # 正常模式：先附件上传，再使用 chat_input
        attach_files = st.file_uploader(
            "📎 可选：附加项目文档（Word / TXT / JSON 等）",
            type=["docx", "txt", "json", "doc"],
            accept_multiple_files=True,
            key="chat_attachments",
            help="功能：把项目参数 Word 文档或其它说明性文件传给 AI，辅助生成/调整进度计划。操作方法：点击选择文件，最多可上传 5 个；发送消息时会一并提交。",
        )
        user_input = st.chat_input(
            "在此描述项目需求，例如：生成XX项目进度计划，总工期240天 / 暴雨导致桩基延迟3天，请调整",
            key="chat_user_input",
        )

    # 聊天消息显示：不再使用固定高度的小滚动窗口，
    # 消息直接在页面流式展开，保证 AI 返回的完整文字全部可见、可随页面滚动。
    chat_area = st.container(border=False)
    with chat_area:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_input)

        with chat_area:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status_placeholder = st.empty()
                status_placeholder.caption("⏳ AI 开始处理...")

                # 进度回调：实时更新状态文字
                def _on_progress(text: str):
                    status_placeholder.caption(f"⏳ {text}")

                try:
                    current_plan = None
                    if st.session_state.current_version and st.session_state.current_version in st.session_state.data_versions:
                        current_plan = st.session_state.data_versions[st.session_state.current_version]

                    answer, structured_output, new_conv_id = call_dify_chatflow(
                        user_input,
                        files=attach_files if attach_files else None,
                        conversation_id=st.session_state.conversation_id,
                        current_plan_json=current_plan,
                        on_progress=_on_progress,
                    )
                    st.session_state.conversation_id = new_conv_id
                    status_placeholder.empty()

                    # 若 answer 是 dify_client 的卡住提示文案（以 ⚠️ 开头），直接显示给用户
                    display_text = answer or ""

                    if not display_text:
                        # 仍然没内容（极端情况），给一个兜底提示
                        display_text = (
                            "⚠️ AI 未返回内容。请检查：\n"
                            "① Dify 工作流最新版本是否已点「发布」；\n"
                            "② 代码节点输入/输出变量是否配置正确；\n"
                            "③ 工作流执行日志中是否有报错节点。"
                        )

                    message_placeholder.markdown(display_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": display_text})

                    # ====== 自动处理 structured_output ======
                    if structured_output:
                        wrapped = normalize_to_wrapped(structured_output)
                        is_valid, msg = validate_data_structure(wrapped)
                        if is_valid:
                            pref_name = wrapped["structured_output"]["overview"].get("project_name") or "AI生成计划"
                            version_name = _consume_plan_and_save(wrapped, preferred_name=pref_name)
                            if version_name:
                                st.success(f"✅ 已自动渲染：{version_name}（图表已在下方展示）")
                            else:
                                st.warning("⚠️ 进度计划已加载，但保存到历史文件失败")
                            st.rerun()
                        else:
                            st.error(f"❌ AI 返回的进度计划数据格式不完整：{msg}")
                    else:
           
