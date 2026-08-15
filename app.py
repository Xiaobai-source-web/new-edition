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
    plot_df["_sec_num"] = plot_df["section_code"].astype(int)
    plot_df = plot_df.sort_values(["_sec_num", "task_id"]).reset_index(drop=True)

    sections = []
    for code, grp in plot_df.groupby("_sec_num", sort=True):
        sec_start = grp["Start"].min()
        sec_finish = grp["Finish"].max()
        sec_duration = (sec_finish - sec_start).days + 1
        sections.append({
            "code": str(int(code)),
            "count": len(grp),
            "Start": sec_start,
            "Finish": sec_finish,
            "duration": sec_duration,
            "children": grp,
        })
    sections.sort(key=lambda s: int(s["code"]))

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
    <div style="margin-bottom: 6px; color: #64748b; font-size: 0.8rem;">
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

    # 聊天消息显示
    chat_area = st.container(height=480, border=False)
    with chat_area:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 附件上传（支持 word / 文本）
    attach_col, _ = st.columns([3, 2])
    with attach_col:
        attach_files = st.file_uploader(
            "📎 可选：附加项目文档（Word / TXT / JSON 等）",
            type=["docx", "txt", "json", "doc"],
            accept_multiple_files=True,
            key="chat_attachments",
            help="功能：把项目参数 Word 文档或其它说明性文件传给 AI，辅助生成/调整进度计划。操作方法：点击选择文件，最多可上传 5 个；发送消息时会一并提交。",
        )

    # 处理快捷指令：点击后显示可编辑文本框，用户可修改模板后再发送
    pending_template = st.session_state.get("pending_input")
    if pending_template:
        # 编辑模式：显示预填充的文本框 + 发送/取消按钮
        edit_col, btn_col1, btn_col2 = st.columns([6, 1, 1])
        with edit_col:
            edited_text = st.text_area(
                "✏️ 编辑请求内容（可直接修改后发送）",
                value=pending_template,
                height=150,
                key="quick_edit_area",
                help="功能：快捷指令已填入模板，您可以在此基础上修改项目参数，然后点击「发送」。",
            )
        with btn_col1:
            send_clicked = st.button("📤 发送", key="quick_send", type="primary")
        with btn_col2:
            cancel_clicked = st.button("✖ 取消", key="quick_cancel")

        if send_clicked and edited_text.strip():
            st.session_state.pop("pending_input", None)
            user_input = edited_text.strip()
        elif cancel_clicked:
            st.session_state.pop("pending_input", None)
            st.rerun()
        else:
            user_input = None
    else:
        # 正常模式：使用 chat_input
        user_input = st.chat_input(
            "在此描述项目需求，例如：生成XX项目进度计划，总工期240天 / 暴雨导致桩基延迟3天，请调整",
            key="chat_user_input",
        )

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
                    message_placeholder.markdown(answer if answer else "(AI 未返回文字)")

                    st.session_state.chat_messages.append({"role": "assistant", "content": answer or "(AI 未返回文字)"})

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
                        if st.session_state.current_version:
                            st.info("ℹ️ 本次对话未返回可渲染的进度计划数据（可能只是文字建议），继续查看当前已加载的计划")
                        else:
                            st.info("ℹ️ 本次对话仅返回文字说明。如需生成完整可渲染的进度计划，请补充项目参数（工程名称、总工期、开工日期等）后重新发送。")

                except Exception as exc:  # noqa: BLE001
                    status_placeholder.empty()
                    err_msg = f"❌ 调用失败：{exc}"
                    message_placeholder.error(err_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})


# ==================== 顶部全局说明横幅 ====================


def render_workflow_banner():
    """展示一体化流程说明。"""
    st.markdown("""
    <div style="background: #eef2ff; border-left: 4px solid #4f46e5; padding: 14px 18px; border-radius: 6px; margin-bottom: 14px;">
        <div style="font-weight: 600; color: #3730a3; margin-bottom: 6px;">📌 使用流程</div>
        <div style="color: #374151; font-size: 0.92rem; line-height: 1.8;">
            <b>① AI 对话（右侧）</b>：在右侧聊天框描述项目需求，或直接把 Word/参数文档拖到「附件」位置发送。AI 会返回：
            <ul style="margin: 4px 0 6px 24px;">
                <li><b>文字描述</b>（直接显示在对话中，可读可复制）</li>
                <li><b>进度计划 JSON 数据</b>（由网页自动处理，<b>您无需手动复制粘贴</b>）</li>
            </ul>
            <b>② 自动渲染（下方）</b>：网页收到 JSON 后，会自动校验格式、自动渲染甘特图/资源曲线、自动保存到左侧「历史文件」列表。<br>
            <b>③ 手动兜底（左侧）</b>：如需查看过往计划，或要手动传入外部 JSON 文件，可使用左侧上传区 / 历史文件列表。
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================== 主应用 ====================

def main():
    st.set_page_config(page_title="进度计划智能助手", page_icon="🏗️", layout="wide")

    st.markdown("""
    <style>
        .main .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%; }
        [data-testid="stMetricValue"] { font-size: 1.1rem; }
        h1, h2, h3 { font-family: "Microsoft YaHei", "微软雅黑", sans-serif; }
        .stChatMessage { padding: 0.5rem 0; }
        /* 文件上传组件中文化（只作用于拖放区，不影响已选文件列表） */
        [data-testid="stFileUploaderDropzone"] > div:first-child { color: rgba(49, 51, 63, 0.8); font-size: 1rem; line-height: 1.4; text-align: center; }
        [data-testid="stFileUploaderDropzone"] > div:first-child > div { display: none; }
        [data-testid="stFileUploaderDropzone"] > div:first-child > button { display: none; }
        [data-testid="stFileUploaderDropzone"] > div:first-child::before { content: "将文件拖放到此处"; color: rgba(49, 51, 63, 0.8); display: block; font-size: 1rem; line-height: 1.4; }
        [data-testid="stFileUploaderDropzone"] > div:first-child::after { content: "限制：每个文件200MB"; color: rgba(49, 51, 63, 0.6); display: block; font-size: 0.8em; margin-top: 4px; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

    # 顶部标题
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
        <svg width="56" height="56" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="46" fill="url(#grad1)" stroke="#2563eb" stroke-width="2"/>
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#1e40af;stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect x="22" y="35" width="56" height="45" fill="white" rx="3" opacity="0.95"/>
            <rect x="28" y="50" width="10" height="15" fill="#3b82f6" rx="1"/>
            <rect x="42" y="45" width="10" height="20" fill="#3b82f6" rx="1"/>
            <rect x="56" y="40" width="10" height="25" fill="#3b82f6" rx="1"/>
            <line x1="70" y1="35" x2="70" y2="18" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="70" y1="20" x2="85" y2="20" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <line x1="82" y1="20" x2="82" y2="26" stroke="white" stroke-width="1.5"/>
            <circle cx="82" cy="27" r="1.5" fill="#fbbf24"/>
        </svg>
        <div>
            <h1 style="margin: 0; font-size: 1.7rem; color: #1e3a8a;">进度计划智能助手</h1>
            <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.9rem;">AI 对话 + 进度计划可视化</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # 初始化 session_state
    if "data_versions" not in st.session_state:
        st.session_state.data_versions = {}
    if "current_version" not in st.session_state:
        st.session_state.current_version = None
    if "history_page" not in st.session_state:
        st.session_state.history_page = 0
    if "demo_loaded" not in st.session_state:
        st.session_state.demo_loaded = False
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None  # (type, msg)

    # 工作流说明
    render_workflow_banner()

    # ==================== 主体：左历史文件 + 右聊天框 ====================
    left_col, right_col = st.columns([2, 3], gap="large")

    with left_col:
        # ===== 上传区 =====
        st.markdown("""
        <div style="background: #f0f9ff; border-left: 4px solid #0284c7; padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
            <div style="font-weight: 600; color: #075985; margin-bottom: 4px;">📤 进度计划 JSON 文件上传</div>
            <div style="color: #374151; font-size: 0.85rem; line-height: 1.6;">
                <b>功能</b>：<b>手动兜底入口</b>。若您已有外部生成的进度计划 JSON 文件，或要从其它设备分享的 JSON 中查看计划，可在此传入，系统自动渲染甘特图、资源曲线等图表。日常使用推荐直接使用<b>右侧聊天框</b>，AI 输出后网页<b>自动渲染，无需手动传文件</b>。<br>
                <b>操作方法</b>：点击"Browse files"按钮或直接将 .json 文件拖入下方区域，松开即自动上传。<br>
                <b>规则</b>：同名文件视为同一文件，<b>不会重复保存</b>，传入同名文件会提示失败。如需更新内容，请更换文件名后重新上传。
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "上传JSON文件（手动兜底入口）",
            type=["json"],
            help="功能：手动兜底入口，用于传入外部/分享的进度计划 JSON 文件。日常使用直接在右侧聊天框与 AI 对话即可，自动渲染无需手动传文件。操作方法：点击选择文件或将 .json 文件拖入此区域。",
            key="file_uploader",
        )

        if uploaded_file is not None:
            # 先校验是否同名
            if check_history_file_exists(uploaded_file.name):
                st.session_state.upload_status = ("error", f"❌ 上传失败：历史中已存在同名文件 '{uploaded_file.name}'，按规则视为同一文件，不重复保存。请使用其他名称的文件。")
                st.error(st.session_state.upload_status[1])
            else:
                try:
                    file_content = uploaded_file.read()
                    data = load_json_from_upload(file_content)
                    is_valid, msg = validate_data_structure(data)
                    if is_valid:
                        wrapped = normalize_to_wrapped(data)
                        ok, info = save_file_unique(uploaded_file.name, file_content)
                        if ok:
                            version_name = uploaded_file.name.replace(".json", "")
                            st.session_state.data_versions[version_name] = wrapped
                            st.session_state.current_version = version_name
                            st.session_state.history_page = 0
                            st.session_state.upload_status = ("success", f"✅ 上传成功：'{uploaded_file.name}'")
                            st.success(st.session_state.upload_status[1])
                            st.rerun()
                        else:
                            st.session_state.upload_status = ("error", f"❌ {info}")
                            st.error(st.session_state.upload_status[1])
                    else:
                        st.session_state.upload_status = ("error", f"❌ 数据格式错误：{msg}")
                        st.error(st.session_state.upload_status[1])
                except Exception as e:
                    st.session_state.upload_status = ("error", f"❌ 解析失败：{str(e)}")
                    st.error(st.session_state.upload_status[1])

        # ===== 历史文件区 =====
        st.markdown("---")
        hist_title_col, hist_refresh_col = st.columns([4, 1])
        with hist_title_col:
            st.subheader("📂 历史文件")
        with hist_refresh_col:
            if st.button("🔄", key="refresh_history", help="功能：刷新左侧历史文件列表。操作方法：点击后重新读取本地存储目录中的所有 JSON 文件。"):
                st.session_state.history_page = 0
                st.rerun()
        history_files = get_history_json_files()

        if history_files:
            st.caption(f"共 {len(history_files)} 个文件（同名不重复保存）")

            page_size = 5
            total_pages = max(1, (len(history_files) + page_size - 1) // page_size)
            current_page = st.session_state.history_page
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(history_files))
            page_files = history_files[start_idx:end_idx]

            if total_pages > 1:
                col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                with col_p1:
                    if st.button("◀", disabled=(current_page == 0), key="prev_page", help="功能：查看上一页历史文件。操作方法：点击后显示前 5 个历史文件。"):
                        st.session_state.history_page = max(0, current_page - 1)
                        st.rerun()
                with col_p2:
                    st.caption(f"第 {current_page + 1}/{total_pages} 页")
                with col_p3:
                    if st.button("▶", disabled=(current_page >= total_pages - 1), key="next_page", help="功能：查看下一页历史文件。操作方法：点击后显示后 5 个历史文件。"):
                        st.session_state.history_page = min(total_pages - 1, current_page + 1)
                        st.rerun()

            for hf in page_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{hf['project_name']}**")
                    st.caption(f"📅 {hf['upload_time']}")
                with col2:
                    if st.button("加载", key=f"load_{hf['file_name']}", type="primary", help="功能：将该历史文件加载到当前会话并渲染图表。操作方法：点击后该计划的甘特图、资源曲线等图表会显示在页面下方。"):
                        try:
                            data = load_json_from_file(hf["file_path"])
                            is_valid, msg = validate_data_structure(data)
                            if is_valid:
                                wrapped = normalize_to_wrapped(data)
                                st.session_state.data_versions[hf["project_name"]] = wrapped
                                st.session_state.current_version = hf["project_name"]
                                st.success("✅ 已加载")
                                st.rerun()
                            else:
                                st.error(f"格式错误：{msg}")
                        except Exception as e:
                            st.error(f"加载失败：{str(e)}")
                with col3:
                    if st.button("🗑️", key=f"del_{hf['file_name']}", help="功能：永久删除该历史文件。操作方法：点击后文件将从本地存储中移除，不可恢复。"):
                        if delete_history_file(hf["file_path"]):
                            if hf["project_name"] in st.session_state.data_versions:
                                del st.session_state.data_versions[hf["project_name"]]
                            if st.session_state.current_version == hf["project_name"]:
                                remaining = list(st.session_state.data_versions.keys())
                                st.session_state.current_version = remaining[0] if remaining else None
                            st.success("已删除")
                            st.rerun()
        else:
            st.info("暂无历史文件")

    with right_col:
        render_chat_panel()

    # ==================== 参数规范与JSON格式规范（可折叠）====================
    st.markdown("---")
    st.markdown("""
    <div style="background: #f8fafc; border-left: 4px solid #64748b; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
        <div style="font-weight: 600; color: #475569;">📖 格式规范参考</div>
        <div style="color: #64748b; font-size: 0.85rem;">
            点击下方的折叠区域查看参数输入建议和 JSON 文件格式说明。这些规范仅供参考，帮助您更好地与 AI 沟通。
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📝 参数输入建议（点击展开）"):
        st.markdown("""
        <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 600; color: #b45309; margin-bottom: 8px;">🤖 AI 智能识别说明</div>
            <div style="color: #78350f; font-size: 0.9rem; line-height: 1.7;">
                <b>AI 会自动识别您的输入内容</b>，您可以选择以下任意方式提交项目信息：
                <ul style="margin: 8px 0; padding-left: 20px;">
                    <li><b>方式一：聊天框直接输入</b> — 在右侧聊天框中用自然语言描述项目需求，无需严格遵循格式</li>
                    <li><b>方式二：上传 Word 文件</b> — 将项目信息整理成 Word 文档，通过聊天框的附件功能上传</li>
                    <li><b>方式三：混合输入</b> — 先上传 Word 文档，再在聊天框补充说明或调整需求</li>
                </ul>
                <b>下方参数列表仅为建议</b>，帮助您梳理项目信息，<b>不强求必须全部填写或按格式输入</b>。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 参数输入建议表格
        st.markdown("""
        <div style="font-weight: 600; color: #1e3a8a; margin-bottom: 12px;">📋 建议输入的项目参数（按模块分类）</div>
        """, unsafe_allow_html=True)

        param_data = {
            "模块": [
                "基本信息", "基本信息", "基本信息", "基本信息", "基本信息", "基本信息", "基本信息", "基本信息", "基本信息",
                "技术约束", "技术约束", "技术约束", "技术约束",
                "资源约束", "资源约束", "资源约束", "资源约束",
                "组织管理", "组织管理", "组织管理", "组织管理",
                "外部约束", "外部约束", "外部约束", "外部约束",
                "调整用例", "调整用例", "调整用例", "调整用例", "调整用例"
            ],
            "参数名称": [
                "工程名称", "建设地点", "总建筑面积", "占地面积", "建筑组成", "结构形式", "层数与高度", "总工期", "计划开工/竣工日期",
                "基坑支护工艺", "桩基工艺", "特殊工艺", "主体结构工艺",
                "劳动力峰值", "主要工种及人数", "主要设备", "主要材料",
                "总承包单位", "项目负责人", "技术负责人", "资金条件",
                "政府管制要求", "强制里程碑", "空间占用规则", "周边关系",
                "调整场景名称", "触发条件", "偏差参数", "影响范围", "调整目标"
            ],
            "说明": [
                "项目的名称", "项目所在地", "如：301354.26㎡", "如：50000㎡", "如：商业裙楼+住宅塔楼", "如：框架-剪力墙结构", "如：地上28层/地下2层", "如：365天", "如：2026年7月1日 至 2027年7月1日",
                "如：地下连续墙+内支撑", "如：钻孔灌注桩", "如：爬模、装配式施工", "如：铝模+爬架",
                "高峰期需要的工人数量", "如：钢筋工50人、木工80人", "如：塔吊3台、施工电梯2台", "如：钢筋5000吨、水泥3000吨",
                "承担总承包的单位名称", "姓名+资质等级", "姓名+职称", "如：按进度付款、预付款比例",
                "如：夜间施工限制、渣土运输时间", "如：封顶日期、竣工验收日期", "如：场地分区占用时段", "如：周边建筑保护要求",
                "如：暴雨导致桩基延迟", "如：连续降雨超过3天", "如：工期延误5天", "如：桩基工程及后续工序", "如：压缩后续工期"
            ]
        }
        df_params = pd.DataFrame(param_data)
        _render_centered_table(df_params)

        st.markdown("""
        <div style="background: #eff6ff; border-radius: 6px; padding: 12px 16px; margin-top: 14px;">
            <div style="font-weight: 600; color: #1e40af; margin-bottom: 6px;">💡 使用提示</div>
            <ul style="color: #1e3a8a; font-size: 0.88rem; margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>您可以只提供<b>部分参数</b>，AI 会根据已有信息进行推理补全</li>
                <li>参数之间用<b>逗号、冒号、换行</b>分隔均可，AI 能自动识别</li>
                <li>如有<b>特殊要求</b>（如工期压缩、资源限制），请在输入中明确说明</li>
                <li>上传 Word 文件时，建议使用<b>清晰的标题和分段</b>，便于 AI 理解</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📋 JSON 文件格式说明（点击展开）"):
        st.markdown("""
        <div style="background: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
            <div style="font-weight: 600; color: #15803d;">💡 关于 JSON 格式</div>
            <div style="color: #166534; font-size: 0.88rem; line-height: 1.7;">
                AI 返回的进度计划数据会以 <b>JSON 格式</b> 呈现。您只需要：
                <ol style="margin: 8px 0; padding-left: 20px;">
                    <li>将 AI 返回的 JSON 文本<b>复制</b></li>
                    <li>保存为 <code>.json</code> 文件（可用记事本或其他文本编辑器）</li>
                    <li>通过左侧上传区传入本应用</li>
                </ol>
                系统会自动渲染甘特图、资源曲线等图表。下方的格式说明供参考，<b>您无需手动编写 JSON</b>。
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; color: #1e3a8a; line-height: 1.7;">
            <div style="font-weight: 600; margin-bottom: 4px;">✅ 实际支持的 JSON 结构</div>
            文件可使用 <code>{"structured_output": {...}}</code>（AI 返回文件的推荐形式），也可直接使用 <code>{...}</code>。<br>
            <b>真正必填的是 <code>overview</code> 和 <code>all_tasks_schedule</code></b>；<code>structured_output</code> 只是可选包装层，不是必填字段。<br>
            若上传完整的 Dify 返回结果，系统会自动读取其中的 <code>structured_output</code>，并忽略 <code>text</code>、<code>usage</code> 等附加字段。
        </div>
        <div style="font-weight: 600; color: #1e3a8a; margin-bottom: 12px;">📋 绘图所需字段</div>
        """, unsafe_allow_html=True)

        json_fields = {
            "字段（两种写法均可）": [
                "structured_output.overview / overview",
                "structured_output.all_tasks_schedule / all_tasks_schedule",
                "每个任务的 task_id、task_name",
                "每个任务的 start_date、finish_date、duration_days",
                "每个任务的 assigned_resources",
                "critical_path_tasks、key_milestones、resource_plan、risks",
            ],
            "格式与用途": [
                "项目名、总工期、计划开始日期、计划完成日期",
                "工序数组；用于生成甘特图、工序表和资源曲线",
                "工序编号与名称；编号建议使用 1.1.1 形式以识别分部",
                "日期须为 YYYY-MM-DD，例如 2026-03-01；工期为天数",
                "资源对象，例如 {\"钢筋工\": 20}；缺失时仍可绘制甘特图",
                "均为可选，用于展示关键路径、里程碑、资源计划和风险",
            ],
            "是否必需": [
                "✅ 必需",
                "✅ 必需",
                "✅ 必需",
                "✅ 必需",
                "可选",
                "可选",
            ],
        }
        df_json = pd.DataFrame(json_fields)
        _render_centered_table(df_json)

    # ==================== 下方：示范项目（名创优品）+ 用户加载的图表 ====================
    st.markdown("---")
    
    # 自动加载名创优品示范项目
    demo_path = DEMO_PLAN_PATH
    if demo_path.exists():
        if "demo_data" not in st.session_state:
            try:
                demo_data_raw = load_json_from_file(demo_path)
                is_valid, msg = validate_data_structure(demo_data_raw)
                if is_valid:
                    st.session_state.demo_data = normalize_to_wrapped(demo_data_raw)
                else:
                    st.warning(f"示范文件验证失败：{msg}")
            except Exception as e:
                st.warning(f"加载示范文件失败：{str(e)}")
        
        if "demo_data" in st.session_state and st.session_state.demo_data:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-left: 4px solid #0ea5e9; padding: 14px 18px; border-radius: 6px;">
                <div style="font-weight: 600; color: #0369a1; font-size: 1.1rem;">🏗️ 示范项目：名创优品施工进度计划</div>
                <div style="color: #075985; font-size: 0.9rem; margin-top: 4px;">
                    下方展示名创优品项目的完整进度计划图表，用于演示本网页的各项功能。<br>
                    您也可以上传自己的 JSON 文件查看其他项目的进度计划。
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_plan_full(st.session_state.demo_data, current_version="demo")

    # 如果用户加载了其他文件，也展示
    if st.session_state.current_version and st.session_state.current_version in st.session_state.data_versions:
        if st.session_state.current_version != "demo":
            st.markdown("---")
            st.markdown(f"""
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
                <div style="font-weight: 600; color: #b45309;">📊 已加载的进度计划：{st.session_state.current_version}</div>
            </div>
            """, unsafe_allow_html=True)
            current_data = st.session_state.data_versions[st.session_state.current_version]
            render_plan_full(current_data, current_version=st.session_state.current_version)
    elif not demo_path.exists():
        st.info("请通过左侧上传区传入 JSON 进度计划文件，或从历史文件列表加载")

    # ==================== 开发者印记 ====================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #94a3b8; font-size: 0.85rem;">
        <div style="font-weight: 600; color: #64748b; margin-bottom: 4px;">🏗️ 智建领航</div>
        <div>华南理工大学 · 进度计划智能助手</div>
        <div style="margin-top: 4px;">Powered by Dify AI + Streamlit + Plotly</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
