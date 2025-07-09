import streamlit as st
import pandas as pd
import json
from datetime import datetime

# --- Local Imports ---
from views.gantt_chart_view import create_enhanced_gantt_chart, create_order_gantt, create_room_capacity_charts
from managers.summary_dashboard_view import create_summary_dashboard_data, render_summary_dashboard
from heuristic_scheduler import *
import config
from utils.data_models import *
# --- INTEGRATION: Import the CapacityAnalyzer ---
from capacity_analyzer import CapacityAnalyzer

# --- Constants for Interactive Task List ---
class TaskStatus:
    BLOCKED = "Blocked"
    READY = "Ready to Start"
    IN_PROGRESS = "In Progress"
    COMPLETED = "✅ Completed"

# --- UI Helper: Interactive Task List (No changes) ---
def display_interactive_todo_list(schedule_result):
    st.markdown("---")
    st.markdown('<h2 class="section-header">✅ Interactive Task List</h2>', unsafe_allow_html=True)
    if not hasattr(schedule_result, 'task_lookup') or not hasattr(schedule_result, 'task_graph'):
        st.error("Schedule result is missing data for the to-do list.")
        return
    task_lookup = schedule_result.task_lookup
    task_graph = schedule_result.task_graph
    tasks_to_display = [
        task for task in task_lookup.values()
        if task.step and
           task.step.process_type != ProcessType.POST_PACKAGING and
           task.task_type not in [TaskType.CIP, TaskType.LOCKED]
    ]
    tasks_to_display.sort(key=lambda t: (t.start_time, t.step_idx))
    def handle_start(task_id):
        st.session_state.task_statuses[task_id] = TaskStatus.IN_PROGRESS
    def handle_complete(task_ids_to_complete):
        for task_id in task_ids_to_complete:
            st.session_state.task_statuses[task_id] = TaskStatus.COMPLETED
        for t_id, prereqs in task_graph.items():
            if st.session_state.task_statuses.get(t_id) == TaskStatus.BLOCKED:
                if all(st.session_state.task_statuses.get(p) == TaskStatus.COMPLETED for p in prereqs):
                    st.session_state.task_statuses[t_id] = TaskStatus.READY
    if st.session_state.get('last_schedule_id') != id(schedule_result):
        new_statuses = {}
        for task in tasks_to_display:
            new_statuses[task.task_id] = TaskStatus.BLOCKED
        for task in tasks_to_display:
            if not task_graph.get(task.task_id, []):
                new_statuses[task.task_id] = TaskStatus.READY
        st.session_state.task_statuses = new_statuses
        st.session_state.last_schedule_id = id(schedule_result)
    processed_packaging_jobs = set()
    for task in tasks_to_display:
        if task.step.process_type == ProcessType.PACKAGING:
            if task.job_id in processed_packaging_jobs:
                continue
            processed_packaging_jobs.add(task.job_id)
            packaging_group = [
                t for t in tasks_to_display
                if t.job_id == task.job_id and t.step.process_type == ProcessType.PACKAGING
            ]
            packaging_group.sort(key=lambda t: t.start_time)
            first_task = packaging_group[0]
            combined_id = f"{first_task.job_id}-PACK-GROUP"
            original_task_ids = [t.task_id for t in packaging_group]
            first_task_status = st.session_state.task_statuses.get(first_task.task_id)
            group_status = first_task_status
            if any(st.session_state.task_statuses.get(tid) == TaskStatus.IN_PROGRESS for tid in original_task_ids):
                group_status = TaskStatus.IN_PROGRESS
            elif all(st.session_state.task_statuses.get(tid) == TaskStatus.COMPLETED for tid in original_task_ids):
                group_status = TaskStatus.COMPLETED
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.markdown(f"**{first_task.job_id}**")
                st.caption(f"SKU: {first_task.sku_id}")
            with col2:
                st.markdown(f"**Step:** {first_task.step.step_id} (Combined)")
                st.caption(f"Resource: {first_task.assigned_resource_id}")
            with col3:
                if group_status == TaskStatus.READY:
                    st.button("Start Task 🟢", key=f"start_{combined_id}", on_click=handle_start, args=(first_task.task_id,), use_container_width=True)
                elif group_status == TaskStatus.IN_PROGRESS:
                    st.button("Complete Task 🔵", key=f"complete_{combined_id}", on_click=handle_complete, args=(original_task_ids,), use_container_width=True, type="primary")
                elif group_status == TaskStatus.COMPLETED:
                    st.success(TaskStatus.COMPLETED, icon="✅")
                else:
                    st.warning(TaskStatus.BLOCKED, icon="🔒")
        else:
            task_id = task.task_id
            status = st.session_state.task_statuses.get(task_id, TaskStatus.BLOCKED)
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.markdown(f"**{task.job_id}**")
                st.caption(f"SKU: {task.sku_id}")
            with col2:
                st.markdown(f"**Step:** {task.step.step_id}")
                st.caption(f"Resource: {task.assigned_resource_id}")
            with col3:
                if status == TaskStatus.READY:
                    st.button("Start Task 🟢", key=f"start_{task_id}", on_click=handle_start, args=(task_id,), use_container_width=True)
                elif status == TaskStatus.IN_PROGRESS:
                    st.button("Complete Task 🔵", key=f"complete_{task_id}", on_click=handle_complete, args=([task_id],), use_container_width=True, type="primary")
                elif status == TaskStatus.COMPLETED:
                    st.success(TaskStatus.COMPLETED, icon="✅")
                else:
                    st.warning(TaskStatus.BLOCKED, icon="🔒")

# --- UI Helper: Analytics Dashboard (No changes) ---
def display_analytics(schedule_result):
    st.markdown('<h2 class="section-header">Schedule Analytics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown("#### Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    packaging_volume = sum(
        t.volume for t in schedule_result.scheduled_tasks
        if t.process_type == ProcessType.PACKAGING
    )
    col1.metric("Total Production Volume", f"{packaging_volume:,.0f} L")
    if schedule_result.scheduled_tasks:
        schedule_end_time = max(t.end_time for t in schedule_result.scheduled_tasks)
        col2.metric("Schedule End Time", schedule_end_time.strftime("%a, %H:%M"))
    else:
        col2.metric("Schedule End Time", "N/A")
    utilization = getattr(schedule_result, 'resource_utilization', {})
    if utilization:
        non_room_utilization = {
            res: util for res, util in utilization.items()
            if not isinstance(config.ROOMS.get(res), Room)
        }
        if non_room_utilization:
            bottleneck_res = max(non_room_utilization, key=non_room_utilization.get)
            bottleneck_val = non_room_utilization[bottleneck_res]
            col3.metric("Bottleneck Resource", f"{bottleneck_res}", f"{bottleneck_val:.1f}% Utilization")
        else:
            col3.metric("Bottleneck Resource", "N/A")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 **Order Gantt**", "🏭 Resource Gantt", "🏠 Room Capacity"])
    with tab1:
        order_gantt = create_order_gantt(schedule_result)
        if order_gantt: st.plotly_chart(order_gantt, use_container_width=True)
        else: st.warning("Could not generate the Order Gantt chart.")
    with tab2:
        resource_gantt = create_enhanced_gantt_chart(schedule_result)
        if resource_gantt: st.plotly_chart(resource_gantt, use_container_width=True)
        else: st.warning("Could not generate the Resource Gantt chart.")
    with tab3:
        room_charts = create_room_capacity_charts(schedule_result)
        if room_charts:
            for room_id, fig in room_charts.items():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No room utilization data to display.")

# --- NEW: Capacity Sandbox View ---
def render_capacity_sandbox(analyzer):
    st.markdown('<h2 class="section-header">🔬 Capacity Sandbox</h2>', unsafe_allow_html=True)
    st.info("Use this tool to test different production mixes and find the theoretical maximum capacity and system bottlenecks.", icon="🧪")

    if 'sku_ratios' not in st.session_state:
        st.session_state.sku_ratios = {}

    available_skus = list(config.SKUS.keys())
    selected_skus = st.multiselect("Select SKUs for your test scenario:", available_skus)

    total_ratio = 0
    if selected_skus:
        st.markdown("##### Define Production Ratios")
        for sku in selected_skus:
            st.session_state.sku_ratios[sku] = st.number_input(f"Ratio for {sku} (%)", min_value=0.0, max_value=100.0, value=st.session_state.sku_ratios.get(sku, 0.0), step=1.0)
        
        total_ratio = sum(st.session_state.sku_ratios.get(sku, 0) for sku in selected_skus)
        
        if not 99.9 < total_ratio < 100.1 and total_ratio > 0:
            st.warning(f"Total ratio is {total_ratio:.1f}%. Please ensure it sums to 100%.")
        else:
            st.success(f"Total ratio is {total_ratio:.1f}%. Ready to run analysis.")

    if st.button("Analyze Max Capacity", use_container_width=True, type="primary", disabled=not (99.9 < total_ratio < 100.1)):
        with st.spinner("Running capacity analysis..."):
            final_ratio = {sku: st.session_state.sku_ratios[sku] / 100.0 for sku in selected_skus}
            report = analyzer.map_maximum_capacity(final_ratio)
            st.session_state.last_capacity_report = report
    
    if 'last_capacity_report' in st.session_state and st.session_state.last_capacity_report:
        st.markdown("---")
        st.markdown("### Analysis Report")
        report = st.session_state.last_capacity_report
        st.json(report)

# --- MODIFIED: Feasibility Checker View ---
def render_feasibility_checker(analyzer):
    st.markdown('<h2 class="section-header">📋 Feasibility Checker</h2>', unsafe_allow_html=True)
    st.info("Build a hypothetical production plan to check if it's feasible within a single day's capacity.", icon="🤔")

    if 'plan_df' not in st.session_state:
        st.session_state.plan_df = pd.DataFrame(columns=["sku_id", "quantity_kg", "type"])

    # --- MODIFICATION: Add title and button in columns ---
    col1, col2 = st.columns([2, 1.5])
    with col1:
        st.markdown("##### Build Production Plan")
    with col2:
        if st.button("📥 Load Current Indents", use_container_width=True, help="Loads all pending indents from the workbench into this table."):
            # Check if indents have been loaded in the main workbench
            if config.USER_INDENTS:
                # Convert indents to the DataFrame format
                indent_list = [
                    {"sku_id": i.sku_id, "quantity_kg": i.qty_required_liters, "type": "hard"}
                    for i in config.USER_INDENTS.values()
                ]
                st.session_state.plan_df = pd.DataFrame(indent_list)
                st.toast("Loaded current indents into the plan.", icon="✅")
                st.rerun() # Rerun to show the updated table
            else:
                st.toast("No indents loaded. Please load them in the 'Scheduling Workbench' tab first.", icon="ℹ️")

    edited_df = st.data_editor(
        st.session_state.plan_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "sku_id": st.column_config.SelectboxColumn("SKU", options=list(config.SKUS.keys()), required=True),
            "quantity_kg": st.column_config.NumberColumn("Quantity (kg)", min_value=1, required=True),
            "type": st.column_config.SelectboxColumn("Type", options=["hard", "soft"], required=True)
        }
    )
    st.session_state.plan_df = edited_df

    optimize = st.checkbox("Attempt to optimize hard constraints if plan is infeasible")

    if st.button("Check Plan Feasibility", use_container_width=True, type="primary", disabled=edited_df.empty):
        with st.spinner("Checking feasibility..."):
            plan_list = edited_df.to_dict('records')
            report = analyzer.check_feasibility(plan_list, optimize_hard_constraints=optimize)
            st.session_state.last_feasibility_report = report

    if 'last_feasibility_report' in st.session_state and st.session_state.last_feasibility_report:
        st.markdown("---")
        st.markdown("### Feasibility Report")
        report = st.session_state.last_feasibility_report
        
        status = report.get("overall_status")
        if status == "FEASIBLE":
            st.success(f"**Status:** {status}")
        elif status == "FEASIBLE_WITH_ADJUSTMENTS":
            st.warning(f"**Status:** {status}")
        else:
            st.error(f"**Status:** {status}")
        
        st.markdown(f"**System Bottleneck for this Plan:** `{report.get('system_bottleneck_for_this_plan')}`")
        
        st.dataframe(pd.DataFrame(report.get("analysis", [])), use_container_width=True, hide_index=True)


# --- Main Render Function ---
def render():
    st.markdown(" # Production Manager Dashboard")

    # --- 1. Initialize Session State & Analyzer ---
    for key in ['factory_data_loaded', 'indents_loaded', 'last_schedule_result', 'task_statuses', 'last_schedule_id', 'capacity_analyzer']:
        if key not in st.session_state:
            st.session_state[key] = None

    if not st.session_state.factory_data_loaded:
        with st.spinner("Loading factory configuration..."):
            try:
                # This assumes a DataLoader instance is created in the main app script
                if 'data_loader' not in st.session_state:
                    st.session_state.data_loader = DataLoader()
                data_loader = st.session_state.data_loader
                data_loader.load_sample_data()
                st.session_state.factory_data_loaded = True
                # --- INTEGRATION: Initialize Analyzer once ---
                st.session_state.capacity_analyzer = CapacityAnalyzer(
                    products=config.PRODUCTS, equipment=config.EQUIPMENTS, lines=config.LINES,
                    tanks=config.TANKS, skus=config.SKUS, rooms=config.ROOMS
                )
                st.toast("✅ Factory configuration loaded!", icon="🏭")
                st.rerun()
            except Exception as e:
                st.error(f"Fatal Error: Could not load factory data. Details: {e}")
                return
    
    analyzer = st.session_state.capacity_analyzer
    if not analyzer:
        st.warning("Capacity Analyzer not loaded. Please refresh the page.")
        return

    # --- 2. Main Tabbed Interface ---
    tab1, tab2, tab3 = st.tabs(["**Scheduling Workbench**", "**🔬 Capacity Sandbox**", "**📋 Feasibility Checker**"])

    with tab1:
        st.markdown('<h2 class="section-header">🚀 Scheduling Workbench</h2>', unsafe_allow_html=True)
        st.markdown('<div class="action-box">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Production Indents", use_container_width=True):
                data_loader = st.session_state.data_loader
                config.USER_INDENTS.clear()
                config.USER_INDENTS.update(data_loader.load_user_indents_with_fallback())
                st.session_state.indents_loaded = True
                st.session_state.last_schedule_result = None
                st.success("Indents are loaded. Ready to generate a schedule.")
                st.rerun()
        with col2:
            if st.button("Generate Schedule", use_container_width=True, type="primary", disabled=not st.session_state.indents_loaded):
                with st.spinner("🧠 Running scheduler..."):
                    scheduler = HeuristicScheduler(
                        indents=config.USER_INDENTS, skus=config.SKUS, products=config.PRODUCTS,
                        lines=config.LINES, tanks=config.TANKS, equipments=config.EQUIPMENTS, shifts=config.SHIFTS
                    )
                    result = scheduler.run_heuristic_scheduler()
                    st.session_state.last_schedule_result = result
                    st.success("Schedule generated successfully!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state.last_schedule_result:
            display_analytics(st.session_state.last_schedule_result)
            display_interactive_todo_list(st.session_state.last_schedule_result)
        elif st.session_state.indents_loaded:
            st.subheader("📋 Pending Indents Queue")
            if config.USER_INDENTS:
                indent_data = [{"Order": i.order_no, "SKU": i.sku_id, "Qty (L)": i.qty_required_liters, "Due Date": i.due_date.strftime("%Y-%m-%d")} for i in config.USER_INDENTS.values()]
                st.dataframe(pd.DataFrame(indent_data), use_container_width=True, hide_index=True)
            else:
                st.info("No indents found in the loaded data.")
        else:
            st.info("Welcome! Please load production indents to begin.")

    with tab2:
        render_capacity_sandbox(analyzer)

    with tab3:
        render_feasibility_checker(analyzer)