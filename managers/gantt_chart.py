# managers/gantt_chart.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_models import SchedulingResult, TaskSchedule, CIPSchedule
from config import TANKS, LINES, EQUIPMENTS, ROOMS, USER_INDENTS
from datetime import datetime, timedelta
import re
import streamlit as st

def is_bulk_job(order_no: str) -> bool:
    """
    Determine if a task is a bulk job or an order job based on the order_no.
    """
    return order_no not in USER_INDENTS

def get_job_type_label(order_no: str) -> str:
    """
    Get a readable label for the job type based on the order_no.
    """
    return "Bulk Production" if is_bulk_job(order_no) else "Order Fulfillment"


def get_dark_theme_color_palette():
    """
    Returns a professional color palette optimized for dark backgrounds.
    """
    return {
        "order": "#4ECDC4",       # Teal for customer orders
        "bulk": "#45B7D1",        # Blue for bulk production
        "cip": "#96CEB4",         # Muted green for CIP
        "idle": "rgba(58, 58, 90, 0.6)", # Dark gray for idle time
        "highlight": "#FF6B6B"    # Red/Coral for highlights or alerts
    }

def configure_gantt_xaxis(fig: go.Figure, schedule_start_time: datetime):
    """
    Configure the figure's x-axis for a gantt chart, including 15-min ticks.
    """
    window_end = schedule_start_time + timedelta(hours=8)
    
    fig.update_xaxes(
        range=[schedule_start_time, window_end],
        tickformat='%H:%M',
        dtick=900000,  # 15 minutes in milliseconds
        gridcolor='rgba(255, 255, 255, 0.1)',
        gridwidth=1,
        tickfont=dict(color='#E0E0E0', size=10)
    )

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='rgba(60, 60, 60, 0.8)',
                bordercolor='rgba(120, 120, 120, 0.8)',
                borderwidth=1
            ),
            type='date'
        ),
        dragmode='pan'
    )
    return fig

def create_production_gantt(result: SchedulingResult):
    """
    Generates an enhanced production Gantt chart showing scheduled tasks per order/job.
    """
    if not result or not result.scheduled_tasks:
        return None

    tasks_data = []
    if not result.scheduled_tasks: return None
    schedule_start_time = min(task.start_time for task in result.scheduled_tasks)
    
    for task in result.scheduled_tasks:
        job_type = get_job_type_label(task.order_no)
        task_label = f"{task.step_id}"
        order_display = f"{task.order_no} (Bulk)" if is_bulk_job(task.order_no) else task.order_no
            
        tasks_data.append({
            "Task": task_label, "Start": task.start_time, "Finish": task.end_time,
            "Resource": task.resource_id, "Order": order_display, "SKU": task.sku_id,
            "JobType": job_type, "Volume": task.volume
        })

    if not tasks_data: return None

    df = pd.DataFrame(tasks_data)
    color_palette = get_dark_theme_color_palette()
    
    # Simple color map for bulk vs order
    color_map = {
        "Bulk Production": color_palette['bulk'],
        "Order Fulfillment": color_palette['order']
    }
    
    order_sort_key = lambda x: (0 if "(Bulk)" in x else 1, x)
    sorted_orders = sorted(df['Order'].unique(), key=order_sort_key, reverse=True)
    
    fig = px.timeline(
        df, x_start="Start", x_end="Finish", y="Order", color="JobType", text="Task",
        hover_data=["Resource", "SKU", "Volume"], title="Production Schedule by Job (15-min blocks)",
        labels={"Order": "Production Job", "JobType": "Job Type"},
        category_orders={"Order": sorted_orders}, color_discrete_map=color_map
    )

    fig = configure_gantt_xaxis(fig, schedule_start_time)
    fig.update_layout(
        xaxis_title="Timeline", yaxis_title="Job / Order", legend_title="Job Type",
        font=dict(family="Inter, sans-serif", size=11, color='#E0E0E0'),
        title_font_size=18, title_x=0.5,
        height=max(450, len(sorted_orders) * 40),
        plot_bgcolor='rgba(30, 30, 47, 0.9)', paper_bgcolor='rgba(15, 15, 26, 1)'
    )
    fig.update_yaxes(autorange="reversed", gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#E0E0E0'))
    fig.update_traces(textposition='inside', textfont_size=9)
    return fig


def create_resource_gantt(result: SchedulingResult, schedule_start_time: datetime, schedule_horizon_days: int):
    """
    [ENHANCED] Generates separate Gantt charts for each resource category (Tanks, Lines, etc.)
    with clear visual distinction for bulk vs. order tasks.
    """
    if not result: return None
    
    all_resources = list(TANKS.keys()) + list(LINES.keys()) + list(EQUIPMENTS.keys()) + list(ROOMS.keys())
    schedule_end_time = schedule_start_time + timedelta(days=schedule_horizon_days)

    activities = []
    for task in result.scheduled_tasks:
        job_type_label = get_job_type_label(task.order_no)
        activities.append({
            "Resource": task.resource_id, "Start": task.start_time, "Finish": task.end_time,
            "Activity": job_type_label, "Details": f"{task.order_no} - {task.step_id}", "SKU": task.sku_id
        })

    for cip in result.CIP_schedules:
        activities.append({
            "Resource": cip.resource_id, "Start": cip.start_time, "Finish": cip.end_time,
            "Activity": "CIP", "Details": f"Clean-in-Place ({cip.CIP_type})", "SKU": "-"
        })

    df_activities = pd.DataFrame(activities)
    all_gantt_data = []

    for resource in all_resources:
        resource_df = df_activities[df_activities['Resource'] == resource].sort_values(by="Start")
        last_finish_time = schedule_start_time
        for _, row in resource_df.iterrows():
            if row['Start'] > last_finish_time:
                all_gantt_data.append({"Resource": resource, "Start": last_finish_time, "Finish": row['Start'], "Activity": "Idle"})
            all_gantt_data.append(row.to_dict())
            last_finish_time = row['Finish']
        if last_finish_time < schedule_end_time:
            all_gantt_data.append({"Resource": resource, "Start": last_finish_time, "Finish": schedule_end_time, "Activity": "Idle"})
            
    if not all_gantt_data: return None

    df_gantt = pd.DataFrame(all_gantt_data)
    
    color_palette = get_dark_theme_color_palette()
    color_map = {
        "Idle": color_palette['idle'],
        "CIP": color_palette['cip'],
        "Bulk Production": color_palette['bulk'],
        "Order Fulfillment": color_palette['order']
    }
    
    resource_categories = {"Tanks": TANKS, "Lines": LINES, "Equipment": EQUIPMENTS, "Rooms": ROOMS}
    
    # Instead of one figure, we now call st.plotly_chart inside the loop
    for category_name, resource_dict in resource_categories.items():
        if not resource_dict: continue

        category_resources = list(resource_dict.keys())
        df_category = df_gantt[df_gantt['Resource'].isin(category_resources)]

        if df_category.empty: continue

        st.markdown(f'<h3 class="section-header" style="font-size:1.4rem; margin-top:1.5rem;">{category_name} Utilization</h3>', unsafe_allow_html=True)
        
        fig = px.timeline(
            df_category, x_start="Start", x_end="Finish", y="Resource", color="Activity",
            hover_data=["Details", "SKU"], color_discrete_map=color_map,
            category_orders={"Resource": sorted(category_resources, reverse=True)}
        )
        
        fig = configure_gantt_xaxis(fig, schedule_start_time)
        fig.update_layout(
            title=None, xaxis_title=None, yaxis_title=None,
            font=dict(family="Inter, sans-serif", size=11, color='#E0E0E0'),
            height=max(150, len(category_resources) * 45), showlegend=False,
            plot_bgcolor='rgba(30, 30, 47, 0.9)', paper_bgcolor='rgba(15, 15, 26, 1)'
        )
        fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#E0E0E0'))
        
        st.plotly_chart(fig, use_container_width=True)

    # Returning None as the charts are now rendered directly
    return None

def create_summary_dashboard(result: SchedulingResult):
    """
    Creates a summary dashboard with key metrics and insights.
    """
    if not result: return ""
    
    total_tasks = len(result.scheduled_tasks)
    bulk_tasks = sum(1 for task in result.scheduled_tasks if is_bulk_job(task.order_no))
    order_tasks = total_tasks - bulk_tasks
    total_cip_time = sum(cip.duration_minutes for cip in result.CIP_schedules) / 60
    total_production_time = sum((task.end_time - task.start_time).total_seconds() / 3600 for task in result.scheduled_tasks)
    
    total_volume = result.metrics.total_production_volume if hasattr(result, 'metrics') else 0
    efficiency = f"{result.metrics.schedule_efficiency:.1%}" if hasattr(result, 'metrics') else "N/A"
    otif_rate = f"{result.metrics.otif_rate:.1%}" if hasattr(result, 'metrics') else "N/A"
    solve_time = f"{result.solve_time:.2f} s" if hasattr(result, 'solve_time') else "N/A"

    return f"""
    <div style="font-family: 'Inter', sans-serif; padding: 25px; background: rgba(30, 30, 47, 0.9); backdrop-filter: blur(10px); border-radius: 15px; color: #EAEAEA; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);">
        <h3 style="color: #FFFFFF; margin-bottom: 20px; font-weight: 600;">Schedule Performance Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
            <div style="background-color: rgba(45, 45, 65, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #4ECDC4;">
                <h4 style="color: #B0B0B8; margin-top: 0; margin-bottom: 15px; font-weight: 500;">Task Breakdown</h4>
                <p style="margin: 8px 0; font-size: 1rem;">Total Tasks: <strong style="color: #FFFFFF; font-size: 1.1rem;">{total_tasks}</strong></p>
                <p style="margin: 8px 0; font-size: 1rem;">Bulk Production: <strong style="color: #FFFFFF; font-size: 1.1rem;">{bulk_tasks}</strong></p>
                <p style="margin: 8px 0; font-size: 1rem;">Order Fulfillment: <strong style="color: #FFFFFF; font-size: 1.1rem;">{order_tasks}</strong></p>
            </div>
            <div style="background-color: rgba(45, 45, 65, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #45B7D1;">
                <h4 style="color: #B0B0B8; margin-top: 0; margin-bottom: 15px; font-weight: 500;">Time Allocation</h4>
                <p style="margin: 8px 0; font-size: 1rem;">Production Time: <strong style="color: #FFFFFF; font-size: 1.1rem;">{total_production_time:.1f} hrs</strong></p>
                <p style="margin: 8px 0; font-size: 1rem;">CIP Time: <strong style="color: #FFFFFF; font-size: 1.1rem;">{total_cip_time:.1f} hrs</strong></p>
                <p style="margin: 8px 0; font-size: 1rem;">Schedule Efficiency: <strong style="color: #FFFFFF; font-size: 1.1rem;">{efficiency}</strong></p>
            </div>
            <div style="background-color: rgba(45, 45, 65, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #FF6B6B;">
                <h4 style="color: #B0B0B8; margin-top: 0; margin-bottom: 15px; font-weight: 500;">Production Summary</h4>
                <p style="margin: 8px 0; font-size: 1rem;">Total Volume: <strong style="color: #FFFFFF; font-size: 1.1rem;">{total_volume:,.0f} L</strong></p>
                <p style="margin: 8px 0; font-size: 1rem;">OTIF Rate: <strong style="color: #FFFFFF; font-size: 1.1rem;">{otif_rate}</strong></p>
                <p style="margin: 8px 0; font-size: 1rem;">Solver Time: <strong style="color: #FFFFFF; font-size: 1.1rem;">{solve_time}</strong></p>
            </div>
        </div>
    </div>
    """
