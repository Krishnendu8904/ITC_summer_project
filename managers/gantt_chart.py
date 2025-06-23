# managers/gantt_chart.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_models import SchedulingResult, TaskSchedule, CIPSchedule
from config import TANKS, LINES, EQUIPMENTS, ROOMS, USER_INDENTS
from datetime import datetime, timedelta
import re

def is_bulk_job(order_no: str) -> bool:
    """
    [CORRECTED] Determine if a task is a bulk job or an order job based on the order_no.
    Bulk jobs have an order_no that is a product name (e.g., 'Strawberry Yogurt')
    which does not exist as a key in the original USER_INDENTS.
    Order jobs have an order_no that matches a key in USER_INDENTS (e.g., 'ORD_101').
    """
    return order_no not in USER_INDENTS

def get_job_type_label(order_no: str) -> str:
    """
    [CORRECTED] Get a readable label for the job type based on the order_no.
    """
    return "Bulk Production" if is_bulk_job(order_no) else "Order Fulfillment"


def get_dark_theme_color_palette():
    """
    Returns a color palette optimized for dark backgrounds with similar shades for bulk/order distinction.
    """
    # Base colors - vibrant enough to stand out on dark background but still professional
    base_colors = [
        "#6B8DD6",  # Bright blue
        "#7BC97B",  # Fresh green
        "#D4A574",  # Warm orange
        "#6BBEBD",  # Teal
        "#B885D4",  # Purple
        "#85C4D4",  # Cyan
        "#D48598",  # Rose
        "#A5D474",  # Lime green
        "#D4B574",  # Gold
        "#74D4A5",  # Mint
    ]
    
    # Create bulk colors (more transparent/muted versions)
    bulk_colors = []
    # Create order colors (more saturated versions)
    order_colors = []
    
    for color in base_colors:
        # Convert hex to RGB
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        
        # Bulk: reduce saturation by moving towards gray (60% of original intensity)
        gray_factor = 0.6
        bulk_r = int(r * gray_factor + 128 * (1 - gray_factor))
        bulk_g = int(g * gray_factor + 128 * (1 - gray_factor))
        bulk_b = int(b * gray_factor + 128 * (1 - gray_factor))
        bulk_colors.append(f"#{bulk_r:02x}{bulk_g:02x}{bulk_b:02x}")
        
        # Order: increase saturation (keep original or slightly enhance)
        order_colors.append(color)
    
    return bulk_colors, order_colors

def configure_8hour_window(fig: go.Figure, schedule_start_time: datetime):
    """
    Configure the figure to show an 8-hour window with horizontal scrolling.
    """
    window_end = schedule_start_time + timedelta(hours=8)
    
    fig.update_layout(
        xaxis=dict(
            range=[schedule_start_time, window_end],
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='rgba(60, 60, 60, 0.8)',
                bordercolor='rgba(120, 120, 120, 0.8)',
                borderwidth=1
            ),
            type='date'
        ),
        # Add scrollbar styling
        dragmode='pan'
    )
    
    # Add zoom controls
    fig.update_layout(
        modebar_add=['pan2d', 'zoom2d', 'zoomin2d', 'zoomout2d', 'autoScale2d', 'resetScale2d']
    )
    
    return fig

def create_production_gantt(result: SchedulingResult):
    """
    [CORRECTED] Generates an enhanced production Gantt chart showing scheduled tasks per order.
    Uses the task's order_no to correctly distinguish between bulk and order jobs.
    print
    """
    if not result or not result.scheduled_tasks:
        return None

    tasks_data = []
    schedule_start_time = min(task.start_time for task in result.scheduled_tasks)
    
    for task in result.scheduled_tasks:
        # Use task.order_no for job type determination
        job_type = get_job_type_label(task.order_no)
        step_display = task.step_id
        
        # Create a more informative task label
        if is_bulk_job(task.order_no):
            task_label = f"{step_display}"
            order_display = f"{task.order_no} (Bulk)"
        else:
            task_label = f"{step_display} (Batch {task.batch_index + 1})"
            order_display = task.order_no
            
        duration_hours = (task.end_time - task.start_time).total_seconds() / 3600
        
        tasks_data.append({
            "Task": task_label,
            "Start": task.start_time,
            "Finish": task.end_time,
            "Resource": task.resource_id,
            "Order": order_display,
            "SKU": task.sku_id,
            "Batch": task.batch_index + 1,
            "JobType": job_type,
            "Volume": task.volume,
            "Duration": f"{duration_hours:.1f}h",
            "Priority": task.priority.name,
            "StepID": task.step_id
        })

    if not tasks_data:
        return None

    df = pd.DataFrame(tasks_data)
    
    # Get dark theme color palettes
    bulk_colors, order_colors = get_dark_theme_color_palette()
    
    # Create color mapping for different job types
    bulk_orders = df[df['JobType'] == 'Bulk Production']['Order'].unique()
    order_orders = df[df['JobType'] == 'Order Fulfillment']['Order'].unique()
    
    color_map = {}
    for i, order in enumerate(bulk_orders):
        color_map[order] = bulk_colors[i % len(bulk_colors)]
    for i, order in enumerate(order_orders):
        color_map[order] = order_colors[i % len(order_colors)]
    
    # Sort orders: bulk jobs first, then order jobs
    order_sort_key = lambda x: (0 if "(Bulk)" in x else 1, x)
    sorted_orders = sorted(df['Order'].unique(), key=order_sort_key, reverse=True)
    
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Order",
        color="Order",
        text="Task",
        hover_data=["Resource", "SKU", "Volume", "Duration", "Priority", "JobType"],
        title="Production Schedule - Orders & Bulk Jobs (8-Hour Scrollable View)",
        labels={"Order": "Production Order/Job"},
        category_orders={"Order": sorted_orders},
        color_discrete_map=color_map
    )

    # Configure 8-hour scrollable window
    fig = configure_8hour_window(fig, schedule_start_time)

    # Customize layout with dark theme styling
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Production Orders & Bulk Jobs",
        legend_title="Orders/Jobs",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color='#E0E0E0'),
        title_font_size=16,
        title_font_color='#FFFFFF',
        title_x=0.5,
        height=max(450, len(sorted_orders) * 40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color='#E0E0E0')
        ),
        plot_bgcolor='rgba(40, 40, 40, 0.8)',
        paper_bgcolor='rgba(20, 20, 20, 0.9)'
    )
    
    fig.update_yaxes(
        autorange="reversed",
        gridcolor='rgba(100, 100, 100, 0.3)',
        gridwidth=1,
        tickfont=dict(color='#E0E0E0')
    )
    
    fig.update_xaxes(
        gridcolor='rgba(100, 100, 100, 0.3)',
        gridwidth=1,
        tickfont=dict(color='#E0E0E0')
    )
    
    fig.update_traces(
        textposition='inside',
        textfont_size=9,
        textfont_color='rgba(255, 255, 255, 0.9)',
        hovertemplate="<br>".join([
            "<b>%{y}</b>",
            "Task: %{text}",
            "Resource: %{customdata[0]}",
            "SKU: %{customdata[1]}",
            "Volume: %{customdata[2]}L",
            "Duration: %{customdata[3]}",
            "Priority: %{customdata[4]}",
            "Type: %{customdata[5]}",
            "Start: %{x}",
            "End: %{base}",
            "<extra></extra>"
        ])
    )

    return fig


def create_resource_gantt(result: SchedulingResult, schedule_start_time: datetime, schedule_horizon_days: int):
    """
    [CORRECTED] Generates an enhanced resource utilization Gantt chart showing production, CIP, setup, and idle time.
    """
    if not result:
        return None
    
    all_resources = list(TANKS.keys()) + list(LINES.keys()) + list(EQUIPMENTS.keys()) + list(ROOMS.keys())
    schedule_end_time = schedule_start_time + timedelta(days=schedule_horizon_days)

    activities = []
    
    # Add scheduled production tasks with enhanced information
    for task in result.scheduled_tasks:
        # Use task.order_no for job type determination
        job_type = "Bulk" if is_bulk_job(task.order_no) else "Order"
        step_display = task.step_id
        duration_hours = (task.end_time - task.start_time).total_seconds() / 3600
        
        activities.append({
            "Resource": task.resource_id,
            "Start": task.start_time,
            "Finish": task.end_time,
            "Activity": f"Production ({job_type})",
            "Details": f"{task.order_no} - {step_display}",
            "Volume": task.volume,
            "Duration": f"{duration_hours:.1f}h",
            "SKU": task.sku_id,
            "Priority": task.priority.name,
            "ActivityType": "Production"
        })

    # Add CIP tasks with enhanced information
    for cip in result.CIP_schedules:
        duration_hours = cip.duration_minutes / 60
        activities.append({
            "Resource": cip.resource_id,
            "Start": cip.start_time,
            "Finish": cip.end_time,
            "Activity": "CIP",
            "Details": f"Clean-in-Place ({cip.CIP_type})",
            "Volume": "-",
            "Duration": f"{duration_hours:.1f}h",
            "SKU": "-",
            "Priority": "-",
            "ActivityType": "CIP"
        })

    # Calculate and add Idle time
    df_activities = pd.DataFrame(activities)
    all_gantt_data = []

    for resource in all_resources:
        resource_df = df_activities[df_activities['Resource'] == resource].sort_values(by="Start").reset_index(drop=True)
        
        last_finish_time = schedule_start_time

        for idx, row in resource_df.iterrows():
            # Add idle time before this task
            if row['Start'] > last_finish_time:
                idle_duration = (row['Start'] - last_finish_time).total_seconds() / 3600
                all_gantt_data.append({
                    "Resource": resource,
                    "Start": last_finish_time,
                    "Finish": row['Start'],
                    "Activity": "Idle",
                    "Details": "Resource Available",
                    "Volume": "-", "Duration": f"{idle_duration:.1f}h", "SKU": "-", "Priority": "-", "ActivityType": "Idle"
                })
            
            # Add the actual activity
            all_gantt_data.append(row.to_dict())
            last_finish_time = row['Finish']

        # Add final idle time until the end of the schedule horizon
        if last_finish_time < schedule_end_time:
            idle_duration = (schedule_end_time - last_finish_time).total_seconds() / 3600
            all_gantt_data.append({
                "Resource": resource,
                "Start": last_finish_time,
                "Finish": schedule_end_time,
                "Activity": "Idle",
                "Details": "Resource Available",
                "Volume": "-", "Duration": f"{idle_duration:.1f}h", "SKU": "-", "Priority": "-", "ActivityType": "Idle"
            })
            
    if not all_gantt_data:
        return None

    df_gantt = pd.DataFrame(all_gantt_data)
    
    # Dark theme color mapping with good contrast
    bulk_colors, order_colors = get_dark_theme_color_palette()
    
    color_map = {
        "Idle": "#2A2A2A",
        "CIP": "#4A6B8A",
        "Production (Bulk)": bulk_colors[0],
        "Production (Order)": order_colors[0]
    }
    
    # Group resources by type for better organization
    resource_categories = {
        "Tanks": [r for r in all_resources if r in TANKS],
        "Lines": [r for r in all_resources if r in LINES],
        "Equipment": [r for r in all_resources if r in EQUIPMENTS],
        "Rooms": [r for r in all_resources if r in ROOMS]
    }
    
    # Create ordered resource list by category
    ordered_resources = []
    for category, resources in resource_categories.items():
        if resources:  # Only add non-empty categories
            ordered_resources.extend(sorted(resources))
    
    fig = px.timeline(
        df_gantt,
        x_start="Start",
        x_end="Finish",
        y="Resource",
        color="Activity",
        title="Resource Utilization Timeline (8-Hour Scrollable View)",
        labels={"Resource": "Factory Resources"},
        color_discrete_map=color_map,
        hover_data=["Details", "Duration", "Volume", "SKU", "Priority"],
        category_orders={"Resource": ordered_resources[::-1]}  # Reverse for top-to-bottom display
    )

    # Configure 8-hour scrollable window
    fig = configure_8hour_window(fig, schedule_start_time)

    # Enhanced layout with dark theme styling
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Resources",
        legend_title="Activity Type",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color='#E0E0E0'),
        title_font_size=16,
        title_font_color='#FFFFFF',
        title_x=0.5,
        height=max(550, len(ordered_resources) * 35),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10, color='#E0E0E0')
        ),
        plot_bgcolor='rgba(40, 40, 40, 0.8)',
        paper_bgcolor='rgba(20, 20, 20, 0.9)'
    )
    
    fig.update_yaxes(
        gridcolor='rgba(100, 100, 100, 0.3)',
        gridwidth=1,
        tickfont=dict(color='#E0E0E0')
    )
    
    fig.update_xaxes(
        gridcolor='rgba(100, 100, 100, 0.3)',
        gridwidth=1,
        tickfont=dict(color='#E0E0E0')
    )
    
    # Enhanced hover template
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{y}</b>",
            "Activity: %{color}",
            "Details: %{customdata[0]}",
            "Duration: %{customdata[1]}",
            "Volume: %{customdata[2]}L",
            "SKU: %{customdata[3]}",
            "Priority: %{customdata[4]}",
            "Start: %{x}",
            "End: %{base}",
            "<extra></extra>"
        ])
    )

    return fig


def create_summary_dashboard(result: SchedulingResult):
    """
    [CORRECTED] Creates a summary dashboard with key metrics and insights.
    """
    if not result:
        return None
    
    # Calculate summary metrics using the corrected logic
    total_tasks = len(result.scheduled_tasks)
    bulk_tasks = sum(1 for task in result.scheduled_tasks if is_bulk_job(task.order_no))
    order_tasks = total_tasks - bulk_tasks
    
    total_cip_time = sum(cip.duration_minutes for cip in result.CIP_schedules) / 60  # Convert to hours
    total_production_time = sum((task.end_time - task.start_time).total_seconds() / 3600 
                               for task in result.scheduled_tasks)
    
    # Create a dashboard with dark theme styling
    metrics_text = f"""
    <div style="font-family: Segoe UI, Arial, sans-serif; padding: 25px; background-color: rgba(30, 30, 30, 0.95); border-radius: 8px; color: #E0E0E0; border: 1px solid rgba(80, 80, 80, 0.5);">
        <h3 style="color: #FFFFFF; margin-bottom: 20px; font-weight: 500;">Schedule Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 25px;">
            <div style="background-color: rgba(45, 45, 45, 0.8); padding: 20px; border-radius: 6px; border-left: 4px solid #6B8DD6;">
                <h4 style="color: #B0B0B0; margin-bottom: 15px; font-weight: 500;">Task Breakdown</h4>
                <p style="margin: 8px 0; color: #E0E0E0;">Total Tasks: <strong style="color: #FFFFFF;">{total_tasks}</strong></p>
                <p style="margin: 8px 0; color: #E0E0E0;">Bulk Production: <strong style="color: #FFFFFF;">{bulk_tasks}</strong></p>
                <p style="margin: 8px 0; color: #E0E0E0;">Order Fulfillment: <strong style="color: #FFFFFF;">{order_tasks}</strong></p>
            </div>
            <div style="background-color: rgba(45, 45, 45, 0.8); padding: 20px; border-radius: 6px; border-left: 4px solid #7BC97B;">
                <h4 style="color: #B0B0B0; margin-bottom: 15px; font-weight: 500;">Time Allocation</h4>
                <p style="margin: 8px 0; color: #E0E0E0;">Production Time: <strong style="color: #FFFFFF;">{total_production_time:.1f} hours</strong></p>
                <p style="margin: 8px 0; color: #E0E0E0;">CIP Time: <strong style="color: #FFFFFF;">{total_cip_time:.1f} hours</strong></p>
                <p style="margin: 8px 0; color: #E0E0E0;">Schedule Efficiency: <strong style="color: #FFFFFF;">{result.metrics.schedule_efficiency:.1%}</strong></p>
            </div>
        </div>
        <div style="margin-top: 20px; background-color: rgba(45, 45, 45, 0.8); padding: 20px; border-radius: 6px; border-left: 4px solid #D4A574;">
            <h4 style="color: #B0B0B0; margin-bottom: 15px; font-weight: 500;">Production Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <p style="margin: 8px 0; color: #E0E0E0;">Total Volume: <strong style="color: #FFFFFF;">{result.metrics.total_production_volume:,} L</strong></p>
                <p style="margin: 8px 0; color: #E0E0E0;">OTIF Rate: <strong style="color: #FFFFFF;">{result.metrics.otif_rate:.1%}</strong></p>
                <p style="margin: 8px 0; color: #E0E0E0;">Solve Time: <strong style="color: #FFFFFF;">{result.solve_time:.2f} seconds</strong></p>
            </div>
        </div>
    </div>
    """
    
    return metrics_text