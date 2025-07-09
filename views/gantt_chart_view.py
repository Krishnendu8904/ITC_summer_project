import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# Assume data models and config are available
from utils.data_models import SchedulingResult, Room, ProcessType
import config


def create_enhanced_gantt_chart(schedule_result, color_by='resource'):
    """
    Creates an enhanced Gantt chart based on the TaskSchedule class, colored by resource or SKU.
    """
    if not schedule_result or not schedule_result.scheduled_tasks:
        return None

    tasks_data = []
    for task in schedule_result.scheduled_tasks:
        tasks_data.append(dict(
            Task=f"{task.order_no} ({task.step_id})",
            Start=task.start_time,
            Finish=task.end_time,
            Resource=task.resource_id,
            SKU=task.sku_id,
            Tooltip=f"<b>Order</b>: {task.order_no}<br><b>Step</b>: {task.step_id}<br><b>Resource</b>: {task.resource_id}"
        ))

    if not tasks_data:
        return None

    df = pd.DataFrame(tasks_data)
    color_map_key = 'Resource' if color_by == 'resource' else 'SKU'

    fig = px.timeline(
        df, x_start="Start", x_end="Finish", y="Resource", color=color_map_key,
        hover_name="Task", custom_data=["Tooltip"], title="<b>Resource Utilization Gantt Chart</b>"
    )

    fig.update_layout(
        xaxis_title="Timeline", yaxis_title="Resource", legend_title=f"<b>Color by {color_map_key}</b>",
        font=dict(family="Inter, sans-serif"), hoverlabel=dict(bgcolor="white", font_size=14),
        title_x=0.5, yaxis={'categoryorder':'total ascending'}
    )
    fig.update_traces(hovertemplate='%{customdata[0]}')
    return fig


def create_order_gantt(schedule_result: SchedulingResult):
    """
    Creates a Gantt chart where each row is an order, and tasks are colored by production stage.
    """
    # --- Add this line for debugging ---
    print(f"Gantt chart received {len(schedule_result.scheduled_tasks)} tasks to plot.")

    if not schedule_result or not schedule_result.scheduled_tasks:
        return None

    tasks_data = []
    stage_colors = {
        ProcessType.PREPROCESSING.value: "#45B7D1",
        ProcessType.PROCESSING.value: "#FFA07A",
        ProcessType.PACKAGING.value: "#4ECDC4",
        ProcessType.POST_PACKAGING.value: "#96CEB4",
        "CIP": "#BDBDBD",
        "LOCKED": "#F7DC6F",
        "Task": "#CCCCCC" # Default color
    }

    for task in schedule_result.scheduled_tasks:
        stage = "Task" # Default stage name
        
        # This logic correctly reads the process_type from the TaskSchedule object
        if hasattr(task, 'process_type') and task.process_type:
            stage = task.process_type.value
        elif hasattr(task, 'task_type') and task.task_type:
            stage = task.task_type.name

        tasks_data.append(dict(
            Order=task.order_no,
            Start=task.start_time,
            Finish=task.end_time,
            Stage=stage,
            Resource=task.resource_id,
            Details=f"{task.step_id} on {task.resource_id}"
        ))

    if not tasks_data:
        return None

    df = pd.DataFrame(tasks_data)
    
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Order",
        color="Stage",
        hover_name="Details",
        title="<b>Production Schedule by Order</b>",
        color_discrete_map=stage_colors,
        category_orders={"Order": sorted(df['Order'].unique(), reverse=True)}
    )
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Order Number",
        legend_title="<b>Production Stage</b>",
        font=dict(family="Inter, sans-serif"),
        title_x=0.5
    )
    fig.update_yaxes(autorange="reversed")
    return fig

def create_room_capacity_charts(schedule_result: SchedulingResult):
    """
    NEW: Creates line charts showing capacity utilization over time for each room.
    """
    if not schedule_result or not schedule_result.scheduled_tasks:
        return None

    room_ids = [res_id for res_id, res_obj in config.ROOMS.items() if isinstance(res_obj, Room)]
    if not room_ids:
        return None

    figs = {}
    for room_id in room_ids:
        room_obj = config.ROOMS[room_id]
        # Filter tasks for the current room
        room_tasks = [task for task in schedule_result.scheduled_tasks if task.resource_id == room_id]

        if not room_tasks:
            continue

        events = []
        for task in room_tasks:
            sku = config.SKUS.get(task.sku_id)
            if sku and hasattr(sku, 'inventory_size'):
                capacity_consumed = task.volume * sku.inventory_size
                events.append((task.start_time, capacity_consumed))
                events.append((task.end_time, -capacity_consumed))

        if not events:
            continue

        events.sort()
        
        plot_data = []
        current_capacity = 0
        plot_data.append({'time': events[0][0] - timedelta(minutes=1), 'capacity': 0})

        for time_event, capacity_change in events:
            if plot_data:
                plot_data.append({'time': time_event - timedelta(seconds=1), 'capacity': current_capacity})
            current_capacity += capacity_change
            plot_data.append({'time': time_event, 'capacity': current_capacity})

        if not plot_data:
            continue

        df = pd.DataFrame(plot_data)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['time'], y=df['capacity'], mode='lines', line_shape='hv',
            name='Used Capacity (EUI)', fill='tozeroy'
        ))

        fig.add_hline(
            y=room_obj.capacity_units, line_dash="dash", line_color="red",
            annotation_text=f"Max Capacity: {room_obj.capacity_units} EUI",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title=f"<b>Capacity Utilization for {room_id}</b>",
            xaxis_title="Time", yaxis_title="Equivalent Units of Inventory (EUI)",
            template="plotly_white", title_x=0.5, font=dict(family="Inter, sans-serif")
        )
        figs[room_id] = fig
        
    return figs