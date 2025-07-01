import plotly.express as px
import pandas as pd
from datetime import datetime

def create_enhanced_gantt_chart(schedule_result, color_by='resource'):
    """
    Creates an enhanced Gantt chart based on the exact TaskSchedule class.

    Args:
        schedule_result: The result object from the scheduler, containing a list of TaskSchedule objects.
        color_by (str): The dimension to color-code the Gantt chart by ('resource' or 'sku').

    Returns:
        A Plotly figure object for the Gantt chart.
    """
    if not schedule_result or not schedule_result.scheduled_tasks:
        return None

    tasks_data = []
    for task in schedule_result.scheduled_tasks:
        # Create a dictionary for the task, using the attributes from the provided dataclass
        tasks_data.append(dict(
            Task=f"{task.order_no} ({task.step_id})",
            Start=task.start_time,
            Finish=task.end_time,
            Resource=task.resource_id,
            SKU=task.sku_id,
            Tooltip=f"""
<b>Order</b>: {task.order_no}<br>
<b>SKU</b>: {task.sku_id}<br>
<b>Step</b>: {task.step_id} (Batch {task.batch_index})<br>
<b>Resource</b>: {task.resource_id}<br>
<b>Volume</b>: {getattr(task, 'volume', 'N/A')} L<br>
<b>Priority</b>: {getattr(task.priority, 'name', 'N/A')}<br>
<b>Start</b>: {task.start_time.strftime('%Y-%m-%d %H:%M')}<br>
<b>End</b>: {task.end_time.strftime('%Y-%m-%d %H:%M')}<br>
<b>Duration</b>: {task.duration_minutes} min
            """
        ))

    if not tasks_data:
        return None

    df = pd.DataFrame(tasks_data)

    # Determine the color mapping key based on user selection
    color_map_key = 'Resource' if color_by == 'resource' else 'SKU'

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Resource",
        color=color_map_key,
        hover_name="Task",
        custom_data=["Tooltip"],
        title="<b>Production Schedule Gantt Chart</b>",
        labels={"Resource": "Production Resource"}
    )

    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Resource",
        legend_title=f"<b>Color by {color_map_key}</b>",
        font=dict(family="Arial, sans-serif", size=12),
        hoverlabel=dict(bgcolor="white", font_size=14),
        title_font_size=20,
        title_x=0.5,
        xaxis=dict(
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
        )
    )
    
    # Use the detailed custom tooltip
    fig.update_traces(hovertemplate='%{customdata[0]}')

    return fig