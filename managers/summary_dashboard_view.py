from collections import defaultdict
from datetime import datetime

def create_summary_dashboard_data(schedule_result, initial_indents):
    """
    Calculates high-level summary metrics from the schedule result.
    
    Args:
        schedule_result: The result object from the scheduler.
        initial_indents: The dictionary of original UserIndent objects.
        
    Returns:
        A dictionary containing the calculated summary metrics.
    """
    if not schedule_result or not schedule_result.scheduled_tasks:
        return None

    # --- Initialize counters ---
    total_prod_mins = 0
    total_cip_mins = 0
    total_volume_produced = 0
    
    # --- Group tasks by order to analyze each order's performance ---
    tasks_by_order = defaultdict(list)
    for task in schedule_result.scheduled_tasks:
        tasks_by_order[task.order_no].append(task)
        total_volume_produced += getattr(task, 'volume', 0)
        
        # Aggregate total times
        if getattr(task, 'CIP_required', False):
            total_cip_mins += task.duration_minutes
        else:
            total_prod_mins += task.duration_minutes

    # --- Calculate OTIF ---
    on_time_count = 0
    in_full_count = 0
    otif_count = 0
    total_orders = len(initial_indents)

    for order_no, original_indent in initial_indents.items():
        scheduled_tasks = tasks_by_order.get(order_no)
        
        if not scheduled_tasks:
            continue # This order was not scheduled

        # Check "In-Full"
        produced_volume = sum(getattr(task, 'volume', 0) for task in scheduled_tasks)
        is_in_full = produced_volume >= original_indent.qty_required_liters
        if is_in_full:
            in_full_count += 1
            
        # Check "On-Time"
        final_end_time = max(task.end_time for task in scheduled_tasks)
        is_on_time = final_end_time <= original_indent.due_date
        if is_on_time:
            on_time_count += 1
            
        # Check OTIF
        if is_on_time and is_in_full:
            otif_count += 1

    return {
        "total_prod_hours": total_prod_mins / 60,
        "total_cip_hours": total_cip_mins / 60,
        "total_volume_produced": total_volume_produced,
        "total_orders": total_orders,
        "otif_percent": (otif_count / total_orders) * 100 if total_orders > 0 else 0,
        "on_time_percent": (on_time_count / total_orders) * 100 if total_orders > 0 else 0,
        "in_full_percent": (in_full_count / total_orders) * 100 if total_orders > 0 else 0,
    }


def render_summary_dashboard(data: dict) -> str:
    """Renders the summary data into a clean HTML block."""
    if not data:
        return ""
        
    # --- CSS for styling the cards ---
    styles = """
    <style>
        .metric-container { display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin-bottom: 20px; }
        .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; flex-grow: 1; border-left: 5px solid #1E90FF; }
        .metric-card-otif { border-left: 5px solid #32CD32; }
        .metric-title { font-size: 0.9em; color: #4F4F4F; margin-bottom: 5px; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #000000; }
    </style>
    """
    
    # --- HTML content ---
    html = f"""
    <div class="metric-container">
        <div class="metric-card metric-card-otif">
            <div class="metric-title">OTIF Rate</div>
            <div class="metric-value">{data['otif_percent']:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Total Production Time</div>
            <div class="metric-value">{data['total_prod_hours']:.1f} hrs</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Total CIP Time</div>
            <div class="metric-value">{data['total_cip_hours']:.1f} hrs</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Total Volume Produced</div>
            <div class="metric-value">{data['total_volume_produced']:,.0f} L</div>
        </div>
    </div>
    """
    
    return styles + html