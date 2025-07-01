import streamlit as st
import pandas as pd
from collections import defaultdict
from datetime import datetime

# --- Local Imports ---
from managers.gantt_chart import create_production_gantt
import config

def get_final_order_completion(result) -> pd.DataFrame:
    """
    Parses the scheduling result to find the final completion time for each customer order.
    """
    if not result or not result.scheduled_tasks:
        return pd.DataFrame()

    # Get the original customer orders from the indents config
    original_order_nos = set(config.USER_INDENTS.keys())
    
    # Find the final end time for each order
    completion_times = defaultdict(lambda: None)
    for task in result.scheduled_tasks:
        # The task's order_no from the scheduler might be a bulk job or a final order
        # We only care about tasks associated with an original customer order
        if task.order_no in original_order_nos:
            if completion_times[task.order_no] is None or task.end_time > completion_times[task.order_no]:
                completion_times[task.order_no] = task.end_time
    
    # Build the summary DataFrame
    dispatch_data = []
    for order_no, end_time in completion_times.items():
        indent = config.USER_INDENTS.get(order_no)
        if indent and end_time:
            dispatch_data.append({
                "Order Number": indent.order_no,
                "SKU ID": indent.sku_id,
                "Quantity (Liters)": indent.qty_required_liters,
                "Estimated Completion": end_time.strftime("%Y-%m-%d %H:%M"),
                "Original Due Date": indent.due_date.strftime("%Y-%m-%d %H:%M"),
            })

    if not dispatch_data:
        return pd.DataFrame()

    # Sort by completion time
    summary_df = pd.DataFrame(dispatch_data)
    summary_df['Estimated Completion'] = pd.to_datetime(summary_df['Estimated Completion'])
    return summary_df.sort_values(by="Estimated Completion").reset_index(drop=True)


def render():
    """
    Renders the UI for the Logistics Manager role.
    """
    st.markdown('<h2 class="section-header">Logistics & Dispatch Dashboard</h2>', unsafe_allow_html=True)

    if 'last_schedule_result' not in st.session_state or not st.session_state.last_schedule_result:
        st.markdown('<div class="info-box">⚠️ No production schedule has been generated yet. Please ask the Production Manager to run the scheduler to see dispatch information.</div>', unsafe_allow_html=True)
        return

    result = st.session_state.last_schedule_result
    
    # --- Main Layout: Two Columns ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📋 Upcoming Dispatch Schedule</h3>', unsafe_allow_html=True)
        dispatch_df = get_final_order_completion(result)

        if not dispatch_df.empty:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            # Format the datetime column for display
            display_df = dispatch_df.copy()
            display_df['Estimated Completion'] = display_df['Estimated Completion'].dt.strftime('%A, %b %d @ %I:%M %p')
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("The current schedule does not contain any completed customer orders.")

    with col2:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📦 Today\'s Dispatches</h3>', unsafe_allow_html=True)
        if not dispatch_df.empty:
            today = pd.to_datetime(datetime.now().date())
            todays_dispatches = dispatch_df[dispatch_df['Estimated Completion'].dt.date == today]
            
            st.metric("Orders to Dispatch Today", len(todays_dispatches))

            if not todays_dispatches.empty:
                 # Displaying as a list for a cleaner look in the smaller column
                for _, row in todays_dispatches.iterrows():
                    st.markdown(f"""
                    <div class="info-box" style="margin-bottom: 0.5rem; border-left-color: #28a745;">
                        <strong>{row['Order Number']}</strong> ({row['SKU ID']})<br>
                        Ready by: <strong>{pd.to_datetime(row['Estimated Completion']).strftime('%I:%M %p')}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No dispatches scheduled for today.")
        else:
            st.info("No dispatch data available.")

    st.markdown("---")
    st.markdown('<h3 class="section-header" style="font-size:1.4rem;">📊 Production Gantt Chart (Order View)</h3>', unsafe_allow_html=True)
    prod_gantt = create_production_gantt(result)
    if prod_gantt:
        st.plotly_chart(prod_gantt, use_container_width=True)
    else:
        st.warning("Could not generate the Production Gantt chart.")

