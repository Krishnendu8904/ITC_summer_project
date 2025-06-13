import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List

from models.data_models import ScheduleItem, SchedulingResult
from scheduler import ProductionScheduler
import config

def create_gantt_chart(schedule_items: List[ScheduleItem]):
    """Create a Gantt chart visualization for the production schedule."""
    if not schedule_items:
        fig = go.Figure()
        fig.add_annotation(
            text="No schedule data available", 
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Production Schedule Gantt Chart", 
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Prepare data for Gantt chart
    gantt_df = pd.DataFrame([
        {
            "SKU_Code": item.sku.sku_id,
            "Line_ID": item.line.line_id,
            "Quantity": item.quantity,
            "Start_Time": item.start_time,
            "End_Time": item.end_time,
            "Duration_Minutes": item.duration_minutes(),
            "Setup_Minutes": item.setup_time_minutes,
            "CIP_Minutes": item.cip_time_minutes,
            "Tank_ID": item.tank.tank_id if item.tank else "N/A",
            "Display_Text": f"{item.sku.sku_id} ({item.quantity}L)"
        }
        for item in schedule_items
    ])

    gantt_df = gantt_df.sort_values(by="Start_Time")

    # Create timeline chart
    fig = px.timeline(
        gantt_df,
        x_start="Start_Time",
        x_end="End_Time",
        y="Line_ID",
        color="SKU_Code",
        text="Display_Text",
        hover_data=["Quantity", "Duration_Minutes", "Setup_Minutes", "CIP_Minutes", "Tank_ID"],
        title="Production Schedule Gantt Chart"
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Time",
        yaxis_title="Production Lines",
        xaxis=dict(tickformat="%Y-%m-%d %H:%M"),
        hovermode="closest",
        showlegend=True
    )
    
    # Sort y-axis categories
    fig.update_yaxes(
        categoryorder="array", 
        categoryarray=sorted(gantt_df['Line_ID'].unique().tolist())
    )
    
    # Update text position
    fig.update_traces()
    
    return fig

def get_indent_data(source: str):
    """Get indent data based on the selected source."""
    if source == "Manual Indent Entry":
        return st.session_state.user_indents
    else:
        try:
            return list(config.USER_INDENTS.values()) if hasattr(config, 'USER_INDENTS') and config.USER_INDENTS else []
        except:
            return []

def run_scheduler_ui(source: str):
    """UI for running the production scheduler."""
    
    # Get available indents
    indents_to_schedule = get_indent_data(source)
    
    # Display current status
    if not indents_to_schedule:
        st.warning(f"⚠️ No indents available to schedule from '{source}' source.")
        if source == "Manual Indent Entry":
            st.info("💡 Go to the 'Data & Indents' tab to add manual indents.")
        else:
            st.info("💡 Please load data first in the 'Data & Indents' tab.")
        return
    
    # Show scheduling options
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📋 Ready to schedule {len(indents_to_schedule)} indents")
    with col2:
        total_volume = sum(indent.qty_required for indent in indents_to_schedule)
        st.info(f"📊 Total volume: {total_volume:,.0f} L")
    
    # Run scheduler button
    if st.button("🚀 Run Production Scheduler", type="primary", use_container_width=True):
        with st.spinner('🔄 Running production scheduler...'):
            try:
                # Update config with current indents if using manual entry
                if source == "Manual Indent Entry":
                    config.USER_INDENTS = {
                        indent.order_no: indent 
                        for indent in st.session_state.user_indents
                    }
                
                scheduler = ProductionScheduler()
                st.session_state.current_schedule = scheduler.schedule_production()
                
                st.success("✅ Scheduling completed successfully!")
                
                # Show quick summary
                result = st.session_state.current_schedule
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Scheduled Items", len(result.schedule_items))
                with col2:
                    st.metric("📦 Total Production", f"{result.total_production:,.0f} L")
                with col3:
                    unfulfilled = len(result.unfulfilled_indents)
                    st.metric("❌ Unfulfilled", unfulfilled, delta=-unfulfilled if unfulfilled > 0 else None)
                
            except Exception as e:
                st.error(f"❌ Scheduling failed: {str(e)}")
                st.session_state.current_schedule = None

def display_schedule_results(source: str):
    """Display the results of the production scheduling."""
    
    if not st.session_state.get("current_schedule"):
        st.info("📋 No schedule has been generated yet. Go to the 'Schedule' tab to run the scheduler.")
        return

    result: SchedulingResult = st.session_state.current_schedule

    # Key Metrics Section
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_indents = len(get_indent_data(source))
        st.metric("Total Indents", total_indents)
    
    with col2:
        st.metric("Scheduled Items", len(result.schedule_items))
    
    with col3:
        st.metric("Production Volume", f"{result.total_production:,} L")
    
    with col4:
        unfulfilled_volume = sum(indent.qty_required for indent in result.unfulfilled_indents)
        st.metric("Unfulfilled Volume", f"{unfulfilled_volume:,} L")

    # Warnings Section
    if result.warnings:
        st.subheader("⚠️ Scheduling Warnings")
        for warning in result.warnings:
            st.warning(warning)

    # Schedule Visualization
    st.subheader("📅 Production Schedule")
    
    if result.schedule_items:
        # Gantt Chart
        fig = create_gantt_chart(result.schedule_items)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Schedule Table
        st.subheader("📋 Detailed Schedule")
        schedule_df = pd.DataFrame([
            {
                'SKU_ID': item.sku.sku_id,
                'Line_ID': item.line.line_id,
                'Tank_ID': item.tank.tank_id if item.tank else 'N/A',
                'Start_Time': item.start_time.strftime('%Y-%m-%d %H:%M'),
                'End_Time': item.end_time.strftime('%Y-%m-%d %H:%M'),
                'Quantity_L': f"{item.quantity:,.0f}",
                'Production_Min': item.duration_minutes(),
                'Setup_Min': item.setup_time_minutes,
                'CIP_Min': item.cip_time_minutes,
                'Total_Min': item.duration_minutes() + item.setup_time_minutes + item.cip_time_minutes
            } for item in result.schedule_items
        ])
        
        st.dataframe(schedule_df, use_container_width=True)

        # Download Options
        st.subheader("💾 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = schedule_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"production_schedule_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = schedule_df.to_json(orient="records", date_format="iso").encode('utf-8')
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"production_schedule_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.warning("📋 No items were scheduled. Please check your indents and constraints.")

    # Unfulfilled Indents Section
    if result.unfulfilled_indents:
        st.subheader("❌ Unfulfilled Indents")
        st.error(f"⚠️ {len(result.unfulfilled_indents)} indents could not be scheduled.")
        
        unfulfilled_df = pd.DataFrame([
            {
                'Indent_ID': indent.order_no,
                'SKU_ID': indent.sku_id,
                'Quantity_Required_L': f"{indent.qty_required:,.0f}",
                'Due_Date': indent.due_date.strftime('%Y-%m-%d %H:%M'),
                'Priority': indent.priority.name
            }
            for indent in result.unfulfilled_indents
        ])
        
        st.dataframe(unfulfilled_df, use_container_width=True)
        
        # Suggest actions
        st.info("""
        **Possible reasons for unfulfilled indents:**
        - Insufficient production capacity
        - SKU-Line compatibility issues
        - Tank availability constraints
        - Due date conflicts
        
        **Suggested actions:**
        - Extend the scheduling time horizon
        - Review SKU-Line compatibility settings
        - Check tank capacity constraints
        - Consider adjusting priorities
        """)
    else:
        st.success("🎉 All indents were successfully scheduled!")

    # Summary Statistics
    if result.schedule_items:
        st.subheader("📈 Summary Statistics")
        
        # Line utilization
        line_usage = {}
        for item in result.schedule_items:
            line_id = item.line.line_id
            total_time = item.duration_minutes() + item.setup_time_minutes + item.cip_time_minutes
            line_usage[line_id] = line_usage.get(line_id, 0) + total_time
        
        if line_usage:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Line Utilization (Minutes)**")
                line_df = pd.DataFrame([
                    {"Line": line, "Total_Minutes": minutes, "Hours": f"{minutes/60:.1f}"}
                    for line, minutes in sorted(line_usage.items())
                ])
                st.dataframe(line_df, use_container_width=True)
            
            with col2:
                st.write("**SKU Distribution**")
                sku_counts = {}
                for item in result.schedule_items:
                    sku_counts[item.sku.sku_id] = sku_counts.get(item.sku.sku_id, 0) + 1
                
                sku_df = pd.DataFrame([
                    {"SKU": sku, "Batches": count}
                    for sku, count in sorted(sku_counts.items())
                ])
                st.dataframe(sku_df, use_container_width=True)