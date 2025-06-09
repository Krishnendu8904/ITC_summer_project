import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from datetime import datetime, time, timedelta # Import timedelta for calculations
from typing import List, Dict

# Import your data models and the DataLoader
from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductTypeRegistry
)
from data_loader import DataLoader
from scheduler import ProductionScheduler # Import the updated scheduler
import config # Import the config module which holds your loaded data

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🥛 Dairy Production Scheduler",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4169E1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px; /* Adjust height as needed */
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
# 'scheduler_data' is no longer needed as config holds data.
# 'current_schedule' will hold the SchedulingResult object.
if 'current_schedule' not in st.session_state:
    st.session_state.current_schedule = None
# A flag to indicate if data has been successfully loaded into config
if 'data_loaded_successfully' not in st.session_state:
    st.session_state.data_loaded_successfully = False


# --- GANTT CHART ---
def create_gantt_chart(schedule_items: List[ScheduleItem]):
    if not schedule_items:
        fig = go.Figure()
        fig.add_annotation(text="No schedule data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Convert list of ScheduleItem dataclasses to a DataFrame for Plotly
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
            "Tank_ID": item.tank.tank_id
        }
        for item in schedule_items
    ])

    # Sort by Start_Time to ensure proper rendering order
    gantt_df = gantt_df.sort_values(by="Start_Time")
    print('Gantt DF')
    print(gantt_df)

    fig = px.timeline(gantt_df, x_start="Start_Time", x_end="End_Time", y="Line_ID", color="SKU_Code",
                      hover_data=["Quantity", "Duration_Minutes", "Setup_Minutes", "CIP_Minutes", "Tank_ID"])
    fig.update_layout(
        height=600, # Increased height for better visibility
        title="Production Schedule Gantt Chart",
        xaxis_title="Time",
        yaxis_title="Lines",
        xaxis=dict(tickformat="%H:%M"), # Display time in HH:MM format
        barmode='stack', # To stack setup/CIP/production if needed, though timeline handles it visually
        hovermode="x unified"
    )
    # Ensure lines are sorted for consistent display
    fig.update_yaxes(categoryorder="array", categoryarray=sorted(gantt_df['Line_ID'].unique().tolist()))
    return fig

# --- MAIN UI ---
def main():
    st.markdown('<h1 class="main-header">🥛 Dairy Production Scheduler</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configuration")
        source = st.radio("Data Source", ["Use Sample Data", "Upload Files"], key="data_source_radio")

        # Instantiate DataLoader
        data_loader = DataLoader()

        if source == "Upload Files":
            st.subheader("Upload CSV Files")
            # Clear previous data loaded from sample if switching
            # config.clear_config() # If you had a clear_config method in config.py
            
            uploaded_files = {}
            file_names = ['sku_config.csv', 'line_config.csv', 'tank_config.csv', 'shift_config.csv', 'user_indent.csv', 'line_sku_compatibility.csv']
            
            for file_name in file_names:
                uploaded_file = st.file_uploader(f"Upload {file_name}", type='csv', key=file_name)
                if uploaded_file:
                    uploaded_files[file_name] = uploaded_file

            # Check if all critical files are uploaded
            critical_files_uploaded = all(f in uploaded_files for f in ['sku_config.csv', 'line_config.csv', 'user_indent.csv'])

            if critical_files_uploaded:
                st.info("Attempting to load data from uploaded files...")
                try:
                    # Modify DataLoader to accept file_content directly or write to temp files
                    # For simplicity, let's assume DataLoader._get_csv_or_warn can handle file-like objects
                    # This requires modification in data_loader.py for _get_csv_or_warn
                    
                    # Option 1: Pass file_contents mapping (requires DataLoader changes)
                    # data_loader.load_all_data(file_contents=uploaded_files)

                    # Option 2: Save to temp files (more robust for existing DataLoader)
                    for filename, file_obj in uploaded_files.items():
                        temp_path = config.DATA_DIR / filename
                        with open(temp_path, "wb") as f:
                            f.write(file_obj.getbuffer())
                        
                    data_loader.load_all_data() # This will now read from temp files in DATA_DIR
                    st.success("All required files loaded successfully!")
                    st.session_state.data_loaded_successfully = True
                    if data_loader.validation_errors:
                        st.markdown('<div class="alert-warning">Some warnings during data loading:</div>', unsafe_allow_html=True)
                        for warn in data_loader.validation_errors:
                            st.warning(warn)
                except Exception as e:
                    st.session_state.data_loaded_successfully = False
                    st.error(f"Error loading files: {e}. Please check file formats.")
                    st.warning("Falling back to sample data.")
                    data_loader.load_sample_data() # Fallback
                    st.session_state.data_loaded_successfully = True # Indicate samples are loaded

            else:
                st.warning("Please upload at least 'sku_config.csv', 'line_config.csv', and 'user_indent.csv'.")
                st.info("Using sample data until all required files are uploaded or you choose 'Use Sample Data'.")
                data_loader.load_sample_data() # Load samples by default
                st.session_state.data_loaded_successfully = True # Indicate samples are loaded


        else: # "Use Sample Data" selected
            data_loader.load_sample_data()
            st.info("Sample Data Loaded.")
            st.session_state.data_loaded_successfully = True

        st.subheader("Run Settings")
        # schedule_date is less relevant for core scheduler but can be used for context if needed
        # current_time in ProductionScheduler handles its own start time
        # schedule_date = st.date_input("Schedule Date", datetime.now().date())
        
        # Only enable scheduling if data is loaded
        if st.session_state.data_loaded_successfully:
            if st.button("Run Scheduling"):
                with st.spinner('Running production scheduler... This may take a moment.'):
                    # ProductionScheduler no longer needs data in __init__ or schedule_production
                    scheduler = ProductionScheduler()
                    st.session_state.current_schedule = scheduler.schedule_production()
                st.success("Scheduling complete!")
                # st.rerun() # Rerunning here might clear some display, let Streamlit handle it naturally
        else:
            st.warning("Load data first to run the scheduler.")

    # --- RESULTS DISPLAY ---
    if st.session_state.current_schedule:
        res: SchedulingResult = st.session_state.current_schedule # Type hint for clarity

        st.markdown('<h2 class="section-header">📊 Metrics</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        #col1.metric("Total Demand", f"{res.total_demand:,} units") # Assuming this metric is added to SchedulingResult
        col2.metric("Scheduled", f"{res.total_production:,} units") # Assuming fulfillment_rate_percentage is added
        col3.metric("Unfulfilled", f"{sum(ind.qty_required for ind in res.unfulfilled_indents):,} units")
        col4.metric("Efficiency Score", f"{res.efficiency_score:.2f}%")

        if res.warnings:
            st.markdown('<h3 class="section-header">⚠️ Warnings</h3>', unsafe_allow_html=True)
            for w in res.warnings:
                st.markdown(f'<div class="alert-warning">{w}</div>', unsafe_allow_html=True)

        st.markdown('<h2 class="section-header">📅 Schedule</h2>', unsafe_allow_html=True)
        fig = create_gantt_chart(res.schedule_items)
        st.plotly_chart(fig, use_container_width=True)

        # Display schedule as DataFrame
        if res.schedule_items:
            schedule_df = pd.DataFrame([
                {
                    'SKU_ID': item.sku.sku_id,
                    'Line_ID': item.line.line_id,
                    'Tank_ID': item.tank.tank_id,
                    'Shift_ID': item.shift.shift_id,
                    'Start_Time': item.start_time.strftime('%Y-%m-%d %H:%M'),
                    'End_Time': item.end_time.strftime('%Y-%m-%d %H:%M'),
                    'Quantity_Scheduled': item.quantity,
                    'Production_Duration_Minutes': item.duration_minutes(),
                    'Setup_Time_Minutes': item.setup_time_minutes,
                    'CIP_Time_Minutes': item.cip_time_minutes
                }
                for item in res.schedule_items
            ])
            st.dataframe(schedule_df, use_container_width=True)
        else:
            st.info("No items were scheduled.")

        # Download button for schedule
        if res.schedule_items:
            # Convert schedule items to a list of dicts for JSON export
            schedule_json_export = [item.__dict__ for item in res.schedule_items]
            # Convert datetime objects in dicts to string for JSON serialization
            for item_dict in schedule_json_export:
                if isinstance(item_dict.get('start_time'), datetime):
                    item_dict['start_time'] = item_dict['start_time'].isoformat()
                if isinstance(item_dict.get('end_time'), datetime):
                    item_dict['end_time'] = item_dict['end_time'].isoformat()
            
            


        if res.unfulfilled_indents:
            st.markdown('<h2 class="section-header">❌ Unfulfilled Indents</h2>', unsafe_allow_html=True)
            unfulfilled_df = pd.DataFrame([
                {
                    'Indent_ID': indent.order_no,
                    'SKU_ID': indent.sku_id,
                    'Quantity_Required': indent.qty_required,
                    'Due_Date': indent.due_date.strftime('%Y-%m-%d') if indent.due_date else 'N/A',
                    'Priority': indent.priority.name if indent.priority else 'N/A'
                }
                for indent in res.unfulfilled_indents
            ])
            st.dataframe(unfulfilled_df, use_container_width=True)
            for u_indent in res.unfulfilled_indents:
                st.markdown(f'<div class="alert-danger">Indent {u_indent.order_no} ({u_indent.sku_id}): {u_indent.qty_required} units - Due: {u_indent.due_date.strftime("%Y-%m-%d") if u_indent.due_date else "N/A"}</div>', unsafe_allow_html=True)
        else:
            st.info("All indents were successfully scheduled!")

    else:
        st.markdown('<h2 class="section-header">🚀 Start Scheduling</h2>', unsafe_allow_html=True)
        st.info("Upload your data or use sample data to begin.")

if __name__ == '__main__':
    main()