# indent_manager.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
from utils.data_models import UserIndent, Priority, SchedulingResult
from utils.new_approach import AdvancedProductionScheduler
from heuristic_scheduler import HeuristicScheduler
from utils.data_loader import DataLoader
import config

# Import the required modules - with fallback to mock classes if not available

def save_manual_indents_to_csv(data_loader, indents_to_save: List[UserIndent]):
    """
    Saves the given list of UserIndent objects to 'user_indent.csv',
    appending new indents and revising existing ones based on 'Order_Number' and 'Due_Date'.
    """
    try:
        # Ensure data directory exists
        data_dir = getattr(data_loader, 'data_dir', Path("./data"))
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / "user_indent.csv"
        
        # Load existing indents from file
        existing_df = pd.DataFrame()
        if file_path.exists():
            try:
                existing_df = pd.read_csv(file_path)
                if 'Due_Date' in existing_df.columns:
                    existing_df['Due_Date'] = existing_df['Due_Date'].astype(str)
                if 'Priority' in existing_df.columns and not existing_df['Priority'].empty:
                    # Handle priority conversion safely
                    pass
            except Exception as e:
                st.warning(f"⚠️ Could not load existing user_indent.csv: {e}. Starting with an empty base.")
                existing_df = pd.DataFrame()

        # Convert new/updated manual indents to a DataFrame
        new_indents_data = [indent._to_dict() for indent in indents_to_save]
        new_df = pd.DataFrame(new_indents_data)

        if not new_df.empty:
            # Ensure all columns from existing_df are in new_df and vice-versa
            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = np.nan
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = np.nan
            
            # Ensure Due_Date consistency
            if 'Due_Date' in new_df.columns:
                new_df['Due_Date'] = new_df['Due_Date'].astype(str)
                
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df = combined_df.drop_duplicates(subset=['Order_Number', 'Due_Date'], keep='last')
        else:
            final_df = existing_df

        # Save the final DataFrame to CSV
        final_df.to_csv(file_path, index=False)
        st.success("✅ Manual indents saved/updated in user_indent.csv!")
        
    except Exception as e:
        st.error(f"❌ Error saving manual indents to CSV: {e}")

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'user_indents_df' not in st.session_state:
        st.session_state.user_indents_df = pd.DataFrame()
    
    if 'user_indents' not in st.session_state:
        st.session_state.user_indents = []
    
    if 'data_loaded_successfully' not in st.session_state:
        st.session_state.data_loaded_successfully = False
    
    if 'current_schedule' not in st.session_state:
        st.session_state.current_schedule = None
    
    if 'last_schedule_result' not in st.session_state:
        st.session_state.last_schedule_result = None
    
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "Manual Indent Entry"

def load_indents_from_config():
    """Load indents from config.USER_INDENTS into session state for display."""
    if hasattr(config, 'USER_INDENTS') and config.USER_INDENTS:
        st.session_state.user_indents = []
        indent_dicts = []
        
        for indent_key, user_indent in config.USER_INDENTS.items():
            if isinstance(user_indent, UserIndent):
                st.session_state.user_indents.append(user_indent)
                indent_dicts.append(user_indent._to_dict())
            else:
                # If it's not a UserIndent object, store the key for later lookup
                st.session_state.user_indents.append(indent_key)
        
        # Update the DataFrame for the data editor
        if indent_dicts:
            st.session_state.user_indents_df = pd.DataFrame(indent_dicts)

def handle_file_upload(data_loader):
    """Handle file upload functionality."""
    st.header("📁 Upload Configuration Files")
    
    required_files = [
        'sku_config.csv', 'line_config.csv', 'tank_config.csv',
        'shift_config.csv', 'user_indent.csv', 'line_sku_compatibility.csv',
        'equipment_config.csv', 'room_config.csv', 'product_config.csv', 
        'cip_circuit_config.csv'
    ]
    
    uploaded_files = {}
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Core Files", "Additional Files"])
    
    with tab1:
        st.subheader("Core Configuration Files")
        for file_name in required_files[:5]:
            uploaded_file = st.file_uploader(
                f"📄 {file_name}", 
                type='csv', 
                key=file_name,
                help=f"Upload the {file_name} configuration file"
            )
            if uploaded_file:
                uploaded_files[file_name] = uploaded_file
    
    with tab2:
        st.subheader("Additional Configuration Files")
        for file_name in required_files[5:]:
            uploaded_file = st.file_uploader(
                f"📄 {file_name}", 
                type='csv', 
                key=file_name,
                help=f"Upload the {file_name} configuration file"
            )
            if uploaded_file:
                uploaded_files[file_name] = uploaded_file

    # Check critical files
    critical_files = ['sku_config.csv', 'line_config.csv', 'user_indent.csv']
    critical_files_uploaded = all(f in uploaded_files for f in critical_files)

    if critical_files_uploaded:
        if st.button("🔄 Load Uploaded Data", type="primary"):
            with st.spinner("Loading data from uploaded files..."):
                try:
                    # Create data directory if it doesn't exist
                    data_dir = getattr(data_loader, 'data_dir', Path("./data"))
                    data_dir = Path(data_dir)
                    data_dir.mkdir(exist_ok=True)
                    
                    # Save uploaded files
                    for filename, file_obj in uploaded_files.items():
                        temp_path = data_dir / filename
                        with open(temp_path, "wb") as f:
                            f.write(file_obj.getbuffer())
                    
                    # Load the data
                    data_loader.load_all_data()
                    st.session_state.data_loaded_successfully = True
                    
                    # Load indents from config into session state
                    load_indents_from_config()
                    
                    st.success("✅ All files loaded successfully!")
                    
                    # Display validation warnings
                    if hasattr(data_loader, 'validation_errors') and data_loader.validation_errors:
                        st.subheader("⚠️ Validation Warnings")
                        for warning in data_loader.validation_errors:
                            st.warning(warning)
                    
                    display_loaded_data_summary()
                    
                except Exception as e:
                    st.session_state.data_loaded_successfully = False
                    st.error(f"❌ Error loading files: {str(e)}")
                    st.info("Please check your file formats and try again.")
    else:
        missing_files = [f for f in critical_files if f not in uploaded_files]
        st.warning(f"⚠️ Please upload the following critical files: {', '.join(missing_files)}")

def handle_sample_data(data_loader):
    """Handle sample data loading."""
    st.header("🎯 Sample Data Configuration")
    
    st.info("""
    **Sample data includes:**
    - Pre-configured SKUs, production lines, and tanks
    - Sample shift schedules  
    - Example indents for testing
    - Line-SKU compatibility matrix
    - Equipment, Room, Product, and CIP Circuit configurations
    """)
    
    if st.button("📋 Load Sample Data", type="primary"):
        with st.spinner("Loading sample data..."):
            try:
                data_loader.load_sample_data()
                st.session_state.data_loaded_successfully = True
                
                # Load indents from config into session state
                load_indents_from_config()
                
                st.success("✅ Sample data loaded successfully!")
                
                display_loaded_data_summary()
                
            except Exception as e:
                print(e)
                st.session_state.data_loaded_successfully = False
                st.error(f"❌ Error loading sample data: {str(e)}")
                st.exception(e)

def handle_manual_indent_entry(data_loader):
    """Handle manual indent entry functionality."""
    st.header("✏️ Manual Indent Entry")
    
    # Ensure base data is loaded
    if not st.session_state.data_loaded_successfully:
        with st.spinner("Loading base configuration data..."):
            try:
                # Try to load sample data or use mock data
                if hasattr(data_loader, 'load_sample_data'):
                    data_loader.load_sample_data()
                st.session_state.data_loaded_successfully = True
                st.info("✅ Base configuration loaded for manual indent entry.")
            except Exception as e:
                st.warning(f"⚠️ Using mock data for demonstration: {str(e)}")
                st.session_state.data_loaded_successfully = True

    # Manual indent entry form
    with st.form("new_indent_form", clear_on_submit=True):
        st.subheader("Add New Indent")
        
        col1, col2 = st.columns(2)
        
        with col1:
            indent_id = st.text_input(
                "Indent ID", 
                value=f"MAN-{len(st.session_state.user_indents)+1:03d}",
                help="Unique identifier for this indent"
            )
            
            # Get available SKUs
            available_skus = {}
            if hasattr(config, 'SKUS') and config.SKUS:
                available_skus = config.SKUS
            else:
                # Use mock SKUs for demonstration
                available_skus = {
                    'SKU001': type('SKU', (), {'sku_id': 'SKU001', 'product_category': 'Dairy'})(),
                    'SKU002': type('SKU', (), {'sku_id': 'SKU002', 'product_category': 'Beverage'})(),
                    'SKU003': type('SKU', (), {'sku_id': 'SKU003', 'product_category': 'Juice'})(),
                }
            
            if not available_skus:
                st.warning("No SKUs available. Please load configuration data first.")
                selected_sku_id = st.selectbox("Select SKU", options=["No SKUs Available"])
            else:
                selected_sku_id = st.selectbox(
                    "Select SKU", 
                    options=list(available_skus.keys()),
                    help="Choose the product to be manufactured"
                )
            
            quantity = st.number_input(
                "Quantity (Liters)", 
                min_value=0.1, 
                value=1000.0, 
                step=100.0,
                help="Required production quantity in liters"
            )
        
        with col2:
            due_date = st.date_input(
                "Due Date", 
                value=datetime.now().date() + timedelta(days=2),
                help="When this order must be completed"
            )
            
            due_time_str = st.text_input(
                "Due Time (HH:MM)", 
                value="08:00",
                help="Time by which the order must be completed"
            )
            
            priority_options = Priority.get_all() if hasattr(Priority, 'get_all') else [Priority.HIGH, Priority.MEDIUM, Priority.LOW]
            priority = st.selectbox(
                "Priority", 
                options=priority_options,
                index=1,  # Default to MEDIUM
                help="Priority level for scheduling"
            )

        # Form submission
        submitted = st.form_submit_button("➕ Add/Update Indent", type="primary")

        if submitted:
            if not available_skus or selected_sku_id == "No SKUs Available":
                st.error("❌ Please select a valid SKU.")
                return

            # Validate time format
            try:
                due_time = datetime.strptime(due_time_str, "%H:%M").time()
            except ValueError:
                st.error("❌ Invalid time format. Please use HH:MM format (e.g., 08:00).")
                return

            # Create full datetime
            full_due_datetime = datetime.combine(due_date, due_time)
            
            # Check for existing indent
            existing_indent_found = False
            for i, indent in enumerate(st.session_state.user_indents):
                if (hasattr(indent, 'order_no') and 
                    indent.order_no == indent_id and 
                    indent.due_date.strftime("%Y-%m-%d") == full_due_datetime.strftime("%Y-%m-%d")):
                    # Update existing indent
                    st.session_state.user_indents[i] = UserIndent(
                        order_no=indent_id,
                        sku_id=selected_sku_id,
                        qty_required_liters=quantity,
                        due_date=full_due_datetime,
                        priority=priority
                    )
                    st.success(f"✅ Indent '{indent_id}' for {due_date.strftime('%Y-%m-%d')} updated successfully!")
                    existing_indent_found = True
                    break
            
            if not existing_indent_found:
                # Create new indent
                new_indent = UserIndent(
                    order_no=indent_id,
                    sku_id=selected_sku_id,
                    qty_required_liters=quantity,
                    due_date=full_due_datetime,
                    priority=priority
                )
                
                st.session_state.user_indents.append(new_indent)
                st.success(f"✅ Indent '{indent_id}' added successfully!")
            
            # Update the DataFrame for the data editor
            update_dataframe_from_indents()
            st.rerun()

def update_dataframe_from_indents():
    """Update the DataFrame from the current list of indents."""
    indent_dicts = []
    for indent in st.session_state.user_indents:
        if hasattr(indent, '_to_dict'):
            indent_dicts.append(indent._to_dict())
    
    if indent_dicts:
        st.session_state.user_indents_df = pd.DataFrame(indent_dicts)
    else:
        st.session_state.user_indents_df = pd.DataFrame()

def display_manual_indents(data_loader):
    """Display and manage current manual indents."""
    st.markdown("---")
    st.subheader("📋 Current Manual Indents")
    
    if st.session_state.user_indents:
        # Create display dataframe
        indent_data = []
        for i, indent in enumerate(st.session_state.user_indents):
            # Handle both UserIndent objects and dictionary entries
            if isinstance(indent, UserIndent):
                # Direct UserIndent object
                indent_data.append({
                    'Index': i,
                    'Indent ID': indent.order_no,
                    'SKU ID': indent.sku_id,
                    'Quantity (Liters)': f"{indent.qty_required_liters:,.0f}",
                    'Due Date': indent.due_date.strftime('%Y-%m-%d'),
                    'Due Time': indent.due_date.strftime('%H:%M'),
                    'Priority': indent.priority
                })
            else:
                # Handle string or dictionary format - similar to indent_ui.py logic
                try:
                    if isinstance(indent, dict):
                        due_date = indent.get('due_date', datetime.now())
                        if isinstance(due_date, str):
                            due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                        
                        indent_data.append({
                            'Index': i,
                            'Indent ID': indent.get('order_no', f'IND-{i}'),
                            'SKU ID': indent.get('sku_id', 'Unknown'),
                            'Quantity (Liters)': f"{indent.get('qty_required_liters', 0):,.0f}",
                            'Due Date': due_date.strftime('%Y-%m-%d'),
                            'Due Time': due_date.strftime('%H:%M'),
                            'Priority': indent.get('priority', 'MEDIUM')
                        })
                    else:
                        # If it's a string key, try to get from config
                        if hasattr(config, 'USER_INDENTS') and indent in config.USER_INDENTS:
                            user_indent = config.USER_INDENTS[indent]
                            indent_data.append({
                                'Index': i,
                                'Indent ID': user_indent.order_no,
                                'SKU ID': user_indent.sku_id,
                                'Quantity (Liters)': f"{user_indent.qty_required_liters:,.0f}",
                                'Due Date': user_indent.due_date.strftime('%Y-%m-%d'),
                                'Due Time': user_indent.due_date.strftime('%H:%M'),
                                'Priority': user_indent.priority
                            })
                        else:
                            # Fallback for unknown format
                            indent_data.append({
                                'Index': i,
                                'Indent ID': str(indent),
                                'SKU ID': 'Unknown',
                                'Quantity (Liters)': '0',
                                'Due Date': datetime.now().strftime('%Y-%m-%d'),
                                'Due Time': datetime.now().strftime('%H:%M'),
                                'Priority': 'MEDIUM'
                            })
                except Exception as e:
                    st.warning(f"⚠️ Error processing indent {i}: {e}")
                    continue
        
        if indent_data:
            df = pd.DataFrame(indent_data)
            st.dataframe(df.drop('Index', axis=1), use_container_width=True)
        
        # Export functionality
        st.markdown("---")
        st.subheader("💾 Export Options")
        
        # Convert indents to export format
        export_data = []
        for indent in st.session_state.user_indents:
            if isinstance(indent, UserIndent):
                export_data.append({
                    'Order_Number': indent.order_no,
                    'SKU_ID': indent.sku_id,
                    'Qty_Required_Liters': indent.qty_required_liters,
                    'Priority': indent.priority,
                    'Due_Date': indent.due_date.isoformat(),
                })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"manual_indents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("💾 Save to File", type="primary"):
                    save_manual_indents_to_csv(data_loader, st.session_state.user_indents)
            
            with col3:
                if st.button("🗑️ Clear All Indents", type="secondary"):
                    st.session_state.user_indents = []
                    st.session_state.user_indents_df = pd.DataFrame()
                    st.success("All manual indents cleared!")
                    st.rerun()

def display_loaded_data_summary():
    """Display a summary of the currently loaded data."""
    st.subheader("📊 Loaded Data Summary")
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sku_count = len(getattr(config, 'SKUS', {}))
            st.metric("SKUs", sku_count)
        
        with col2:
            line_count = len(getattr(config, 'LINES', {}))
            st.metric("Production Lines", line_count)
        
        with col3:
            tank_count = len(getattr(config, 'TANKS', {}))
            st.metric("Storage Tanks", tank_count)
        
        with col4:
            indent_count = len(getattr(config, 'USER_INDENTS', {}))
            st.metric("Loaded Indents", indent_count)

        # Show sample data if available
        if hasattr(config, 'SKUS') and config.SKUS:
            with st.expander("📋 View Loaded SKUs"):
                sku_data = []
                for sku in config.SKUS.values():
                    sku_data.append({
                        'SKU_ID': getattr(sku, 'sku_id', 'N/A'),
                        'Product_Category': getattr(sku, 'product_category', 'N/A'),
                        'Inventory_Size': getattr(sku, 'inventory_size', 'N/A'),
                    })
                if sku_data:
                    st.dataframe(pd.DataFrame(sku_data), use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not display data summary: {str(e)}")

def render_user_indent_manager():
    """
    Enhanced indent management UI that combines data source selection,
    manual entry, and scheduler triggering functionality.
    """
    # Initialize session state
    initialize_session_state()
    
    # Create a data loader instance
    data_loader = DataLoader()

    st.title("📋 Advanced Indent Management & Scheduling System")
    
    # === DATA SOURCE SELECTION ===
    st.subheader("🔧 Choose Data Source")
    
    # Create columns for the data source buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        manual_selected = st.button(
            "✏️ Manual Indent Entry", 
            use_container_width=True,
            type="primary" if st.session_state.data_source == "Manual Indent Entry" else "secondary"
        )
    
    with col2:
        sample_selected = st.button(
            "🎯 Use Sample Data", 
            use_container_width=True,
            type="primary" if st.session_state.data_source == "Use Sample Data" else "secondary"
        )
    
    with col3:
        upload_selected = st.button(
            "📁 Upload Files", 
            use_container_width=True,
            type="primary" if st.session_state.data_source == "Upload Files" else "secondary"
        )
    
    # Handle button clicks
    if manual_selected and st.session_state.data_source != "Manual Indent Entry":
        st.session_state.data_source = "Manual Indent Entry"
        st.rerun()
    elif sample_selected and st.session_state.data_source != "Use Sample Data":
        st.session_state.data_source = "Use Sample Data"
        st.rerun()
    elif upload_selected and st.session_state.data_source != "Upload Files":
        st.session_state.data_source = "Upload Files"
        st.rerun()
    
    source = st.session_state.data_source
    
    st.markdown("---")

    # === HANDLE DIFFERENT DATA SOURCES ===
    if source == "Upload Files":
        handle_file_upload(data_loader)
    elif source == "Use Sample Data":
        handle_sample_data(data_loader)
    elif source == "Manual Indent Entry":
        handle_manual_indent_entry(data_loader)
        display_manual_indents(data_loader)

    # === INDENT TABLE EDITOR ===
    st.markdown("---")
    st.subheader("📝 Current Production Indents")
    
    # Ensure DataFrame is initialized
    if st.session_state.user_indents_df.empty and st.session_state.user_indents:
        update_dataframe_from_indents()
    
    # If we have indents from config but not in DataFrame, load them
    if (st.session_state.user_indents_df.empty and 
        hasattr(config, 'USER_INDENTS') and config.USER_INDENTS):
        load_indents_from_config()

    st.info("You can add, edit, or remove indents below. The scheduler will use this table as its input.")
    # Ensure the 'Due_Date' column is in a compatible format before passing to the editor
    df_for_editing = st.session_state.user_indents_df.copy()
    df_for_editing["Due_Date"] = pd.to_datetime(df_for_editing["Due_Date"])
    st.session_state.user_indents_df = df_for_editing

    if not st.session_state.user_indents_df.empty:
        edited_indents_df = st.data_editor(
            st.session_state.user_indents_df,
            num_rows="dynamic",
            key="indents_editor",
            use_container_width=True,
            column_config={
                "Order_Number": st.column_config.TextColumn("Order Number", required=True),
                "SKU_ID": st.column_config.SelectboxColumn(
                    "SKU ID", 
                    options=list(config.SKUS.keys()) if hasattr(config, 'SKUS') and config.SKUS else ['SKU001', 'SKU002', 'SKU003'], 
                    required=True
                ),
                "Qty_Required_Liters": st.column_config.NumberColumn("Required Liters", min_value=1.0, required=True),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority", 
                    options=[p.value for p in Priority], 
                    required=True
                ),
                "Due_Date": st.column_config.DateColumn("Due Date", required=True, format="YYYY-MM-DD")
            }
        )
        
        # Update session state with edited data
        st.session_state.user_indents_df = edited_indents_df
    else:
        st.info("📝 No indents available. Please add indents using the form above or load data from files.")
        edited_indents_df = pd.DataFrame()

    # === SCHEDULER SECTION ===
    st.markdown("---")
    st.markdown("### 🚀 Production Scheduler")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        time_limit_seconds = st.number_input(
            "Solver Time Limit (seconds)", 
            min_value=10, 
            max_value=600, 
            value=60, 
            step=10,
            help="Maximum time for the optimization solver to run"
        )
    
    with col2:
        st.write("##")  # for vertical alignment
        if st.button("🚀 Generate Optimized Schedule", type="primary", use_container_width=True):
            # Validate that we have indents to schedule
            if edited_indents_df.empty:
                st.error("❌ No indents to schedule. Please add some indents first.")
                return
            
            # Update config with edited indents before running scheduler
            try:
                updated_indents = {}
                for _, row in edited_indents_df.iterrows():
                    order_no = str(row["Order_Number"])
                    if not order_no:
                        st.error("❌ Order Number cannot be empty. Please fix the table.")
                        return

                    # Handle due date conversion
                    due_date = row["Due_Date"]
                    if isinstance(due_date, str):
                        due_date = datetime.strptime(due_date, "%Y-%m-%d").date()
                    
                    # Combine date with default time (end of workday)
                    full_due_datetime = datetime.combine(due_date, datetime.min.time()).replace(hour=18)

                    updated_indents[order_no] = UserIndent(
                        order_no=order_no,
                        sku_id=str(row["SKU_ID"]),
                        qty_required_liters=float(row["Qty_Required_Liters"]),
                        priority=Priority(int(row["Priority"])),
                        due_date=full_due_datetime
                    )
                
                # Update the main config object so scheduler can see it
                if hasattr(config, 'USER_INDENTS'):
                    config.USER_INDENTS = updated_indents
                
                # Save edits to session state
                st.session_state.user_indents_df = edited_indents_df
                
                # Update the user_indents list as well
                st.session_state.user_indents = list(updated_indents.values())

                with st.spinner(f"Running Scheduler for {time_limit_seconds} seconds... This may take a while."):
                    # Calculate schedule start time (next working day)
                    schedule_start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
                    if schedule_start <= datetime.now():
                        schedule_start += timedelta(days=1)
                    
                    scheduler = HeuristicScheduler(
                        indents=config.USER_INDENTS,
                        skus=config.SKUS,
                        products=config.PRODUCTS,
                        lines=config.LINES,
                        tanks=config.TANKS,
                        equipments=config.EQUIPMENTS,
                        shifts=config.SHIFTS
                    )

                    result = scheduler.run_heuristic_scheduler()
                    
                    # Save result to session state
                    st.session_state.last_schedule_result = result
                    st.session_state.current_schedule = result
                
                if result and result.is_feasible:
                    st.success("✅ Optimized schedule generated successfully!")
                    display_scheduling_results(result)
                else:
                    st.error("❌ The solver could not find a feasible schedule with the given constraints and time limit.")
                    st.warning("💡 Consider increasing the time limit or checking for resource conflicts.")
                
            except Exception as e:
                st.error(f"❌ An error occurred during scheduling: {str(e)}")
                st.exception(e)
                st.info("Please check your data and try again.")

    # === RESULTS SUMMARY ===
    display_previous_results()

def display_scheduling_results(result: SchedulingResult):
    """
    Display detailed scheduling results. This version filters the summary
    to show only final orders, not internal bulk production jobs.
    """
    if not result or not result.is_feasible:
        return
    
    st.markdown("---")
    st.subheader("📈 Scheduling Results")

    # --- THIS IS THE KEY FIX ---
    # Get the set of actual order numbers from the original indents.
    # This allows us to filter the summary and show only what matters to the end-user.
    actual_order_nos = {indent.order_no for indent in st.session_state.user_indents}

    # Filter the production summary to only include actual orders.
    filtered_summary = {
        order_no: summary for order_no, summary in result.production_summary.items()
        if order_no in actual_order_nos
    }
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total_orders = len(actual_order_nos)
    scheduled_orders = len(filtered_summary)
    
    with col1:
        st.metric("Total Orders", total_orders)
    with col2:
        st.metric("Scheduled Orders", scheduled_orders)
    with col3:
        success_rate = (scheduled_orders / total_orders * 100) if total_orders > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Total Tasks Created", len(result.scheduled_tasks))

    # Production summary table for FINAL ORDERS ONLY
    st.markdown("#### 📊 Final Order Production Summary")
    
    summary_data = []
    # Iterate through the original indents to maintain a clean order
    for indent in st.session_state.user_indents:
        order_no = indent.order_no
        summary = filtered_summary.get(order_no) # Get the summary if this order was scheduled

        if summary:
            # Find the completion time for this specific order
            order_tasks = [t for t in result.scheduled_tasks if t.order_no == order_no]
            completion_time = max(t.end_time for t in order_tasks) if order_tasks else "N/A"
            
            summary_data.append({
                "Order No.": order_no,
                "SKU": indent.sku_id,
                "Status": "✅ Scheduled",
                "Qty Required (L)": f"{indent.qty_required_liters:,.0f}",
                "Qty Produced (L)": f"{summary.get('produced_quantity', 0):,.0f}",
                "Underproduction (L)": f"{summary.get('underproduction', 0):,.0f}",
                "Due Date": indent.due_date.strftime("%Y-%m-%d %H:%M"),
                "Est. Completion": completion_time.strftime("%Y-%m-%d %H:%M") if isinstance(completion_time, datetime) else str(completion_time)
            })
        else:
            # If the order is not in the filtered summary, it was not scheduled.
            summary_data.append({
                "Order No.": order_no,
                "SKU": indent.sku_id,
                "Status": "❌ Not Scheduled",
                "Qty Required (L)": f"{indent.qty_required_liters:,.0f}",
                "Qty Produced (L)": "0", "Underproduction (L)": f"{indent.qty_required_liters:,.0f}",
                "Due Date": indent.due_date.strftime("%Y-%m-%d %H:%M"),
                "Est. Completion": "N/A"
            })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Detailed task schedule (shows ALL tasks, including bulk)
    if result.scheduled_tasks:
        with st.expander("📋 View Full Detailed Task Schedule (includes all steps)"):
            task_data = []
            for task in result.scheduled_tasks:
                task_data.append({
                    "Job/Order ID": task.order_no,
                    "SKU/Component": task.sku_id,
                    "Step": task.step_id,
                    "Batch": task.batch_index,
                    "Resource": task.resource_id,
                    "Start Time": task.start_time.strftime('%Y-%m-%d %H:%M'),
                    "End Time": task.end_time.strftime('%Y-%m-%d %H:%M'),
                    "Volume (L)": f"{task.volume:,.0f}",
                })
            
            if task_data:
                df_tasks = pd.DataFrame(task_data)
                st.dataframe(df_tasks.sort_values(by="Start Time").reset_index(drop=True), use_container_width=True)

def display_previous_results():
    """Display previously generated scheduling results if available."""
    st.markdown("---")
    st.subheader("📈 Previous Scheduling Results")
    
    if st.session_state.get('last_schedule_result'):
        result = st.session_state.last_schedule_result
        if result.is_feasible:
            display_scheduling_results(result)
        else:
            st.warning("⚠️ No feasible schedule was found in the last run.")
            st.info("Try adjusting your indents or increasing the solver time limit.")
    else:
        st.info("🔄 Run the scheduler to see results here.")
