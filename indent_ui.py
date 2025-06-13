import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

# Mock classes to handle imports that might not exist
class Priority:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    
    @classmethod
    def get_all(cls):
        return [cls.HIGH, cls.MEDIUM, cls.LOW]

class UserIndent:
    def __init__(self, order_no: str, sku_id: str, qty_required_liters: float, 
                 due_date: datetime, priority: str):
        self.order_no = order_no
        self.sku_id = sku_id
        self.qty_required_liters = qty_required_liters
        self.due_date = due_date
        self.priority = priority
    
    def _to_dict(self):
        return {
            'Order_Number': self.order_no,
            'SKU_ID': self.sku_id,
            'Qty_Required_Liters': self.qty_required_liters,
            'Priority': self.priority,
            'Due_Date': self.due_date.strftime('%Y-%m-%d')
        }

class MockConfig:
    def __init__(self):
        self.SKUS = {
            'SKU001': type('SKU', (), {'sku_id': 'SKU001', 'product_category': 'Dairy', 'inventory_size': 1000})(),
            'SKU002': type('SKU', (), {'sku_id': 'SKU002', 'product_category': 'Beverage', 'inventory_size': 2000})(),
            'SKU003': type('SKU', (), {'sku_id': 'SKU003', 'product_category': 'Juice', 'inventory_size': 1500})(),
        }
        self.USER_INDENTS = {}
        self.LINES = {}
        self.TANKS = {}
        self.SHIFTS = {}

class MockDataLoader:
    def __init__(self):
        self.data_dir = Path("./data")
        self.validation_errors = []
    
    def load_sample_data(self):
        """Mock method to simulate loading sample data"""
        pass
    
    def load_all_data(self):
        """Mock method to simulate loading all data"""
        pass

# Initialize mock objects
try:
    from models.data_models import UserIndent, Priority, SKU
    from data_loader import DataLoader
    import config
except ImportError:
    # Use mock objects if imports fail
    config = MockConfig()
    DataLoader = MockDataLoader

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
    if 'user_indents' not in st.session_state:
        st.session_state.user_indents = []
    
    if 'data_loaded_successfully' not in st.session_state:
        st.session_state.data_loaded_successfully = False
    
    if 'current_schedule' not in st.session_state:
        st.session_state.current_schedule = None
    
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "Use Sample Data"

def render_indent_ui(source: str = "Use Sample Data") -> List[UserIndent]:
    """
    Renders the indent management UI based on the selected data source.
    
    Args:
        source: The data source type ("Upload Files", "Use Sample Data", "Manual Indent Entry")
    
    Returns:
        List of UserIndent objects
    """
    
    # Initialize session state
    initialize_session_state()
    
    # Create a mock data loader if not available
    try:
        data_loader = DataLoader()
    except:
        data_loader = MockDataLoader()

    st.title("📋 Indent Management System")
    
    # Data source selection in main area
    st.subheader("🔧 Choose Data Source")
    source_options = ["Manual Indent Entry", "Use Sample Data", "Upload Files"]
    
    # Create columns for the radio buttons to make them horizontal
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
    
    # Add a visual separator
    st.markdown("---")

    # Handle different data sources
    if source == "Upload Files":
        handle_file_upload(data_loader)
    elif source == "Use Sample Data":
        handle_sample_data(data_loader)
    elif source == "Manual Indent Entry":
        handle_manual_indent_entry(data_loader)
    
    # Display current data status
    display_data_status(source)
    
    return st.session_state.user_indents

def load_indents_from_config():
    """Load indents from config.USER_INDENTS into session state for display."""
    if hasattr(config, 'USER_INDENTS') and config.USER_INDENTS:
        st.session_state.user_indents = []
        for indent_key, user_indent in config.USER_INDENTS.items():
            if isinstance(user_indent, UserIndent):
                st.session_state.user_indents.append(user_indent)
            else:
                # If it's not a UserIndent object, store the key for later lookup
                st.session_state.user_indents.append(indent_key)

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
                st.session_state.data_loaded_successfully = False
                st.error(f"❌ Error loading sample data: {str(e)}")

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
            
            priority_options = Priority.get_all() if hasattr(Priority, 'get_all') else [1, 2, 3]
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
                if (indent.order_no == indent_id and 
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
            
            st.rerun()

    # Display current manual indents
    display_manual_indents(data_loader)

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
                # Handle string or dictionary format
                try:
                    # If it's from a loaded configuration, it might be a dictionary or string key
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
        
        df = pd.DataFrame(indent_data)
        st.dataframe(df.drop('Index', axis=1), use_container_width=True)
        
        # Management buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Clear All Indents", type="secondary"):
                st.session_state.user_indents = []
                st.success("All manual indents cleared!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save to File", type="primary"):
                save_manual_indents_to_csv(data_loader, st.session_state.user_indents)
        
        with col3:
            if len(st.session_state.user_indents) > 0:
                indent_to_remove = st.selectbox(
                    "Select indent to remove:",
                    options=range(len(st.session_state.user_indents)),
                    format_func=lambda x: st.session_state.user_indents[x].order_no,
                    key="remove_indent_selectbox"
                )
        
        with col4:
            if len(st.session_state.user_indents) > 0 and st.button("❌ Remove Selected"):
                removed_indent = st.session_state.user_indents.pop(indent_to_remove)
                st.success(f"Removed indent: {removed_indent.order_no}")
                st.rerun()
        
        # Export functionality
        st.markdown("---")
        st.subheader("💾 Export Options")
        
        # Convert indents to export format, handling different types
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
            elif isinstance(indent, dict):
                due_date = indent.get('due_date', datetime.now())
                if isinstance(due_date, str):
                    due_date_str = due_date
                else:
                    due_date_str = due_date.isoformat()
                
                export_data.append({
                    'Order_Number': indent.get('order_no', 'Unknown'),
                    'SKU_ID': indent.get('sku_id', 'Unknown'),
                    'Qty_Required_Liters': indent.get('qty_required_liters', 0),
                    'Priority': indent.get('priority', 'MEDIUM'),
                    'Due_Date': due_date_str,
                })
            else:
                # String key - lookup in config
                if hasattr(config, 'USER_INDENTS') and indent in config.USER_INDENTS:
                    user_indent = config.USER_INDENTS[indent]
                    export_data.append({
                        'Order_Number': user_indent.order_no,
                        'SKU_ID': user_indent.sku_id,
                        'Qty_Required_Liters': user_indent.qty_required_liters,
                        'Priority': user_indent.priority,
                        'Due_Date': user_indent.due_date.isoformat(),
                    })
        
        export_df = pd.DataFrame(export_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"manual_indents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        
        with col2:
            # Today's indents
            today = datetime.now().date()
            todays_indents = []
            
            for indent in st.session_state.user_indents:
                indent_date = None
                indent_data = None
                
                if isinstance(indent, UserIndent):
                    indent_date = indent.due_date.date()
                    indent_data = {
                        'Order_Number': indent.order_no,
                        'SKU_ID': indent.sku_id,
                        'Qty_Required_Liters': indent.qty_required_liters,
                        'Priority': indent.priority,
                        'Due_Date': indent.due_date.isoformat(),
                    }
                elif isinstance(indent, dict):
                    due_date = indent.get('due_date', datetime.now())
                    if isinstance(due_date, str):
                        try:
                            due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                        except:
                            due_date = datetime.now()
                    indent_date = due_date.date()
                    indent_data = {
                        'Order_Number': indent.get('order_no', 'Unknown'),
                        'SKU_ID': indent.get('sku_id', 'Unknown'),
                        'Qty_Required_Liters': indent.get('qty_required_liters', 0),
                        'Priority': indent.get('priority', 'MEDIUM'),
                        'Due_Date': due_date.isoformat(),
                    }
                else:
                    # String key - lookup in config
                    if hasattr(config, 'USER_INDENTS') and indent in config.USER_INDENTS:
                        user_indent = config.USER_INDENTS[indent]
                        indent_date = user_indent.due_date.date()
                        indent_data = {
                            'Order_Number': user_indent.order_no,
                            'SKU_ID': user_indent.sku_id,
                            'Qty_Required_Liters': user_indent.qty_required_liters,
                            'Priority': user_indent.priority,
                            'Due_Date': user_indent.due_date.isoformat(),
                        }
                
                if indent_date == today and indent_data:
                    todays_indents.append(indent_data)
            
            if todays_indents:
                todays_df = pd.DataFrame(todays_indents)
                todays_csv = todays_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🗓️ Today's Indents",
                    data=todays_csv,
                    file_name=f"todays_indents_{today.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No indents due today.")

    else:
        st.info("📝 No manual indents added yet. Use the form above to add indents.")

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

def display_data_status(source: str):
    """Display the current data loading status."""
    st.markdown("---")
    st.subheader("📈 Data Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.data_loaded_successfully:
            st.success("✅ Configuration data loaded")
        else:
            st.error("❌ No configuration data loaded")
    
    with col2:
        if source == "Manual Indent Entry":
            indent_count = len(st.session_state.user_indents)
            if indent_count > 0:
                st.info(f"📝 {indent_count} manual indents ready")
            else:
                st.warning("📝 No manual indents added")
        else:
            # For loaded data sources, check both session state and config
            session_indent_count = len(st.session_state.user_indents)
            config_indent_count = len(getattr(config, 'USER_INDENTS', {}))
            total_indents = max(session_indent_count, config_indent_count)
            
            if total_indents > 0:
                st.info(f"📦 {total_indents} indents loaded from data")
            else:
                st.warning("📦 No indents in loaded data")
    
    # Quick actions
    if st.session_state.data_loaded_successfully:
        st.markdown("**Quick Actions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Data"):
                st.session_state.data_loaded_successfully = False
                st.session_state.current_schedule = None
                st.session_state.user_indents = []  # Clear user indents too
                st.info("Data reset. Please reload your data source.")
                st.rerun()

# Main function to run the app
def main():
    """Main function to run the indent UI application."""
    st.set_page_config(
        page_title="Indent Management System",
        page_icon="📋",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Run the main UI
    indents = render_indent_ui()
    
    # Display help information
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ Help")
        st.markdown("""
        **Getting Started:**
        1. Choose your data source using the buttons above
        2. For manual entry, use the form to add indents
        3. Save your indents to CSV file
        4. Export data as needed
        
        **Features:**
        - Add/Update indents manually
        - Save to CSV with duplicate handling
        - Export in multiple formats
        - Data validation
        """)

if __name__ == "__main__":
    main()