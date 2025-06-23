import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
import logging # Import the logging library

# Updated imports for the new structure
import config
from utils.data_loader import DataLoader
from managers.sku_manager import render_sku_manager
from managers.line_manager import render_line_manager
from managers.tank_manager import render_tank_manager
from managers.shift_manager import render_shift_manager
from managers.indent_manager import render_user_indent_manager
from managers.equipment_manager import render_equipment_manager
from managers.room_manager import render_room_manager
from managers.product_manager import render_product_manager
from managers.cip_manager import render_cip_circuit_manager
from managers.gantt_chart import create_production_gantt, create_resource_gantt

# --- Page Configuration ---
st.set_page_config(
    page_title="Dairy Production Plant Management",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEW: Logging Setup ---
def setup_logging():
    """Configures the logging to write to a file in the logs directory."""
    log_file_path = config.LOGS_DIR / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='a'), # 'a' for append
            logging.StreamHandler() # To also see logs in the console
        ]
    )
    logging.info("Application starting, logging configured.")


# --- App State Management ---
def initialize_app_state():
    """Initializes the DataLoader and loads all data into the app's state."""
    if 'app_initialized' not in st.session_state:
        # The config module already creates the logs directory
        
        # Instantiate, store, and run the DataLoader
        data_loader = DataLoader(data_dir=config.DATA_DIR)
        data_loader.load_all_data()
        st.info("All data loaded")
        st.session_state.data_loader = data_loader
        
        # Mark as initialized and set default for results
        st.session_state.app_initialized = True
        st.session_state.last_schedule_result = None
        logging.info("Application initialized. All configuration data has been loaded.")


# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    # Setup logging right at the start
    setup_logging()

    st.markdown('<h1 style="text-align:center; color:#00441B; font-size:2.5rem; font-weight:bold;">🏭 Production Management</h1>', unsafe_allow_html=True)
    
    initialize_app_state()

    # --- Main Tab Layout ---
    tab1, tab2, tab3 = st.tabs(["**⚙️ Factory Setup**", "**📅 Scheduler**", "**📊 Analytics**"])

    # (The rest of the main function remains exactly the same)
    # ...
    with tab1:
        st.header("Factory Setup & Configuration")
        st.markdown("Visually manage all factory elements. **Remember to click the save buttons within each tab to persist your changes to the source CSV files.**")
        
        (sku_tab, line_tab, tank_tab, shift_tab, equip_tab, room_tab, product_tab, cip_tab) = st.tabs(
            ["SKUs", "Lines", "Tanks", "Shifts", "Equipment", "Rooms", "Products", "CIP Circuits"]
        )
        
        data_loader = st.session_state.data_loader
        
        with sku_tab: render_sku_manager(data_loader)
        with line_tab: render_line_manager(data_loader)
        with tank_tab: render_tank_manager(data_loader)
        with shift_tab: render_shift_manager(data_loader)
        with equip_tab: render_equipment_manager(data_loader)
        with room_tab: render_room_manager(data_loader)
        with product_tab: render_product_manager(data_loader)
        with cip_tab: render_cip_circuit_manager(data_loader)

    with tab2:
        st.header("Indent Accepting and Production Scheduler")
        render_user_indent_manager()

    with tab3:
        st.header("Results & Analytics")
        st.info("This dashboard displays the Gantt charts for the most recently generated schedule.")
        
        if st.session_state.last_schedule_result and st.session_state.last_schedule_result.is_feasible:
            result = st.session_state.last_schedule_result
            
            gantt_tab1, gantt_tab2 = st.tabs(["**Production Gantt (by Order)**", "**Resource Gantt (by Machine/Line)**"])
            
            with gantt_tab1:
                prod_gantt = create_production_gantt(result)
                if prod_gantt: st.plotly_chart(prod_gantt, use_container_width=True)
                
            with gantt_tab2:
                schedule_start = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=1)
                horizon = 4 # This should match the horizon used in the scheduler
                resource_gantt = create_resource_gantt(result, schedule_start, horizon)
                if resource_gantt: st.plotly_chart(resource_gantt, use_container_width=True)
                
        else:
            st.warning("No schedule has been generated yet. Please run the scheduler in the 'Scheduler' tab to see results.")


if __name__ == "__main__":
    main()