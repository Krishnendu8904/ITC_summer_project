import streamlit as st
from pathlib import Path
import config
from data_loader import DataLoader
import managers

# Import all manager modules
from managers.sku_manager import render_sku_manager
from managers.line_manager import render_line_manager
from managers.tank_manager import render_tank_manager
from managers.shift_manager import render_shift_manager
from indent_ui import render_indent_ui as render_user_indent_manager
from managers.equipment_manager import render_equipment_manager
from managers.room_manager import render_room_manager
from managers.product_manager import render_product_manager
from managers.cip_manager import render_cip_circuit_manager

# --- Page Configuration ---
st.set_page_config(
    page_title="Dairy Production Plant Management",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App State Management ---
def initialize_app_state():
    """Initializes the DataLoader and loads all data into session state."""
    if 'app_initialized' not in st.session_state:
        st.session_state.data_dir = Path("./data")
        st.session_state.data_dir.mkdir(exist_ok=True)
        
        # Instantiate and store a single DataLoader
        st.session_state.data_loader = DataLoader(data_dir=st.session_state.data_dir)
        
        # Load all data from CSVs (or create samples) into the central config objects
        st.session_state.data_loader.load_all_data()

        # Copy loaded data into session_state variables for interactive editing
        st.session_state.sku_config = config.SKUS.copy()
        st.session_state.line_config = config.LINES.copy()
        st.session_state.tank_config = config.TANKS.copy()
        st.session_state.shift_config = config.SHIFTS.copy()
        st.session_state.user_indents = config.USER_INDENTS.copy()
        st.session_state.equipment_config = config.EQUIPMENTS.copy()
        st.session_state.room_config = config.ROOMS.copy()
        st.session_state.product_config = config.PRODUCTS.copy()
        st.session_state.cip_circuit_config = config.CIP_CIRCUIT.copy()
        st.session_state.app_initialized = True

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.markdown('<h1 style="text-align:center; color:#00441B; font-size:2.5rem; font-weight:bold;">🏭 Dairy Production Plant Management</h1>', unsafe_allow_html=True)
    
    # Initialize the app state once
    initialize_app_state()
    data_loader = st.session_state.data_loader

    # --- Main Tab Layout ---
    tab1, tab2, tab3 = st.tabs(["**Factory Setup**", "**Scheduler**", "**Analytics**"])

    with tab1:
        st.header("⚙️ Factory Setup & Configuration")
        st.markdown("Visually manage all factory elements. Changes can be saved directly back to the source CSV files.")
        
        # Sub-tabs for each manager
        (sku_tab, line_tab, tank_tab, shift_tab, equip_tab, room_tab, product_tab, cip_tab) = st.tabs(
            ["SKUs", "Lines", "Tanks", "Shifts", "Equipment", "Rooms", "Products", "CIP Circuits"]
        )
        
        with sku_tab:
            render_sku_manager(data_loader)
        

        with line_tab:
            render_line_manager(data_loader)
            
        with tank_tab:
            render_tank_manager(data_loader)
            
        with shift_tab:
            render_shift_manager(data_loader)
            
        with equip_tab:
            render_equipment_manager(data_loader)
            
        with room_tab:
            render_room_manager(data_loader)
            
        with product_tab:
            render_product_manager(data_loader)
            
        with cip_tab:
            render_cip_circuit_manager(data_loader)

    with tab2:
        st.header("📅 Indent Accepting and Production Scheduler")
        st.info("Scheduler UI and logic will be integrated here.")
        indent_tab = st.tabs(
            ["Indent Tab"]
        )
        render_user_indent_manager(data_loader)
    

    with tab3:
        st.header("📊 Results & Analytics")
        st.info("Dashboards for schedule results and KPIs will be displayed here.")
        # Placeholder for results display
        # display_schedule_results(...)


if __name__ == "__main__":
    main()