import streamlit as st
import pandas as pd
from datetime import datetime

# --- Local Imports ---
import config
from managers.sku_manager import render_sku_manager
from managers.line_manager import render_line_manager
from managers.tank_manager import render_tank_manager
from managers.shift_manager import render_shift_manager
from managers.equipment_manager import render_equipment_manager
from managers.room_manager import render_room_manager
from managers.product_manager import render_product_manager
from managers.cip_manager import render_cip_circuit_manager

def render():
    """
    Renders the UI for the Factory Manager role, providing a complete overview
    and full control over all factory configurations.
    """
    st.markdown('<h2 class="section-header">Factory Command Center</h2>', unsafe_allow_html=True)

    # --- High-Level Metrics Dashboard ---
    st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📈 Plant-Wide KPIs</h3>', unsafe_allow_html=True)
    
    # Get metrics from the latest schedule run, if available
    result = st.session_state.get('last_schedule_result')
    total_volume = 0
    efficiency = "N/A"
    active_processes = "N/A" # Placeholder for now

    if result and result.scheduled_tasks:
        total_volume = sum(task.volume for task in result.scheduled_tasks)
        if hasattr(result, 'metrics') and hasattr(result.metrics, 'schedule_efficiency'):
            efficiency = f"{result.metrics.schedule_efficiency:.1%}"
        
        # A simple placeholder logic for active processes
        now = datetime.now()
        live_task_count = sum(1 for task in result.scheduled_tasks if task.start_time <= now <= task.end_time)
        active_processes = str(live_task_count)

    # Get other live metrics
    active_indents = len(config.USER_INDENTS)
    stock_levels = "N/A" # Placeholder for stock levels

    # Display the KPIs in two rows for better organization
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Indents", active_indents)
    with col2:
        st.metric("Active Processes", active_processes)
    with col3:
        st.metric("Stock Levels", stock_levels)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Scheduled Volume", f"{total_volume:,.0f} L")
    with col5:
        st.metric("Schedule Efficiency", efficiency)
    with col6:
         st.metric("Total SKUs", len(config.SKUS))


    st.markdown("---")

    # --- Full Configuration Management ---
    st.markdown('<h3 class="section-header" style="font-size:1.4rem;">⚙️ Full Factory Configuration</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="border-left-color: #003865;">
    As the Factory Manager, you have complete control over all system configurations. 
    Changes made in these tabs will be saved and will affect all other user roles.
    </div>
    """, unsafe_allow_html=True)

    # Using tabs to neatly organize all the individual manager modules
    (sku_tab, prod_tab, line_tab, tank_tab, equip_tab, room_tab, shift_tab, cip_tab) = st.tabs([
        "SKUs", "Products", "Lines", "Tanks", "Equipment", "Rooms", "Shifts", "CIP Circuits"
    ])
    
    data_loader = st.session_state.data_loader
    
    with sku_tab:
        render_sku_manager(data_loader)
    with prod_tab:
        render_product_manager(data_loader)
    with line_tab:
        render_line_manager(data_loader)
    with tank_tab:
        render_tank_manager(data_loader)
    with equip_tab:
        render_equipment_manager(data_loader)
    with room_tab:
        render_room_manager(data_loader)
    with shift_tab:
        render_shift_manager(data_loader)
    with cip_tab:
        render_cip_circuit_manager(data_loader)
