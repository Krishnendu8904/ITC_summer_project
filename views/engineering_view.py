import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.data_loader import DataLoader
# --- Local Imports ---
import config
from utils.data_models import ResourceStatus

def save_eng_changes(data_loader):
    """Saves the status changes for all engineering assets to their respective CSV files."""
    try:
        # Save Lines
        if config.LINES:
            line_data = [line._to_dict() for line in config.LINES.values()]
            pd.DataFrame(line_data).to_csv(data_loader.data_dir / "line_config.csv", index=False)

        # Save Tanks
        if config.TANKS:
            tank_data = [tank._to_dict() for tank in config.TANKS.values()]
            pd.DataFrame(tank_data).to_csv(data_loader.data_dir / "tank_config.csv", index=False)

        # Save Equipment
        if config.EQUIPMENTS:
            equip_data = [equip._to_dict() for equip in config.EQUIPMENTS.values()]
            pd.DataFrame(equip_data).to_csv(data_loader.data_dir / "equipment_config.csv", index=False)

        st.success("✅ All engineering changes have been saved to files.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save engineering changes: {e}")
        return False

def render_asset_control(asset_type: str, assets: dict):
    """Generic function to render the control UI for a given asset type."""
    st.markdown(f"#### Manage {asset_type} Status")

    if not assets:
        st.info(f"No {asset_type} configured in the system.")
        return

    # Create a simple DataFrame for display
    asset_data = [{"ID": asset_id, "Status": asset.status.name} for asset_id, asset in assets.items()]
    df = pd.DataFrame(asset_data)
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Form for updating status
    with st.form(key=f"{asset_type}_status_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            asset_to_update = st.selectbox(f"Select {asset_type} ID", options=[""] + list(assets.keys()))
        
        with col2:
            new_status = st.selectbox("Set New Status", options=[s.name for s in ResourceStatus], key=f"{asset_type}_status")
        
        with col3:
            # Placeholder for duration. The data model needs to be updated to store this.
            duration_hours = st.number_input("Duration (hours, if applicable)", min_value=0, value=8, step=1)
            
        submitted = st.form_submit_button(f"Update {asset_type} Status", use_container_width=True)

        if submitted and asset_to_update:
            # Update the status in the config object
            assets[asset_to_update].status = ResourceStatus[new_status]
            # Note: The 'duration' is not saved yet as the data model doesn't support it.
            st.success(f"Updated {asset_type} '{asset_to_update}' status to {new_status}. Click 'Save All Changes' to persist.")
            
def render():
    """
    Renders the UI for the Engineering Control panel.
    """
    st.markdown('<h2 class="section-header">Engineering Control Panel</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="border-left-color: #3498DB;">
    Use this panel to manage the real-time operational status of key factory equipment. 
    Set assets to inactive or under maintenance to remove them from scheduling consideration.
    </div>
    """, unsafe_allow_html=True)
    data_loader = DataLoader()

    # Sub-tabs for each asset type
    tank_tab, line_tab, equip_tab = st.tabs(["**Tanks**", "**Lines**", "**Equipment**"])
    
    with tank_tab:
        if not config.TANKS:
            config.TANKS.update(data_loader.load_tanks_with_fallback())
        render_asset_control("Tank", config.TANKS)
    
    with line_tab:
        if not config.LINES:
            config.LINES.update(data_loader.load_lines_with_fallback())
        render_asset_control("Line", config.LINES)
        
    with equip_tab:
        if not config.EQUIPMENTS:
            config.EQUIPMENTS.update(data_loader.load_equipment_with_fallback())
        render_asset_control("Equipment", config.EQUIPMENTS)

    st.markdown("---")
    if st.button("💾 Save All Engineering Changes to Files", use_container_width=True, type="primary"):
        save_eng_changes(st.session_state.data_loader)
