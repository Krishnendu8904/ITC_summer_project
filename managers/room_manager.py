import streamlit as st
import pandas as pd
import config
from models.data_models import Room, RoomType, ResourceStatus # Import Room, RoomType, ResourceStatus
from data_loader import DataLoader
from datetime import datetime # Keep datetime for general utility, though not specifically for Room's times here
import numpy as np # Import numpy to handle NaN values from data editor

def render_room_manager(data_loader: DataLoader): # Changed function name
    """
    Renders the Room configuration UI for editing and saving Rooms.
    """
    st.subheader("🏠 Room Configuration") # Changed subheader
    st.markdown("Define production rooms, their capacities, and environmental controls.") # Changed description

    # Use session state for interactive editing
    if 'room_config' not in st.session_state: # Changed session state key
        st.session_state.room_config = config.ROOMS.copy() # Changed config.SKUS to config.ROOMS

    try:
        # Convert Room objects to dictionary for DataFrame display
        df_display = pd.DataFrame([room._to_dict() for room in st.session_state.room_config.values()]) # Changed variable name
    except Exception as e:
        st.error(f"Error preparing Room data for display: {e}") # Changed error message
        st.error("Please ensure the data models and CSV headers are consistent.")
        return

    # Get available RoomType and ResourceStatus values for selectbox columns
    room_type_options = [rt.value for rt in RoomType]
    resource_status_options = [status.value for status in ResourceStatus]

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        key="room_editor", # Changed key
        use_container_width=True,
        column_config={
            "Room_ID": st.column_config.TextColumn("Room ID", required=True),
            "Capacity_Units": st.column_config.NumberColumn("Capacity (Units)", required=True, min_value=0.0),
            "Supported_SKUs": st.column_config.TextColumn("Supported SKUs (comma-separated)"),
            "Room_Type": st.column_config.SelectboxColumn("Room Type", options=room_type_options, required=True),
            "Current_Occupancy_Units": st.column_config.NumberColumn("Current Occupancy (Units)", min_value=0.0),
            "Status": st.column_config.SelectboxColumn("Status", options=resource_status_options, required=True),
        }
    )

    if st.button("💾 Save Rooms to CSV", use_container_width=True, type="primary"): # Changed button text
        with st.spinner("Saving Rooms..."): # Changed spinner text
            try:
                # Convert edited DF back to a dictionary of Room objects for validation
                updated_rooms = {} # Changed variable name
                for _, row in edited_df.iterrows():
                    room_id = row["Room_ID"]
                    if not room_id:
                        st.error("Room ID cannot be empty. Please correct and save again.") # Changed error message
                        return
                    if room_id in updated_rooms: # Changed variable name
                        st.error(f"Duplicate Room ID '{room_id}' found. IDs must be unique.") # Changed error message
                        return
                    
                    # Handle Supported_SKUs: split string by comma and strip whitespace
                    supported_skus = [sku.strip() for sku in str(row["Supported_SKUs"]).split(',') if sku.strip()]

                    # Handle Room_Type: convert string back to RoomType enum
                    try:
                        room_type_enum = RoomType(row["Room_Type"])
                    except ValueError:
                        st.error(f"Invalid Room Type for Room ID '{room_id}'. Please select a valid type.")
                        return

                    # Handle Status: convert string back to ResourceStatus enum
                    try:
                        status_enum = ResourceStatus(row["Status"])
                    except ValueError:
                        st.error(f"Invalid Status for Room ID '{room_id}'. Please select a valid status.")
                        return
                    
                    # Handle optional float columns (Temperature, Humidity)

                    updated_rooms[room_id] = Room( # Changed class and variable name
                        room_id=room_id,
                        capacity_units=float(row["Capacity_Units"]),
                        supported_skus=supported_skus,
                        room_type=room_type_enum,
                        current_occupancy_units=float(row["Current_Occupancy_Units"]),
                        status=status_enum,
                    )

                # Get file path and save
                file_path = data_loader.data_dir / "room_config.csv" # Changed file name
                df_to_save = pd.DataFrame([room._to_dict() for room in updated_rooms.values()]) # Changed variable name
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ Room configuration saved successfully!") # Changed success message

                # Reload data to reflect changes
                config.ROOMS = data_loader.load_rooms_with_fallback() # Changed config and loader method
                st.session_state.room_config = config.ROOMS.copy() # Changed session state key and config
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving Room data: {e}") # Changed error message