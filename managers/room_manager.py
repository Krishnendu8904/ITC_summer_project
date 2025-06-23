import streamlit as st
import pandas as pd
import config
from utils.data_models import Room, RoomType, ResourceStatus
from utils.data_loader import DataLoader
from datetime import datetime
import numpy as np

def render_room_manager(data_loader: DataLoader):
    """
    Renders the Room configuration UI for editing and saving Rooms.
    """

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        st.subheader("🏠 Room Configuration")
        st.markdown("Define production rooms, their capacities, and environmental controls.")
    with col2:
        if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_rooms_btn"):
            try:
                config.ROOMS.update(data_loader.load_skus_with_fallback())
                st.session_state.sku_config = config.SKUS.copy()
            except Exception as e:
                st.error(f"❌ Error reloading data: {e}")

    # Use session state for interactive editing
    if 'room_config' not in st.session_state:
        st.session_state.room_config = config.ROOMS.copy()

    # Create list of room IDs for selectbox
    room_ids = list(st.session_state.room_config.keys())
    selectbox_options = ["-- Add New Room --"] + room_ids

    # Room selection dropdown
    selected_option = st.selectbox(
        "Select Room to Edit or Add New:",
        options=selectbox_options,
        key="room_selector"
    )

    # Determine if we're adding new or editing existing
    is_new_room = selected_option == "-- Add New Room --"
    selected_room = None if is_new_room else st.session_state.room_config.get(selected_option)

    # Get available options for dropdowns
    room_type_options = [rt.value for rt in RoomType]
    resource_status_options = [status.value for status in ResourceStatus]

    # Room editing form
    with st.form(key="room_form"):
        st.markdown("### Room Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            room_id = st.text_input(
                "Room ID",
                value="" if is_new_room else selected_room.room_id,
                disabled=not is_new_room,  # Disable editing ID for existing rooms
                help="Unique identifier for the room"
            )
            
            capacity_units = st.number_input(
                "Capacity (Units)",
                min_value=0.0,
                value=0.0 if is_new_room else selected_room.capacity_units,
                step=1.0,
                help="Maximum capacity of the room in units"
            )
            
            room_type = st.selectbox(
                "Room Type",
                options=room_type_options,
                index=0 if is_new_room else room_type_options.index(selected_room.room_type.value),
                help="Type/category of the room"
            )

        with col2:
            current_occupancy_units = st.number_input(
                "Current Occupancy (Units)",
                min_value=0.0,
                value=0.0 if is_new_room else selected_room.current_occupancy_units,
                step=1.0,
                help="Current occupancy level in units"
            )
            
            status = st.selectbox(
                "Status",
                options=resource_status_options,
                index=0 if is_new_room else resource_status_options.index(selected_room.status.value),
                help="Current operational status"
            )

        # Supported SKUs input
        supported_skus_str = st.text_area(
            "Supported SKUs (comma-separated)",
            value="" if is_new_room else ", ".join(selected_room.supported_skus),
            help="List of SKUs that can be produced in this room, separated by commas"
        )

        # Form submission buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if is_new_room:
                save_button = st.form_submit_button("🆕 Create New Room", type="primary")
            else:
                save_button = st.form_submit_button("💾 Save Changes", type="primary")

        with col2:
            if not is_new_room:
                # We'll handle delete outside the form since form_submit_button has limitations
                pass

        # Handle form submission
        if save_button:
            # Validation
            if not room_id or not room_id.strip():
                st.error("Room ID cannot be empty.")
            elif is_new_room and room_id in st.session_state.room_config:
                st.error(f"Room ID '{room_id}' already exists. Please choose a different ID.")
            else:
                try:
                    # Process supported SKUs
                    supported_skus = [sku.strip() for sku in supported_skus_str.split(',') if sku.strip()]
                    
                    # Convert enums
                    room_type_enum = RoomType(room_type)
                    status_enum = ResourceStatus(status)
                    
                    # Create/update room object
                    room_obj = Room(
                        room_id=room_id,
                        capacity_units=float(capacity_units),
                        supported_skus=supported_skus,
                        room_type=room_type_enum,
                        current_occupancy_units=float(current_occupancy_units),
                        status=status_enum,
                    )
                    
                    # Update session state
                    st.session_state.room_config[room_id] = room_obj
                    
                    if is_new_room:
                        st.success(f"✅ Room '{room_id}' created successfully!")
                    else:
                        st.success(f"✅ Room '{room_id}' updated successfully!")
                    
                    st.rerun()
                    
                except ValueError as e:
                    st.error(f"Invalid value: {e}")
                except Exception as e:
                    st.error(f"Error saving room: {e}")

    # Delete button (outside form to avoid form submission conflicts)
    if not is_new_room and selected_room:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Delete Selected Room", type="secondary"):
                # Use a confirmation dialog
                st.session_state.show_delete_confirmation = True
        
        # Handle delete confirmation
        if st.session_state.get('show_delete_confirmation', False):
            st.warning(f"Are you sure you want to delete room '{selected_option}'?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary"):
                    del st.session_state.room_config[selected_option]
                    st.session_state.show_delete_confirmation = False
                    st.success(f"Room '{selected_option}' deleted successfully!")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()

    # Display current rooms summary
    if st.session_state.room_config:
        st.markdown("---")
        st.markdown("### Current Rooms Summary")
        
        # Create summary dataframe
        summary_data = []
        for room_id, room in st.session_state.room_config.items():
            summary_data.append({
                "Room ID": room.room_id,
                "Type": room.room_type.value,
                "Capacity": room.capacity_units,
                "Current Occupancy": room.current_occupancy_units,
                "Status": room.status.value,
                "Supported SKUs": len(room.supported_skus)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    # Save to CSV button
    st.markdown("---")
    if st.button("💾 Save All Changes to CSV", use_container_width=True, type="primary", key= 'save_rooms_changes'):
        with st.spinner("Saving Rooms to CSV..."):
            try:
                # Validate all rooms before saving
                for room_id, room in st.session_state.room_config.items():
                    if not room_id:
                        st.error("Found room with empty ID. Please correct before saving.")
                        return

                # Save to CSV
                file_path = data_loader.data_dir / "room_config.csv"
                df_to_save = pd.DataFrame([room._to_dict() for room in st.session_state.room_config.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ All room configurations saved to CSV successfully!")

                # Reload data to reflect changes
                config.ROOMS = data_loader.load_rooms_with_fallback()
                st.session_state.room_config = config.ROOMS.copy()

            except Exception as e:
                st.error(f"❌ Error saving room data to CSV: {e}")

    # Show current count
    st.markdown(f"**Total Rooms:** {len(st.session_state.room_config)}")