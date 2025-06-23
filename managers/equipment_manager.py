import streamlit as st
import pandas as pd
import config
from utils.data_models import Equipment, ResourceStatus # Import Equipment and ResourceStatus
from utils.data_loader import DataLoader
from datetime import datetime # Import datetime for parsing times, though not directly used for Equipment's times in _to_dict


def render_equipment_manager(data_loader: DataLoader):
    """
    Renders the Equipment configuration UI for editing and saving Equipment using select-then-edit pattern.
    """
    # Use session state for interactive editing
    if 'equipment_config' not in st.session_state:
        st.session_state.equipment_config = config.EQUIPMENTS.copy()

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        st.subheader("⚙️ Equipment Configuration")
        st.markdown("Define production equipment, their properties, and operational parameters.")
    with col2:
        if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_equipment_btn"):
            try:
                config.EQUIPMENTS.update(data_loader.load_equipment_with_fallback())
                st.session_state.equipment_config = config.EQUIPMENTS.copy()
            except Exception as e:
                st.error(f"❌ Error reloading data: {e}")
                st.exception(e)

    

    # Get list of equipment IDs for selectbox
    equipment_ids = list(st.session_state.equipment_config.keys())
    selectbox_options = ["-- Add New --"] + equipment_ids

    # Step 1: Select equipment to edit or add new
    selected_option = st.selectbox(
        "Select Equipment to Edit or Add New:",
        options=selectbox_options,
        key="equipment_selector"
    )

    # Step 2: Form for editing/creating equipment
    with st.form("equipment_form"):
        st.write("### Equipment Details")
        
        # Determine if we're editing or creating
        is_editing = selected_option != "-- Add New --"
        
        # Get available ResourceStatus values for default selection
        resource_status_options = [status.value for status in ResourceStatus]
        
        if is_editing:
            # Pre-fill form with existing equipment data
            selected_equipment = st.session_state.equipment_config[selected_option]
            default_equipment_id = selected_equipment.equipment_id
            default_processing_speed = selected_equipment.processing_speed
            default_compatible_categories = ", ".join(selected_equipment.compatible_product_categories)
            default_cip_duration = selected_equipment.CIP_duration_minutes
            default_status = selected_equipment.status.value
            default_setup_time = selected_equipment.setup_time_minutes
            default_current_category = selected_equipment.current_product_category or ""
            default_startup_time = selected_equipment.start_up_time
        else:
            # Empty form for new equipment
            default_equipment_id = ""
            default_processing_speed = 0.0
            default_compatible_categories = ""
            default_cip_duration = 0
            default_status = resource_status_options[0] if resource_status_options else ""  # Use first available status
            default_setup_time = 0
            default_current_category = ""
            default_startup_time = 0

        # Form fields
        equipment_id = st.text_input(
            "Equipment ID *",
            value=default_equipment_id,
            disabled=is_editing,  # Disable ID editing for existing equipment
            help="Unique identifier for the equipment"
        )
        
        processing_speed = st.number_input(
            "Processing Speed (L/min or Kg/min) *",
            min_value=0.0,
            value=float(default_processing_speed),
            step=0.1,
            help="Processing capacity of the equipment"
        )
        
        compatible_categories = st.text_input(
            "Compatible Product Categories",
            value=default_compatible_categories,
            help="Comma-separated list of product categories this equipment can handle"
        )
        
        cip_duration = st.number_input(
            "CIP Duration (Minutes)",
            min_value=0,
            value=int(default_cip_duration),
            help="Clean-in-place duration required"
        )
        
        status = st.selectbox(
            "Status *",
            options=resource_status_options,
            index=resource_status_options.index(default_status) if default_status in resource_status_options else 0,
            help="Current operational status of the equipment"
        )
        
        setup_time = st.number_input(
            "Setup Time (Minutes)",
            min_value=0,
            value=int(default_setup_time),
            help="Time required to set up the equipment for production"
        )
        
        current_category = st.text_input(
            "Current Product Category",
            value=default_current_category,
            help="Product category currently being processed (optional)"
        )
        
        startup_time = st.number_input(
            "Startup Time (Minutes)",
            min_value=0,
            value=int(default_startup_time),
            help="Time required to start up the equipment"
        )

        # Form submission buttons
        col1, col2 = st.columns(2)
        with col1:
            if is_editing:
                save_button = st.form_submit_button("💾 Save Changes", use_container_width=True)
            else:
                save_button = st.form_submit_button("➕ Create New Equipment", use_container_width=True, type="primary")
        
        with col2:
            if is_editing:
                reset_button = st.form_submit_button("🔄 Reset Form", use_container_width=True)

    # Handle reset button
    if is_editing and 'reset_button' in locals() and reset_button:
        st.rerun()

    # Handle form submission
    if save_button:
        # Validation
        if not equipment_id.strip():
            st.error("Equipment ID is required!")
            return
        
        if not is_editing and equipment_id in st.session_state.equipment_config:
            st.error(f"Equipment ID '{equipment_id}' already exists!")
            return
            
        try:
            # Parse compatible categories
            compatible_categories_list = [cat.strip() for cat in compatible_categories.split(',') if cat.strip()]
            
            # Convert status string to enum
            status_enum = ResourceStatus(status)
            
            # Create or update equipment object
            equipment_obj = Equipment(
                equipment_id=equipment_id,
                processing_speed=processing_speed,
                compatible_product_categories=compatible_categories_list,
                CIP_duration_minutes=cip_duration,
                status=status_enum,
                setup_time_minutes=setup_time,
                current_product_category=current_category if current_category.strip() else None,
                start_up_time=startup_time
            )
            
            # Add/update in session state
            st.session_state.equipment_config[equipment_id] = equipment_obj
            
            if is_editing:
                st.success(f"✅ Equipment '{equipment_id}' updated successfully!")
            else:
                st.success(f"✅ Equipment '{equipment_id}' created successfully!")
                
            st.rerun()
            
        except ValueError as e:
            st.error(f"Invalid status value: {e}")
        except Exception as e:
            st.error(f"Error saving equipment: {e}")

    # Step 3: Delete button (outside form)
    if selected_option != "-- Add New --":
        st.write("---")
        if st.button("🗑️ Delete Selected Equipment", use_container_width=True, type="secondary", key="delete_equipment"):
            if st.session_state.get('confirm_delete_equipment') != selected_option:
                st.session_state.confirm_delete_equipment = selected_option
                st.warning(f"Click again to confirm deletion of equipment '{selected_option}'")
            else:
                del st.session_state.equipment_config[selected_option]
                if 'confirm_delete_equipment' in st.session_state:
                    del st.session_state.confirm_delete_equipment
                st.success(f"✅ Equipment '{selected_option}' deleted successfully!")
                st.rerun()

    # Step 4: Display current equipment summary
    st.write("---")
    st.write("### Current Equipment Summary")
    if st.session_state.equipment_config:
        summary_data = []
        for eq_id, equipment in st.session_state.equipment_config.items():
            summary_data.append({
                "Equipment ID": eq_id,
                "Processing Speed": f"{equipment.processing_speed} L/min",
                "Status": equipment.status.value,
                "Compatible Categories": ", ".join(equipment.compatible_product_categories),
                "CIP Duration": f"{equipment.CIP_duration_minutes} min"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No equipment configured yet.")

    # Step 5: Save all changes to CSV
    st.write("---")
    if st.button("💾 Save All Changes to CSV", use_container_width=True, type="primary", key="save_all_equipment"):
        with st.spinner("Saving equipment configuration..."):
            try:
                # Get file path and save
                file_path = data_loader.data_dir / "equipment_config.csv"
                df_to_save = pd.DataFrame([equipment._to_dict() for equipment in st.session_state.equipment_config.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ Equipment configuration saved to CSV successfully!")
                
                # Reload data to reflect changes
                config.EQUIPMENTS = data_loader.load_equipment_with_fallback()
                st.session_state.equipment_config = config.EQUIPMENTS.copy()
                
            except Exception as e:
                st.error(f"❌ Error saving equipment data: {e}")
                st.exception(e)