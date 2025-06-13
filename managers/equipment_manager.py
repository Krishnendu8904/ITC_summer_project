import streamlit as st
import pandas as pd
import config
from models.data_models import Equipment, ResourceStatus # Import Equipment and ResourceStatus
from data_loader import DataLoader
from datetime import datetime # Import datetime for parsing times, though not directly used for Equipment's times in _to_dict


def render_equipment_manager(data_loader: DataLoader): # Changed function name
    """
    Renders the Equipment configuration UI for editing and saving Equipment.
    """
    st.subheader("⚙️ Equipment Configuration") # Changed subheader
    st.markdown("Define production equipment, their properties, and operational parameters.") # Changed description

    # Use session state for interactive editing
    if 'equipment_config' not in st.session_state: # Changed session state key
        st.session_state.equipment_config = config.EQUIPMENTS.copy() # Changed config.SKUS to config.EQUIPMENTS

    try:
        # Convert Equipment objects to dictionary for DataFrame display
        df_display = pd.DataFrame([equipment._to_dict() for equipment in st.session_state.equipment_config.values()]) # Changed variable name
    except Exception as e:
        st.error(f"Error preparing Equipment data for display: {e}") # Changed error message
        st.error("Please ensure the data models and CSV headers are consistent.")
        return

    # Get available ResourceStatus values for the Status column
    resource_status_options = [status.value for status in ResourceStatus]

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        key="equipment_editor", # Changed key
        use_container_width=True,
        column_config={
            "Equipment_ID": st.column_config.TextColumn("Equipment ID", required=True),
            "Processing Speed": st.column_config.NumberColumn("Processing Speed (L/min or Kg/min)", required=True, min_value=0.0),
            "Supported_Product_Categories": st.column_config.TextColumn("Supported Product Categories (comma-separated)"),
            "CIP_Circuit": st.column_config.TextColumn("CIP Circuit (Optional)"),
            "CIP_Duration_Min": st.column_config.NumberColumn("CIP Duration (Min)", min_value=0),
            "Status": st.column_config.SelectboxColumn("Status", options=resource_status_options, required=True), # Selectbox for Enum
            "Setup_Time_Min": st.column_config.NumberColumn("Setup Time (Min)", min_value=0),
            "Current_Product_Category": st.column_config.TextColumn("Current Product Category (Optional)"),
            # Note: start_up_time, last_cip_time, last_product_category are in the dataclass but not in _to_dict,
            # so they are not included in the data editor for consistency with saving logic.
        }
    )

    if st.button("💾 Save Equipment to CSV", use_container_width=True, type="primary"): # Changed button text
        with st.spinner("Saving Equipment..."): # Changed spinner text
            try:
                # Convert edited DF back to a dictionary of Equipment objects for validation
                updated_equipments = {} # Changed variable name
                for _, row in edited_df.iterrows():
                    equipment_id = row["Equipment_ID"]
                    if not equipment_id:
                        st.error("Equipment ID cannot be empty. Please correct and save again.") # Changed error message
                        return
                    if equipment_id in updated_equipments: # Changed variable name
                        st.error(f"Duplicate Equipment ID '{equipment_id}' found. IDs must be unique.") # Changed error message
                        return
                    
                    # Handle Supported_Product_Categories: split string by comma and strip whitespace
                    supported_categories = [cat.strip() for cat in str(row["Supported_Product_Categories"]).split(',') if cat.strip()]

                    # Handle Status: convert string back to ResourceStatus enum
                    try:
                        status_enum = ResourceStatus(row["Status"])
                    except ValueError:
                        st.error(f"Invalid Status for Equipment ID '{equipment_id}'. Please select a valid status.")
                        return

                    updated_equipments[equipment_id] = Equipment( # Changed class and variable name
                        equipment_id=equipment_id,
                        processing_speed=float(row["Processing Speed"]),
                        supported_product_categories=supported_categories,
                        cip_circuit=str(row["CIP_Circuit"]) if pd.notna(row["CIP_Circuit"]) else None,
                        cip_duration_minutes=int(row["CIP_Duration_Min"]),
                        status=status_enum,
                        setup_time_minutes=int(row["Setup_Time_Min"]),
                        current_product_category=str(row["Current_Product_Category"]) if pd.notna(row["Current_Product_Category"]) else None,
                        # Default values for start_up_time, last_cip_time, last_product_category
                        # as they are not edited or part of the provided _to_dict for saving.
                        start_up_time=0, 
                        last_cip_time=None,
                        last_product_category=None
                    )

                # Get file path and save
                file_path = data_loader.data_dir / "equipment_config.csv" # Changed file name
                df_to_save = pd.DataFrame([equipment._to_dict() for equipment in updated_equipments.values()]) # Changed variable name
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ Equipment configuration saved successfully!") # Changed success message

                # Reload data to reflect changes
                config.EQUIPMENTS = data_loader.load_equipment_with_fallback() # Changed config and loader method
                st.session_state.equipment_config = config.EQUIPMENTS.copy() # Changed session state key and config
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving Equipment data: {e}") # Changed error message