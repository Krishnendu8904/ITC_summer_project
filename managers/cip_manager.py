import streamlit as st
import pandas as pd
import config
from models.data_models import CIP_circuit # Import CIPCircuit
from data_loader import DataLoader
import numpy as np # Import numpy to handle NaN values from data editor

def render_cip_circuit_manager(data_loader: DataLoader): # Changed function name
    """
    Renders the CIP Circuit configuration UI for editing and saving CIP Circuits.
    """
    st.subheader("💧 CIP Circuit Configuration") # Changed subheader
    st.markdown("Define Clean-In-Place (CIP) circuits and their connected resources.") # Changed description

    # Use session state for interactive editing
    if 'cip_circuit_config' not in st.session_state: # Changed session state key
        st.session_state.cip_circuit_config = config.CIP_CIRCUIT.copy() # Changed config.PRODUCTS to config.CIP_CIRCUIT

    try:
        # Convert CIPCircuit objects to dictionary for DataFrame display
        df_display = pd.DataFrame([circuit._to_dict() for circuit in st.session_state.cip_circuit_config.values()]) # Changed variable name
    except Exception as e:
        st.error(f"Error preparing CIP Circuit data for display: {e}") # Changed error message
        st.error("Please ensure the data models and CSV headers are consistent.")
        return

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        key="cip_circuit_editor", # Changed key
        use_container_width=True,
        column_config={
            "Circuit_ID": st.column_config.TextColumn("Circuit ID", required=True),
            "Connected_Resource_IDs": st.column_config.TextColumn("Connected Resource IDs (comma-separated)"),
            "Is_Available": st.column_config.CheckboxColumn("Is Available?", default=True), # Checkbox column
            "Standard_CIP_Duration_Min": st.column_config.NumberColumn("Standard CIP Duration (Min)", required=True, min_value=0),
        }
    )

    if st.button("💾 Save CIP Circuits to CSV", use_container_width=True, type="primary"): # Changed button text
        with st.spinner("Saving CIP Circuits..."): # Changed spinner text
            try:
                # Convert edited DF back to a dictionary of CIPCircuit objects for validation
                updated_cip_circuits = {} # Changed variable name
                for _, row in edited_df.iterrows():
                    circuit_id = str(row["Circuit_ID"]).strip()
                    if not circuit_id:
                        st.error("Circuit ID cannot be empty. Please correct and save again.") # Changed error message
                        return
                    if circuit_id in updated_cip_circuits: # Changed variable name
                        st.error(f"Duplicate Circuit ID '{circuit_id}' found. IDs must be unique.") # Changed error message
                        return
                    
                    # Handle Connected_Resource_IDs: split string by comma and strip whitespace
                    connected_resource_ids = [res.strip() for res in str(row["Connected_Resource_IDs"]).split(',') if res.strip()]
                    
                    # Convert Is_Available to boolean, handling potential None or NaN from editor
                    is_available = bool(row["Is_Available"]) if pd.notna(row["Is_Available"]) else True
                    
                    # Handle Standard_CIP_Duration_Min: ensure it's an integer
                    standard_cip_duration_minutes = int(row["Standard_CIP_Duration_Min"]) if pd.notna(row["Standard_CIP_Duration_Min"]) else 60

                    updated_cip_circuits[circuit_id] = CIP_circuit( # Changed class and variable name
                        circuit_id=circuit_id,
                        connected_resource_ids=connected_resource_ids,
                        is_available=is_available,
                        standard_cip_duration_minutes=standard_cip_duration_minutes
                    )

                # Get file path and save
                file_path = data_loader.data_dir / "cip_circuit_config.csv" # Changed file name
                df_to_save = pd.DataFrame([circuit._to_dict() for circuit in updated_cip_circuits.values()]) # Changed variable name
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ CIP Circuit configuration saved successfully!") # Changed success message

                # Reload data to reflect changes
                config.CIP_CIRCUIT = data_loader.load_cip_circuits_with_fallback() # Changed config and loader method
                st.session_state.cip_circuit_config = config.CIP_CIRCUIT.copy() # Changed session state key and config
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving CIP Circuit data: {e}") # Changed error message