import streamlit as st
import pandas as pd
import config
from utils.data_models import CIP_circuit # Import CIPCircuit
from utils.data_loader import DataLoader
import numpy as np # Import numpy to handle NaN values from data editor

def render_cip_circuit_manager(data_loader: DataLoader):
    """
    Renders the CIP Circuit configuration UI for editing and saving CIP Circuits using select-then-edit pattern.
    """
    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        st.subheader("💧 CIP Circuit Configuration")
        st.markdown("Define Clean-In-Place (CIP) circuits and their connected resources.")
    with col2:
        if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_cip_btn"):
            try:
                config.SKUS.update(data_loader.load_CIP_circuits_with_fallback())
                st.session_state.cip_circuit_config = config.CIP_CIRCUIT.copy()
            except Exception as e:
                st.error(f"❌ Error reloading data: {e}")

    st.markdown("---")
    

    # Use session state for interactive editing
    if 'cip_circuit_config' not in st.session_state:
        st.session_state.cip_circuit_config = config.CIP_CIRCUIT.copy()

    # Get list of CIP circuit IDs for selectbox
    circuit_ids = list(st.session_state.cip_circuit_config.keys())
    selectbox_options = ["-- Add New --"] + circuit_ids

    # Step 1: Select CIP circuit to edit or add new
    selected_option = st.selectbox(
        "Select CIP Circuit to Edit or Add New:",
        options=selectbox_options,
        key="cip_circuit_selector"
    )

    # Step 2: Form for editing/creating CIP circuit
    with st.form("cip_circuit_form"):
        st.write("### CIP Circuit Details")
        
        # Determine if we're editing or creating
        is_editing = selected_option != "-- Add New --"
        
        if is_editing:
            # Pre-fill form with existing CIP circuit data
            selected_circuit = st.session_state.cip_circuit_config[selected_option]
            default_circuit_id = selected_circuit.circuit_id
            default_connected_resources = ", ".join(selected_circuit.connected_resource_ids)
            default_is_available = selected_circuit.is_available
        else:
            # Empty form for new CIP circuit
            default_circuit_id = ""
            default_connected_resources = ""
            default_is_available = True

        # Form fields
        circuit_id = st.text_input(
            "Circuit ID *",
            value=default_circuit_id,
            disabled=is_editing,  # Disable ID editing for existing circuits
            help="Unique identifier for the CIP circuit"
        )
        
        connected_resources = st.text_input(
            "Connected Resource IDs",
            value=default_connected_resources,
            help="Comma-separated list of resource IDs connected to this CIP circuit"
        )
        
        is_available = st.checkbox(
            "Is Available",
            value=default_is_available,
            help="Whether this CIP circuit is currently available for use"
        )

        # Form submission buttons
        col1, col2 = st.columns(2)
        with col1:
            if is_editing:
                save_button = st.form_submit_button("💾 Save Changes", use_container_width=True)
            else:
                save_button = st.form_submit_button("➕ Create New CIP Circuit", use_container_width=True, type="primary")
        
        with col2:
            if is_editing:
                reset_button = st.form_submit_button("🔄 Reset Form", use_container_width=True)

    # Handle reset button
    if is_editing and 'reset_button' in locals() and reset_button:
        st.rerun()

    # Handle form submission
    if save_button:
        # Validation
        if not circuit_id.strip():
            st.error("Circuit ID is required!")
            return
        
        if not is_editing and circuit_id in st.session_state.cip_circuit_config:
            st.error(f"Circuit ID '{circuit_id}' already exists!")
            return
            
        try:
            # Parse connected resource IDs
            connected_resource_ids = [res.strip() for res in connected_resources.split(',') if res.strip()]
            
            # Create or update CIP circuit object
            circuit_obj = CIP_circuit(
                circuit_id=circuit_id,
                connected_resource_ids=connected_resource_ids,
                is_available=is_available
            )
            
            # Add/update in session state
            st.session_state.cip_circuit_config[circuit_id] = circuit_obj
            
            if is_editing:
                st.success(f"✅ CIP Circuit '{circuit_id}' updated successfully!")
            else:
                st.success(f"✅ CIP Circuit '{circuit_id}' created successfully!")
                
            st.rerun()
            
        except Exception as e:
            st.error(f"Error saving CIP circuit: {e}")

    # Step 3: Delete button (outside form)
    if selected_option != "-- Add New --":
        st.write("---")
        if st.button("🗑️ Delete Selected CIP Circuit", use_container_width=True, type="secondary", key="delete_cip_circuit"):
            if st.session_state.get('confirm_delete_cip') != selected_option:
                st.session_state.confirm_delete_cip = selected_option
                st.warning(f"Click again to confirm deletion of CIP circuit '{selected_option}'")
            else:
                del st.session_state.cip_circuit_config[selected_option]
                if 'confirm_delete_cip' in st.session_state:
                    del st.session_state.confirm_delete_cip
                st.success(f"✅ CIP Circuit '{selected_option}' deleted successfully!")
                st.rerun()

    # Step 4: Display current CIP circuits summary
    st.write("---")
    st.write("### Current CIP Circuits Summary")
    if st.session_state.cip_circuit_config:
        summary_data = []
        for circuit_id, circuit in st.session_state.cip_circuit_config.items():
            summary_data.append({
                "Circuit ID": circuit_id,
                "Connected Resources": ", ".join(circuit.connected_resource_ids) if circuit.connected_resource_ids else "None",
                "Available": "✅ Yes" if circuit.is_available else "❌ No",
                "Resource Count": len(circuit.connected_resource_ids)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No CIP circuits configured yet.")

    # Step 5: Save all changes to CSV
    st.write("---")
    if st.button("💾 Save All Changes to CSV", use_container_width=True, type="primary", key="save_all_cip_circuits"):
        with st.spinner("Saving CIP circuit configuration..."):
            try:
                # Get file path and save
                file_path = data_loader.data_dir / "cip_circuit_config.csv"
                df_to_save = pd.DataFrame([circuit._to_dict() for circuit in st.session_state.cip_circuit_config.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ CIP Circuit configuration saved to CSV successfully!")
                
                # Reload data to reflect changes
                config.CIP_CIRCUIT = data_loader.load_CIP_circuits_with_fallback()
                st.session_state.cip_circuit_config = config.CIP_CIRCUIT.copy()
                
            except Exception as e:
                st.error(f"❌ Error saving CIP circuit data: {e}")
                st.exception(e)