import streamlit as st
import pandas as pd
import config
from models.data_models import Line, ResourceStatus
from data_loader import DataLoader
from typing import Dict, Any

def render_line_manager(data_loader: DataLoader): 
    """
    Renders the Line configuration UI for editing and saving production lines.
    """
    st.subheader("🏭 Production Line Configuration")
    st.markdown("Manage properties of each production line.")

    # Initialize session state for line_config if not present
    if 'line_config' not in st.session_state:
        st.session_state.line_config = config.LINES.copy()

    st.markdown("### Production Lines Overview")
    
    # --- Prepare DataFrame for the main lines editor ---
    try:
        # Create display dataframe excluding the compatible SKUs column for main display
        lines_data_for_display = []
        for line_obj in st.session_state.line_config.values():
            line_dict = line_obj._to_dict()
            # Remove the complex dictionary/string for display in the main table
            # This column's data is managed in the separate editor
            line_dict.pop('Compatible_SKUs_Max_Production', None) 
            lines_data_for_display.append(line_dict)
        
        # Ensure Line_ID is always a string type in the DataFrame to prevent ArrowInvalid
        df_display = pd.DataFrame(lines_data_for_display)
        if 'Line_ID' in df_display.columns:
            df_display['Line_ID'] = df_display['Line_ID'].astype(str) # Force string type
        
    except Exception as e:
        st.error(f"Error preparing Line data for display: {e}")
        st.exception(e) # Show full traceback
        return
    
    # --- Main lines editor (without compatible SKUs column) ---
    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        key="line_editor",
        use_container_width=True,
        column_config={
            "Line_ID": st.column_config.TextColumn("Line ID", required=True),
            "CIP_Circuit": st.column_config.TextColumn("CIP Circuit"),
            "CIP_Duration_Min": st.column_config.NumberColumn("CIP Duration (Min)", min_value=0),
            "Status": st.column_config.SelectboxColumn("Status", options=[s.value for s in ResourceStatus]),
            "Setup_Time_Min": st.column_config.NumberColumn("Setup Time (Min)", min_value=0),
            "Current_SKU": st.column_config.TextColumn("Current SKU"),
            "Current_Product_Category": st.column_config.TextColumn("Current Product Category")
        }
    )

    # --- Update st.session_state.line_config based on main editor changes ---
    # This must happen immediately after data_editor to reflect row additions/deletions
    # and scalar edits before the next section relies on st.session_state.line_config.
    new_line_config_state = {}
    main_editor_errors = []

    for _, row in edited_df.iterrows():
        line_id = str(row["Line_ID"]).strip() # Ensure string and strip whitespace

        # Skip rows where Line_ID is empty or just whitespace (e.g., newly added empty rows)
        if not line_id or line_id == "nan": # 'nan' can appear if pd.NA is converted to str
            continue

        if line_id in new_line_config_state:
            main_editor_errors.append(f"Duplicate Line ID '{line_id}' found in the main table. Please ensure IDs are unique.")
            continue
        
        # Get the existing Line object from session state to preserve its compatible_skus_max_production
        # or create a new one if it's a new line
        existing_line_obj = st.session_state.line_config.get(line_id)
        
        try:
            # Create a new Line object with values from edited_df,
            # but retain compatible_skus_max_production from the existing object if available.
            current_line_obj = Line(
                line_id=line_id,
                CIP_circuit=str(row["CIP_Circuit"]) if pd.notna(row["CIP_Circuit"]) else None,
                cip_duration=int(row["CIP_Duration_Min"]) if pd.notna(row["CIP_Duration_Min"]) else 0,
                status=ResourceStatus(row["Status"]) if pd.notna(row["Status"]) else ResourceStatus.IDLE, # Default to IDLE if Status is missing
                setup_time_minutes=int(row["Setup_Time_Min"]) if pd.notna(row["Setup_Time_Min"]) else 0,
                current_sku=str(row["Current_SKU"]) if pd.notna(row["Current_SKU"]) else None,
                current_product_category=str(row["Current_Product_Category"]) if pd.notna(row["Current_Product_Category"]) else None,
                # Preserve the dictionary from the previous state, or set as empty dict for new lines
                compatible_skus_max_production=existing_line_obj.compatible_skus_max_production if existing_line_obj else {}
            )
            new_line_config_state[line_id] = current_line_obj

        except Exception as e:
            main_editor_errors.append(f"Error processing line '{line_id}' in main table: {e}")
            # Display full traceback for debugging specific data conversion issues in main editor
            st.exception(e) 

    # Display errors from the main editor
    if main_editor_errors:
        for error in main_editor_errors:
            st.error(f"❌ {error}")
        # It might be better to return here if the main data is fundamentally flawed,
        # but for user experience, let's allow them to try fixing compatible SKUs if other lines are fine.
        # return 

    # Overwrite the session state with the cleaned and updated lines from the main editor
    st.session_state.line_config = new_line_config_state

    # --- Step 2: Line Selection for SKU Management ---
    st.markdown("---") # Separator
    st.subheader("Compatible SKUs Management")
    st.markdown("#### Select a production line to manage its compatible SKUs and production rates.")
    
    # Get available line IDs from the now updated session state
    available_lines = sorted(list(st.session_state.line_config.keys()))
    
    if not available_lines:
        st.warning("No production lines available for SKU management. Please add lines in the 'Production Lines Overview' table first.")
        return # Exit if no lines exist
    
    # Initialize selected_line_id_for_skus in session state if not present or no longer valid
    if 'selected_line_id_for_skus' not in st.session_state or st.session_state.selected_line_id_for_skus not in available_lines:
        st.session_state.selected_line_id_for_skus = available_lines[0]

    # Line selection dropdown
    selected_line_id = st.selectbox(
        "Choose a production line:",
        options=available_lines,
        key="selected_line_for_skus",
        index=available_lines.index(st.session_state.selected_line_id_for_skus)
    )
    st.session_state.selected_line_id_for_skus = selected_line_id # Keep session state updated for next rerun

    # --- Step 3: SKU Management Table for Selected Line ---
    if selected_line_id:
        st.markdown(f"### Compatible SKUs for Line: **{selected_line_id}**")
        
        # Get current compatible SKUs for the selected line object from session state
        current_line = st.session_state.line_config.get(selected_line_id)
        compatible_skus_dict = current_line.compatible_skus_max_production if current_line else {}
        
        # Create dataframe for SKU editing
        sku_data = []
        for sku_id, production_rate in compatible_skus_dict.items():
            sku_data.append({
                "SKU_ID": sku_id,
                "Max_Production_Rate": production_rate
            })
        
        # If no compatible SKUs exist for the selected line, add an empty row for new entry
        if not sku_data:
            sku_data.append({"SKU_ID": "", "Max_Production_Rate": 0.0})
        
        sku_df_for_editor = pd.DataFrame(sku_data)
        
        # Get all existing SKU IDs for the selectbox options
        all_sku_ids_options = sorted(list(config.SKUS.keys())) if hasattr(config, 'SKUS') else []
        if not all_sku_ids_options:
            st.warning("No SKUs configured yet. Please go to the 'SKUs' tab to add SKUs first.")
            all_sku_ids_options = ["N/A - Add SKUs"] # Placeholder to avoid errors

        # SKU editor table for the selected line
        edited_sku_df = st.data_editor(
            sku_df_for_editor,
            num_rows="dynamic",
            key=f"sku_editor_for_line_{selected_line_id}", # Unique key for each line
            use_container_width=True,
            column_config={
                "SKU_ID": st.column_config.SelectboxColumn(
                    "SKU ID",
                    options=all_sku_ids_options,
                    required=True,
                    help="Select an existing SKU ID that this line can produce."
                ),
                "Max_Production_Rate": st.column_config.NumberColumn(
                    "Max Production Rate (L/min)",
                    min_value=0.0, # Allow 0 initially, but validate later
                    format="%.2f",
                    required=True,
                    help="Maximum production rate for this SKU on this line in Liters per minute."
                )
            }
        )
        
        # --- Validate and update SKUs for selected line ---
        if st.button(f"💾 Update Compatible SKUs for {selected_line_id}", key=f"update_skus_btn_{selected_line_id}"):
            sku_edit_errors = []
            updated_compatible_skus_for_line = {}
            
            for _, row in edited_sku_df.iterrows():
                sku_id = str(row.get("SKU_ID", "")).strip()
                production_rate = row.get("Max_Production_Rate")
                
                # Skip rows that are empty or have default placeholders
                if not sku_id or sku_id == "N/A - Add SKUs":
                    continue
                
                # Validate SKU ID exists in overall config
                if sku_id not in config.SKUS:
                    sku_edit_errors.append(f"SKU ID '{sku_id}' not found in SKU configurations. Please configure it in the 'SKUs' tab.")
                    continue
                
                # Validate production rate
                if pd.isna(production_rate) or not isinstance(production_rate, (int, float)) or production_rate <= 0:
                    sku_edit_errors.append(f"Production rate for SKU '{sku_id}' must be a positive number (got {production_rate}).")
                    continue
                
                # Check for duplicates within the current edited table
                if sku_id in updated_compatible_skus_for_line:
                    sku_edit_errors.append(f"Duplicate entry for SKU ID '{sku_id}' in this list.")
                    continue
                
                updated_compatible_skus_for_line[sku_id] = float(production_rate)
            
            if sku_edit_errors:
                for error_msg in sku_edit_errors:
                    st.error(f"❌ {error_msg}")
            else:
                # Update the compatible_skus_max_production for the selected line object in session state
                if current_line: # Ensure the line object exists
                    current_line.compatible_skus_max_production = updated_compatible_skus_for_line
                    st.success(f"✅ Compatible SKUs for '{selected_line_id}' updated successfully in memory. Click 'Save All Lines to CSV' below to persist changes.")
                    # No st.rerun() here to avoid clearing the main table's edits
                else:
                    st.error(f"Error: Line '{selected_line_id}' not found for updating compatible SKUs.")

    # --- Step 4: Save All Changes to CSV ---
    st.markdown("### Persist All Changes")
    st.markdown("Click this button to save all edits from both tables to the CSV file.")
    if st.button("💾 Save All Lines to CSV", use_container_width=True, type="primary"):
        with st.spinner("Saving All Line Configurations..."):
            try:
                final_save_errors = []
                final_lines_to_save = {}

                # Iterate over the comprehensive st.session_state.line_config
                # This ensures we save all lines, including those not edited in the main table
                # and those whose compatible SKUs were updated in the sub-editor.
                for line_id, line_obj in st.session_state.line_config.items():
                    # Perform final validation for each line before saving
                    print(line_id, line_obj)
                    if not line_obj.line_id:
                        final_save_errors.append(f"Line '{line_id}': ID cannot be empty.")
                        continue
                    # Add any other critical validation here (e.g., min CIP duration)

                    final_lines_to_save[line_id] = line_obj
                
                if final_save_errors:
                    for error_msg in final_save_errors:
                        st.error(f"❌ {error_msg}")
                    return # Stop save process if critical errors exist
                
                # Convert all Line objects to DataFrame for saving
                file_path = data_loader.data_dir / "line_config.csv"
                df_to_save = pd.DataFrame([line._to_dict() for line in final_lines_to_save.values()])
                print(df_to_save)
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ All Line configurations saved successfully to CSV!")
                
                # Reload all data into config and session state to ensure fresh state
                config.LINES = data_loader.load_lines_with_fallback() # Reloads all config objects
                st.session_state.line_config = config.LINES.copy() # Update session state copy
                st.rerun() # Rerun to refresh the UI with potentially new/reloaded data
                
            except Exception as e:
                st.error(f"❌ Error saving Line data: {e}")
                st.exception(e)

    # --- Step 5: Display Current Configuration Summary ---
    with st.expander("📋 Current Configuration Summary (Full Details)", expanded=False):
        st.markdown("### All Lines and Their Compatible SKUs (Read-Only)")
        if st.session_state.line_config:
            summary_data = []
            for line_id, line in st.session_state.line_config.items():
                compatible_skus_str = "; ".join([f"{sku_id}: {rate:.2f}" for sku_id, rate in line.compatible_skus_max_production.items()])
                if not compatible_skus_str:
                    compatible_skus_str = "No compatible SKUs configured."
                
                summary_data.append({
                    "Line ID": line.line_id,
                    "Status": line.status.value,
                    "CIP Duration (Min)": line.cip_duration,
                    "Setup Time (Min)": line.setup_time_minutes,
                    "Current SKU": line.current_sku if line.current_sku else "N/A",
                    "Compatible SKUs (SKU:Rate)": compatible_skus_str
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        else:
            st.info("No line configuration data available.")