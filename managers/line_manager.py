import streamlit as st
import pandas as pd
import config
from utils.data_models import Line, ResourceStatus
from utils.data_loader import DataLoader
from typing import Dict, Any, List, Optional

class LineManagerUI:
    """Handles the Line Manager UI with improved structure and organized editing."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for line management."""
        if 'line_config' not in st.session_state:
            st.session_state.line_config = config.LINES.copy()
        
        if 'selected_line_for_editing' not in st.session_state:
            st.session_state.selected_line_for_editing = None
            
        if 'line_editing_mode' not in st.session_state:
            st.session_state.line_editing_mode = 'overview'  # 'overview', 'edit_line', 'edit_skus'
            
        if 'show_line_deletion_confirm' not in st.session_state:
            st.session_state.show_line_deletion_confirm = False
    
    def render(self):
        """Main render method for the Line Manager UI."""
        
        self._render_header_controls()
        
        # Main content area based on editing mode
        if st.session_state.line_editing_mode == 'overview':
            self._render_lines_overview()
        elif st.session_state.line_editing_mode == 'edit_line':
            self._render_line_editor()
        elif st.session_state.line_editing_mode == 'edit_skus':
            self._render_sku_editor()
    
    def _render_header_controls(self):
        """Render the header controls (reload button)."""
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.subheader("🏭 Production Line Configuration")
            st.markdown("Manage properties of each production line and their compatible SKUs.")
        with col2:
            if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_lines_btn"):
                self._reload_lines()
    
    def _render_lines_overview(self):
        """Render the main lines overview with summary and actions."""
        st.markdown("### Production Lines Overview")
        
        # Lines summary table
        if st.session_state.line_config:
            self._render_lines_summary_table()
            
            # Action buttons for existing lines
            st.markdown("#### Line Management Actions")
            
            # Line selection for actions
            available_lines = sorted(list(st.session_state.line_config.keys()))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with st.container(border=True):
                    st.markdown("**Edit Line Properties**")
                    selected_line_props = st.selectbox(
                        "Select line to edit:",
                        options=available_lines,
                        key="line_props_selector"
                    )
                    if st.button("✏️ Edit Properties", use_container_width=True, type="secondary", key="edit_props_btn"):
                        st.session_state.selected_line_for_editing = selected_line_props
                        st.session_state.line_editing_mode = 'edit_line'
                        st.rerun()
            
            with col2:
                with st.container(border=True):
                    st.markdown("**Manage Compatible SKUs**")
                    selected_line_skus = st.selectbox(
                        "Select line for SKUs:",
                        options=available_lines,
                        key="line_skus_selector"
                    )
                    if st.button("🔧 Manage SKUs", use_container_width=True, type="secondary", key="manage_skus_btn"):
                        st.session_state.selected_line_for_editing = selected_line_skus
                        st.session_state.line_editing_mode = 'edit_skus'
                        st.rerun()
            
            with col3:
                with st.container(border=True):
                    st.markdown("**Delete Line**")
                    selected_line_delete = st.selectbox(
                        "Select line to delete:",
                        options=available_lines,
                        key="line_delete_selector"
                    )
                    if st.button("🗑️ Delete Line", use_container_width=True, type="secondary", key="delete_line_btn"):
                        self._initiate_line_deletion(selected_line_delete)
        
        else:
            st.info("No production lines configured yet. Create your first line below.")
        
        # Add new line section
        st.markdown("---")
        self._render_add_new_line_section()
        
        # Handle line deletion confirmation
        if st.session_state.show_line_deletion_confirm:
            self._render_deletion_confirmation()
        
        # Global save button
        st.markdown("---")
        if st.button("💾 Save All Changes", use_container_width=True, type="primary", key="save_all_lines_btn"):
            self._save_all_lines()
    
    def _render_lines_summary_table(self):
        """Render a comprehensive summary table of all lines."""
        summary_data = []
        for line_id, line in st.session_state.line_config.items():
            # Count compatible SKUs
            sku_count = len(line.compatible_skus_max_production)
            sku_list = list(line.compatible_skus_max_production.keys())[:3]  # Show first 3
            sku_preview = ", ".join(sku_list)
            if sku_count > 3:
                sku_preview += f" (+{sku_count - 3} more)"
            if not sku_preview:
                sku_preview = "None configured"
            
            # Calculate total production capacity
            total_capacity = sum(line.compatible_skus_max_production.values())
            
            summary_data.append({
                "Line ID": line.line_id,
                "Status": line.status,
                "Current SKU": line.current_sku or "None",
                "Current Product": line.current_product_category or "None",
                "CIP Duration (min)": line.CIP_duration_minutes,
                "Setup Time (min)": line.setup_time_minutes,
                "Compatible SKUs": f"{sku_count} SKUs",
                "SKU Preview": sku_preview,
                "Total Capacity (L/min)": f"{total_capacity:.2f}" if total_capacity > 0 else "0.00"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    def _render_add_new_line_section(self):
        """Render the section for adding new production lines."""
        with st.container(border=True):
            st.markdown("#### Add New Production Line")
            
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                new_line_id = st.text_input(
                    "New Line ID:",
                    key="new_line_id_input",
                    help="Enter a unique identifier for the new production line"
                )
            
            with col2:
                st.markdown("##")  # Space for alignment
                if st.button("➕ Add New Line", type="secondary", use_container_width=True, key="add_new_line_btn"):
                    self._add_new_line(new_line_id)
    
    def _render_line_editor(self):
        """Render the individual line properties editor."""
        if not st.session_state.selected_line_for_editing:
            st.error("No line selected for editing.")
            st.session_state.line_editing_mode = 'overview'
            st.rerun()
            return
        
        line_id = st.session_state.selected_line_for_editing
        line = st.session_state.line_config.get(line_id)
        
        if not line:
            st.error(f"Line '{line_id}' not found.")
            st.session_state.line_editing_mode = 'overview'
            st.rerun()
            return
        
        # Header with navigation
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col1:
            if st.button("← Back to Overview", type="secondary", key="back_to_overview_from_edit"):
                st.session_state.line_editing_mode = 'overview'
                st.session_state.selected_line_for_editing = None
                st.rerun()
        
        with col2:
            st.markdown(f"### Editing Line Properties: **{line_id}**")
        
        # Line properties form
        with st.container(border=True):
            st.markdown("#### Line Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                line_id_new = st.text_input(
                    "Line ID:",
                    value=line.line_id,
                    key="edit_line_id",
                    help="Unique identifier for this line"
                )
                
                cip_duration = st.number_input(
                    "CIP Duration (minutes):",
                    value=float(line.CIP_duration_minutes),
                    min_value=0.0,
                    key="edit_cip_duration",
                    help="Time required for Clean-in-Place operations"
                )
                
                setup_time = st.number_input(
                    "Setup Time (minutes):",
                    value=float(line.setup_time_minutes),
                    min_value=0.0,
                    key="edit_setup_time",
                    help="Time required to set up the line for production"
                )
            
            with col2:
                status = st.selectbox(
                    "Current Status:",
                    options=[s.value for s in ResourceStatus],
                    index=[s.value for s in ResourceStatus].index(line.status.value),
                    key="edit_status",
                    help="Current operational status of the line"
                )
                
                current_sku = st.text_input(
                    "Current SKU:",
                    value=line.current_sku or "",
                    key="edit_current_sku",
                    help="SKU currently being produced on this line"
                )
                
                current_product = st.text_input(
                    "Current Product Category:",
                    value=line.current_product_category or "",
                    key="edit_current_product",
                    help="Product category currently being produced"
                )
        
        # Current compatible SKUs display
        if line.compatible_skus_max_production:
            st.markdown("#### Current Compatible SKUs")
            sku_data = []
            for sku_id, rate in line.compatible_skus_max_production.items():
                sku_data.append({
                    "SKU ID": sku_id,
                    "Max Production Rate (L/min)": rate
                })
            
            df_skus = pd.DataFrame(sku_data)
            st.dataframe(df_skus, use_container_width=True, hide_index=True)
            
            if st.button("🔧 Edit Compatible SKUs", type="secondary", key="edit_skus_from_line_editor"):
                st.session_state.line_editing_mode = 'edit_skus'
                st.rerun()
        else:
            st.info("No compatible SKUs configured for this line.")
            if st.button("➕ Add Compatible SKUs", type="secondary", key="add_skus_from_line_editor"):
                st.session_state.line_editing_mode = 'edit_skus'
                st.rerun()
        
        # Save changes button
        st.markdown("---")
        col1, col2 = st.columns([0.7, 0.3])
        with col2:
            if st.button("💾 Save Line Properties", type="primary", use_container_width=True, key="save_line_props_btn"):
                self._save_line_properties(line_id, line_id_new, cip_duration, setup_time, status, current_sku, current_product)
    
    def _render_sku_editor(self):
        """Render the SKU compatibility editor for a specific line."""
        if not st.session_state.selected_line_for_editing:
            st.error("No line selected for SKU editing.")
            st.session_state.line_editing_mode = 'overview'
            st.rerun()
            return
        
        line_id = st.session_state.selected_line_for_editing
        line = st.session_state.line_config.get(line_id)
        
        if not line:
            st.error(f"Line '{line_id}' not found.")
            st.session_state.line_editing_mode = 'overview'
            st.rerun()
            return
        
        # Header with navigation
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col1:
            if st.button("← Back to Overview", type="secondary", key="back_to_overview_from_sku_editor"):
                st.session_state.line_editing_mode = 'overview'
                st.session_state.selected_line_for_editing = None
                st.rerun()
        
        with col2:
            st.markdown(f"### Managing Compatible SKUs: **{line_id}**")
        
        with col3:
            if st.button("✏️ Edit Line Properties", type="secondary", key="edit_line_props_from_sku_editor"):
                st.session_state.line_editing_mode = 'edit_line'
                st.rerun()
        
        # Get available SKUs
        all_sku_ids = sorted(list(config.SKUS.keys())) if hasattr(config, 'SKUS') and config.SKUS else []
        
        if not all_sku_ids:
            st.warning("⚠️ No SKUs configured yet. Please configure SKUs first before setting up line compatibility.")
            st.info("Go to the SKU Configuration tab to add SKUs.")
            return
        
        # Prepare SKU data for editor
        sku_data = []
        for sku_id, production_rate in line.compatible_skus_max_production.items():
            sku_data.append({
                "SKU_ID": sku_id,
                "Max_Production_Rate": production_rate
            })
        
        # Add empty row if no SKUs configured
        if not sku_data:
            sku_data.append({"SKU_ID": "", "Max_Production_Rate": 0.0})
        
        sku_df = pd.DataFrame(sku_data)
        
        # SKU compatibility editor
        st.markdown("#### Compatible SKUs Configuration")
        st.markdown(f"Configure which SKUs can be produced on **{line_id}** and their maximum production rates.")
        
        edited_sku_df = st.data_editor(
            sku_df,
            num_rows="dynamic",
            key=f"sku_editor_{line_id}",
            use_container_width=True,
            column_config={
                "SKU_ID": st.column_config.SelectboxColumn(
                    "SKU ID",
                    options=all_sku_ids,
                    required=True,
                    help="Select an existing SKU ID that this line can produce"
                ),
                "Max_Production_Rate": st.column_config.NumberColumn(
                    "Max Production Rate (L/min)",
                    min_value=0.01,
                    format="%.2f",
                    required=True,
                    help="Maximum production rate for this SKU on this line"
                )
            }
        )
        
        # SKU statistics
        if not edited_sku_df.empty:
            total_skus = len(edited_sku_df[edited_sku_df['SKU_ID'].str.strip() != ''])
            total_capacity = edited_sku_df['Max_Production_Rate'].fillna(0).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Compatible SKUs", total_skus)
            with col2:
                st.metric("Total Capacity", f"{total_capacity:.2f} L/min")
            with col3:
                if total_skus > 0:
                    avg_rate = total_capacity / total_skus
                    st.metric("Average Rate", f"{avg_rate:.2f} L/min")
        
        # Save SKU changes
        st.markdown("---")
        col1, col2 = st.columns([0.7, 0.3])
        with col2:
            if st.button("💾 Save SKU Configuration", type="primary", use_container_width=True, key="save_sku_config_btn"):
                self._save_sku_configuration(line_id, edited_sku_df)
    
    def _add_new_line(self, line_id: str):
        """Add a new production line."""
        line_id = line_id.strip()
        if not line_id:
            st.warning("Please enter a Line ID.")
            return
        
        if line_id in st.session_state.line_config:
            st.warning(f"Line ID '{line_id}' already exists.")
            return
        
        # Create new line with default values
        new_line = Line(
            line_id=line_id,
            CIP_duration_minutes=0,
            status=ResourceStatus.IDLE,
            setup_time_minutes=0,
            current_sku=None,
            current_product_category=None,
            compatible_skus_max_production={}
        )
        
        st.session_state.line_config[line_id] = new_line
        st.success(f"✅ Line '{line_id}' added successfully!")
        st.rerun()
    
    def _save_line_properties(self, original_line_id: str, new_line_id: str, cip_duration: float, 
                             setup_time: float, status: str, current_sku: str, current_product: str):
        """Save changes to line properties."""
        new_line_id = new_line_id.strip()
        
        # Validation
        if not new_line_id:
            st.error("Line ID cannot be empty.")
            return
        
        if new_line_id != original_line_id and new_line_id in st.session_state.line_config:
            st.error(f"Line ID '{new_line_id}' already exists.")
            return
        
        # Get the existing line
        line = st.session_state.line_config.get(original_line_id)
        if not line:
            st.error(f"Original line '{original_line_id}' not found.")
            return
        
        # Create updated line
        updated_line = Line(
            line_id=new_line_id,
            CIP_duration_minutes=int(cip_duration),
            status=ResourceStatus(status),
            setup_time_minutes=int(setup_time),
            current_sku=current_sku.strip() if current_sku.strip() else None,
            current_product_category=current_product.strip() if current_product.strip() else None,
            compatible_skus_max_production=line.compatible_skus_max_production.copy()
        )
        
        # Handle line ID change
        if new_line_id != original_line_id:
            del st.session_state.line_config[original_line_id]
            st.session_state.selected_line_for_editing = new_line_id
        
        st.session_state.line_config[new_line_id] = updated_line
        st.success(f"✅ Line properties saved successfully!")
        
        # Return to overview
        st.session_state.line_editing_mode = 'overview'
        st.session_state.selected_line_for_editing = None
        st.rerun()
    
    def _save_sku_configuration(self, line_id: str, edited_sku_df: pd.DataFrame):
        """Save SKU configuration changes for a line."""
        line = st.session_state.line_config.get(line_id)
        if not line:
            st.error(f"Line '{line_id}' not found.")
            return
        
        # Validate and process SKU data
        errors = []
        updated_skus = {}
        
        for index, row in edited_sku_df.iterrows():
            sku_id = str(row.get("SKU_ID", "")).strip()
            production_rate = row.get("Max_Production_Rate")
            
            # Skip empty rows
            if not sku_id:
                continue
            
            # Validate SKU exists
            if hasattr(config, 'SKUS') and sku_id not in config.SKUS:
                errors.append(f"Row {index + 1}: SKU ID '{sku_id}' not found in SKU configurations.")
                continue
            
            # Validate production rate
            if pd.isna(production_rate) or production_rate <= 0:
                errors.append(f"Row {index + 1}: Production rate must be greater than 0 (got {production_rate}).")
                continue
            
            # Check for duplicates
            if sku_id in updated_skus:
                errors.append(f"Row {index + 1}: Duplicate SKU ID '{sku_id}' found.")
                continue
            
            updated_skus[sku_id] = float(production_rate)
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            return
        
        # Update the line's compatible SKUs
        line.compatible_skus_max_production = updated_skus
        st.success(f"✅ Compatible SKUs for '{line_id}' updated successfully!")
        
        # Return to overview
        st.session_state.line_editing_mode = 'overview'
        st.session_state.selected_line_for_editing = None
        st.rerun()
    
    def _initiate_line_deletion(self, line_id: str):
        """Initiate line deletion process with confirmation."""
        st.session_state.line_to_delete = line_id
        st.session_state.show_line_deletion_confirm = True
        st.rerun()
    
    def _render_deletion_confirmation(self):
        """Render line deletion confirmation dialog."""
        line_id = st.session_state.get('line_to_delete')
        if not line_id:
            st.session_state.show_line_deletion_confirm = False
            return
        
        st.error(f"⚠️ **Confirm Line Deletion**")
        st.markdown(f"Are you sure you want to delete line **'{line_id}'** and all its compatible SKU configurations?")
        
        col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
        
        with col1:
            if st.button("✅ Yes, Delete", type="primary", use_container_width=True, key="confirm_delete_btn"):
                self._delete_line(line_id)
        
        with col2:
            if st.button("❌ Cancel", type="secondary", use_container_width=True, key="cancel_delete_btn"):
                st.session_state.show_line_deletion_confirm = False
                st.session_state.line_to_delete = None
                st.rerun()
    
    def _delete_line(self, line_id: str):
        """Delete a production line."""
        if line_id in st.session_state.line_config:
            del st.session_state.line_config[line_id]
            st.success(f"✅ Line '{line_id}' deleted successfully!")
        
        st.session_state.show_line_deletion_confirm = False
        st.session_state.line_to_delete = None
        st.rerun()
    
    def _save_all_lines(self):
        """Save all line configurations to CSV."""
        with st.spinner("Saving all line configurations..."):
            try:
                # Validate all lines before saving
                errors = []
                for line_id, line in st.session_state.line_config.items():
                    if not line.line_id.strip():
                        errors.append(f"Line '{line_id}': Line ID cannot be empty.")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                    return
                
                # Convert to DataFrame and save
                if st.session_state.line_config:
                    df_to_save = pd.DataFrame([line._to_dict() for line in st.session_state.line_config.values()])
                else:
                    # Create empty DataFrame with correct structure
                    df_to_save = pd.DataFrame(columns=[
                        "Line_ID", "CIP_Duration_Min", "Status", "Setup_Time_Min", 
                        "Current_SKU", "Current_Product_Category", "Compatible_SKUs_Max_Production"
                    ])
                
                file_path = self.data_loader.data_dir / "line_config.csv"
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ All line configurations saved successfully!")
                
                # Reload configuration
                config.LINES = self.data_loader.load_lines_with_fallback()
                st.session_state.line_config = config.LINES.copy()
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error saving line configurations: {e}")
                st.exception(e)
    
    def _reload_lines(self):
        """Reload lines from data source."""
        config.LINES.clear()
        config.LINES = self.data_loader.load_lines_with_fallback()
        st.session_state.line_config = config.LINES.copy()
        st.session_state.line_editing_mode = 'overview'
        st.session_state.selected_line_for_editing = None
        st.session_state.show_line_deletion_confirm = False
        st.rerun()


def render_line_manager(data_loader: DataLoader):
    """
    Main function to render the Line Manager UI.
    This maintains compatibility with the existing codebase.
    """
    line_manager = LineManagerUI(data_loader)
    line_manager.render()