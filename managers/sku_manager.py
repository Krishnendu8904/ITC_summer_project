import streamlit as st
import pandas as pd
import config
from utils.data_models import SKU
from utils.data_loader import DataLoader
from typing import Dict, Optional, List
import logging

class SKUManager:
    """Enhanced SKU Manager with better state management and error handling."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        
    def _initialize_session_state(self):
        """Initialize session state variables if they don't exist."""
        if 'sku_config' not in st.session_state:
            st.session_state.sku_config = config.SKUS.copy()
        
        if 'confirm_delete' not in st.session_state:
            st.session_state.confirm_delete = False
            
        if 'selected_sku_id' not in st.session_state:
            st.session_state.selected_sku_id = None
            
        if 'show_success_message' not in st.session_state:
            st.session_state.show_success_message = None
    
    def _load_fresh_data(self):
        """Load fresh data from files and update session state."""
        try:
            config.PRODUCTS.update(self.data_loader.load_products_with_fallback())
            config.SKUS.update(self.data_loader.load_skus_with_fallback())
            st.session_state.sku_config = config.SKUS.copy()
        except Exception as e:
            st.error(f"❌ Error reloading data: {e}")
            self.logger.error(f"Error reloading data: {e}")
    
    def _validate_sku_input(self, sku_id: str, is_adding_new: bool) -> Optional[str]:
        """Validate SKU input and return error message if invalid."""
        if not sku_id or not sku_id.strip():
            return "SKU ID cannot be empty."
        
        sku_id = sku_id.strip()
        
        # Check for invalid characters
        if not sku_id.replace('_', '').replace('-', '').isalnum():
            return "SKU ID can only contain letters, numbers, hyphens, and underscores."
        
        # Check for duplicates when adding new
        if is_adding_new and sku_id in st.session_state.sku_config:
            return f"SKU ID '{sku_id}' already exists. Please choose a different ID."
        
        return None
    
    def _create_or_update_sku(self, sku_id: str, product_category: str, 
                             inventory_size: float, is_adding_new: bool) -> bool:
        """Create or update SKU. Returns True if successful."""
        try:
            # Validate input
            error_msg = self._validate_sku_input(sku_id, is_adding_new)
            if error_msg:
                st.error(error_msg)
                return False
            
            # Validate product category exists
            if product_category not in config.PRODUCTS:
                st.error(f"Product category '{product_category}' does not exist.")
                return False
            
            # Create SKU object
            new_sku = SKU(
                sku_id=sku_id.strip(),
                product_category=product_category,
                inventory_size=inventory_size
            )
            
            # Update session state
            st.session_state.sku_config[new_sku.sku_id] = new_sku
            
            # Set success message
            action = "created" if is_adding_new else "updated"
            st.session_state.show_success_message = f"✅ SKU '{new_sku.sku_id}' {action} successfully!"
            
            # Update selected SKU for editing mode
            if is_adding_new:
                st.session_state.selected_sku_id = new_sku.sku_id
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing SKU: {e}")
            self.logger.error(f"Error processing SKU: {e}")
            return False
    
    def _delete_sku(self, sku_id: str) -> bool:
        """Delete SKU. Returns True if successful."""
        try:
            if sku_id in st.session_state.sku_config:
                del st.session_state.sku_config[sku_id]
                st.session_state.show_success_message = f"✅ SKU '{sku_id}' deleted successfully!"
                st.session_state.selected_sku_id = None
                st.session_state.confirm_delete = False
                return True
            else:
                st.error(f"SKU '{sku_id}' not found.")
                return False
        except Exception as e:
            st.error(f"❌ Error deleting SKU: {e}")
            self.logger.error(f"Error deleting SKU: {e}")
            return False
    
    def _save_to_csv(self) -> bool:
        """Save all SKUs to CSV file. Returns True if successful."""
        try:
            file_path = self.data_loader.data_dir / "sku_config.csv"
            
            if st.session_state.sku_config:
                df_to_save = pd.DataFrame([
                    sku._to_dict() for sku in st.session_state.sku_config.values()
                ])
            else:
                # Create empty DataFrame with proper columns if no SKUs
                df_to_save = pd.DataFrame(columns=["SKU_ID", "Product_Category", "Inventory_Size"])
            
            df_to_save.to_csv(file_path, index=False)
            
            # Reload data to sync with file
            config.SKUS = self.data_loader.load_skus_with_fallback()
            st.session_state.sku_config = config.SKUS.copy()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving SKU data to CSV: {e}")
            self.logger.error(f"Error saving SKU data to CSV: {e}")
            return False
    
    def _render_header(self):
        """Render the header section with title and reload button."""
        col1, col2 = st.columns([0.8, 0.2])
        
        with col1:
            st.subheader("🥛 SKU Configuration")
            st.markdown("Define product SKUs, their properties, and associated production parameters.")
        
        with col2:
            if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_skus_btn"):
                self._load_fresh_data()
    
    def _render_sku_selector(self) -> tuple[bool, Optional[str]]:
        """Render SKU selector and return (is_adding_new, selected_sku_id)."""
        existing_sku_ids = sorted(list(st.session_state.sku_config.keys()))
        selectbox_options = ["-- Add a New SKU --"] + existing_sku_ids
        
        # Use session state to maintain selection
        if st.session_state.selected_sku_id and st.session_state.selected_sku_id in existing_sku_ids:
            default_index = existing_sku_ids.index(st.session_state.selected_sku_id) + 1
        else:
            default_index = 0
        
        selected_option = st.selectbox(
            "Select SKU to Edit or Add New:",
            options=selectbox_options,
            index=default_index,
            key="sku_selector"
        )
        
        is_adding_new = selected_option == "-- Add a New SKU --"
        selected_sku_id = None if is_adding_new else selected_option
        
        # Update session state
        st.session_state.selected_sku_id = selected_sku_id
        
        return is_adding_new, selected_sku_id
    
    def _render_sku_form(self, is_adding_new: bool, selected_sku_id: Optional[str]):
        """Render the SKU editing form."""
        selected_sku = None
        if not is_adding_new and selected_sku_id:
            selected_sku = st.session_state.sku_config.get(selected_sku_id)
        
        form_key = "add_sku_form" if is_adding_new else f"edit_sku_form_{selected_sku_id}"
        
        with st.form(key=form_key):
            st.write("### SKU Details")
            
            # SKU ID field
            if is_adding_new:
                sku_id_input = st.text_input(
                    "SKU ID *",
                    value="",
                    placeholder="Enter unique SKU ID (letters, numbers, -, _ only)",
                    help="SKU ID must be unique and contain only letters, numbers, hyphens, and underscores."
                )
            else:
                sku_id_input = st.text_input(
                    "SKU ID *",
                    value=selected_sku.sku_id if selected_sku else "",
                    disabled=True,
                    help="SKU ID cannot be changed after creation."
                )
            
            # Product Category dropdown
            product_categories = sorted(list(config.PRODUCTS.keys()))
            
            if not product_categories:
                st.error("⚠️ No product categories available. Please configure products first.")
                return
            
            if selected_sku and selected_sku.product_category in product_categories:
                default_category_index = product_categories.index(selected_sku.product_category)
            else:
                default_category_index = 0
            
            product_category_input = st.selectbox(
                "Product Category *",
                options=product_categories,
                index=default_category_index,
                help="Select the product category this SKU belongs to."
            )
            
            # Inventory Size input
            inventory_size_input = st.number_input(
                "Inventory Size (L/Unit) *",
                min_value=0.01,
                max_value=10000.0,
                value=float(selected_sku.inventory_size) if selected_sku else 1.0,
                step=0.01,
                help="Volume per unit in liters (must be greater than 0.01)."
            )
            
            # Form submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if is_adding_new:
                    submit_button = st.form_submit_button("Create New SKU", type="primary", use_container_width=True)
                else:
                    submit_button = st.form_submit_button("Save Changes", type="primary", use_container_width=True)
            
            # Handle form submission
            if submit_button:
                success = self._create_or_update_sku(
                    sku_id_input, 
                    product_category_input, 
                    inventory_size_input, 
                    is_adding_new
                )
                if success:
                    st.rerun()
    
    def _render_delete_section(self, selected_sku_id: str):
        """Render the delete section for existing SKUs."""
        st.write("---")
        st.write("### Delete SKU")
        
        if not st.session_state.confirm_delete:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    f"🗑️ Delete SKU '{selected_sku_id}'",
                    type="secondary",
                    use_container_width=True,
                    help="This action cannot be undone."
                ):
                    st.session_state.confirm_delete = True
                    st.rerun()
        else:
            st.warning(f"⚠️ Are you sure you want to delete SKU '{selected_sku_id}'?")
            st.write("This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                    if self._delete_sku(selected_sku_id):
                        st.rerun()
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()
    
    def _render_summary_section(self):
        """Render the summary and save section."""
        st.write("---")
        st.write("### Configuration Summary")
        
        if st.session_state.sku_config:
            st.write(f"**Total SKUs configured:** {len(st.session_state.sku_config)}")
            
            # Create summary table
            summary_data = []
            for sku_id, sku in sorted(st.session_state.sku_config.items()):
                summary_data.append({
                    "SKU ID": sku.sku_id,
                    "Product Category": sku.product_category,
                    "Inventory Size (L/Unit)": f"{sku.inventory_size:.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Display with expander for better UX
            with st.expander("📋 View All SKUs", expanded=len(summary_data) <= 5):
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No SKUs configured yet.")
        
        # Save button
        st.write("### Save Configuration")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Save All Changes to CSV", 
                        type="primary", 
                        use_container_width=True,
                        disabled=len(st.session_state.sku_config) == 0):
                with st.spinner("💾 Saving changes..."):
                    if self._save_to_csv():
                        st.success("✅ All changes saved successfully!")
    
    def _show_success_message(self):
        """Display success message if available."""
        if st.session_state.show_success_message:
            st.success(st.session_state.show_success_message)
            st.session_state.show_success_message = None
    
    def render(self):
        """Main render method for the SKU manager."""
        # Initialize session state
        self._initialize_session_state()
        
        # Load initial data
        config.PRODUCTS.update(self.data_loader.load_products_with_fallback())
        config.SKUS.update(self.data_loader.load_skus_with_fallback())
        
        # Render header
        self._render_header()
        
        # Show success message if any
        self._show_success_message()
        
        # Render SKU selector
        is_adding_new, selected_sku_id = self._render_sku_selector()
        
        # Render main form
        self._render_sku_form(is_adding_new, selected_sku_id)
        
        # Render delete section for existing SKUs
        if not is_adding_new and selected_sku_id:
            self._render_delete_section(selected_sku_id)
        
        # Render summary and save section
        self._render_summary_section()


def render_sku_manager(data_loader: DataLoader):
    """
    Main function to render the SKU manager interface.
    This maintains backward compatibility with the existing codebase.
    """
    manager = SKUManager(data_loader)
    manager.render()