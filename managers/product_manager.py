import streamlit as st
import pandas as pd
import config
from utils.data_models import Product, ProcessingStep, ResourceType, ResourceRequirement, ProcessType
from utils.data_loader import DataLoader
import numpy as np
from typing import List, Dict, Any, Optional

class ProductManagerUI:
    """Handles the Product Manager UI with a step-by-step editing interface."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initializes the Product Manager UI.

        Args:
            data_loader (DataLoader): An instance of the data loader.
        """
        self.data_loader = data_loader
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for product and step management."""
        if 'product_config_display' not in st.session_state:
            st.session_state.product_config_display = {
                product.product_category: product for product in config.PRODUCTS.values()
            }
        
        if 'selected_product_for_editing' not in st.session_state:
            st.session_state.selected_product_for_editing = None
            
        # Overall mode for the product manager
        if 'editing_mode' not in st.session_state:
            st.session_state.editing_mode = 'select'  # 'select', 'edit_product', 'new_product'

        # State for tracking the specific step being edited (None, 'new', or a step_id)
        if 'editing_step_id' not in st.session_state:
            st.session_state.editing_step_id = None
            
        # State for step reordering mode
        if 'reordering_mode' not in st.session_state:
            st.session_state.reordering_mode = False
            
        # Store the step order for each product
        if 'step_orders' not in st.session_state:
            st.session_state.step_orders = {}
    
    def render(self):
        """Main render method for the Product Manager UI."""
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            st.subheader("📦 Product Configuration")
            st.markdown("Define product categories and their sequential processing steps.")
        with col2:
            self._render_header_controls()
        st.markdown("---")
        
        # Main content area based on editing mode
        if st.session_state.editing_mode == 'select':
            self._render_product_selection()
        elif st.session_state.editing_mode == 'new_product':
            self._render_new_product_form()
        elif st.session_state.editing_mode == 'edit_product':
            self._render_product_editor()
    
    def _render_header_controls(self):
        """Render the header controls (reload button)."""
        if st.button("🔃 RELOAD", use_container_width=True, type='primary'):
            self._reload_products()
    
    def _render_product_selection(self):
        """Render the product selection interface."""
        st.markdown("### Product Management")
        
        available_products = list(st.session_state.product_config_display.keys())
        
        if available_products:
            with st.container(border=True):
                st.markdown("#### Select Product to Edit or Delete")
                selected_product = st.selectbox(
                    "Choose a product category:",
                    options=[""] + available_products,
                    key="product_selector",
                    index=0
                )
                
                if selected_product:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✏️ Edit Selected", type="secondary", use_container_width=True):
                            st.session_state.selected_product_for_editing = selected_product
                            st.session_state.editing_mode = 'edit_product'
                            st.session_state.editing_step_id = None # Reset step editing
                            st.session_state.reordering_mode = False # Reset reordering mode
                            self._initialize_step_order(selected_product)
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Delete Selected", type="secondary", use_container_width=True):
                            self._handle_delete_product(selected_product)
                else:
                    st.button("✏️ Edit Selected", type="secondary", use_container_width=True, disabled=True)
                    st.button("🗑️ Delete Selected", type="secondary", use_container_width=True, disabled=True)
            
            st.markdown("#### Product Overview")
            self._render_product_overview()
        else:
            st.info("No products configured yet. Create your first product below.")
        
        with st.container(border=True):
            st.markdown("#### Add New Product Category")
            new_product_category = st.text_input("New Product Category Name", key="new_prod_cat_name_input")
            if st.button("➕ Add New Product", type="primary", use_container_width=True):
                self._start_new_product(new_product_category)
    
    def _handle_delete_product(self, product_category: str):
        """Handle the deletion of a product with confirmation."""
        # Use a more specific key for confirmation state
        confirm_key = f'confirm_delete_product_{product_category}'
        
        # Initialize confirmation state if not exists
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False
        
        if not st.session_state[confirm_key]:
            # Show confirmation dialog
            st.warning(f"⚠️ Are you sure you want to delete the product category '{product_category}'?")
            st.markdown("This action will permanently remove:")
            
            product = st.session_state.product_config_display.get(product_category)
            if product:
                st.markdown(f"- **{len(product.processing_steps)}** processing steps")
                step_names = [step.name for step in product.processing_steps]
                if step_names:
                    st.markdown(f"- Steps: {', '.join(step_names)}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary", use_container_width=True, key=f"confirm_delete_{product_category}"):
                    self._delete_product(product_category)
                    # Clean up the confirmation state
                    if confirm_key in st.session_state:
                        del st.session_state[confirm_key]
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key=f"cancel_delete_{product_category}"):
                    # Clean up the confirmation state and return to selection
                    if confirm_key in st.session_state:
                        del st.session_state[confirm_key]
                    st.rerun()
    
    def _delete_product(self, product_category: str):
        """Actually delete the product from session state and save to file."""
        # Remove from session state
        del st.session_state.product_config_display[product_category]

        st.success(f"✅ Line '{[product_category]}' deleted successfully!")
        
        # Remove from step orders
        del st.session_state.step_orders[product_category]
        st.success(f"✅ Line '{product_category}' deleted successfully!")
        
        # Save changes to file
        self._save_all_products()
        
        st.success(f"Product '{product_category}' has been deleted successfully.")
    
    def _initialize_step_order(self, product_category: str):
        """Initialize the step order for a product based on current step_id sorting."""
        product = st.session_state.product_config_display.get(product_category)
        if product and product.processing_steps:
            # Sort by step_id to get consistent initial order
            sorted_steps = product.processing_steps
            st.session_state.step_orders[product_category] = [step.step_id for step in sorted_steps]
        else:
            st.session_state.step_orders[product_category] = []
    
    def _get_ordered_steps(self, product: Product) -> List[ProcessingStep]:
        """Get steps in the order defined by step_orders, falling back to step_id sort."""
        product_category = product.product_category
        
        # Initialize if not exists
        if product_category not in st.session_state.step_orders or not st.session_state.step_orders[product_category]:
            self._initialize_step_order(product_category)
        
        step_order = st.session_state.step_orders[product_category]
        step_dict = {step.step_id: step for step in product.processing_steps}
        
        # Return steps in the specified order, handling any missing steps
        ordered_steps = []
        for step_id in step_order:
            if step_id in step_dict:
                ordered_steps.append(step_dict[step_id])
        
        # Add any steps that aren't in the order list (newly added steps)
        for step in product.processing_steps:
            if step.step_id not in step_order:
                ordered_steps.append(step)
                st.session_state.step_orders[product_category].append(step.step_id)
        
        return ordered_steps
    
    def _render_product_overview(self):
        """Render a summary table of all products and their step counts."""
        overview_data = []
        for product_category, product in st.session_state.product_config_display.items():
            total_duration = sum(step.duration_minutes for step in product.processing_steps)
            resource_types = set()
            for step in product.processing_steps:
                for req in step.requirements:
                    resource_types.add(req.resource_type.value)

            overview_data.append({
                "Product Category": product_category,
                "Steps Count": len(product.processing_steps),
                "Resource Types": ", ".join(sorted(resource_types)) if resource_types else "None",
                "Total Duration (min)": total_duration
            })
        
        if overview_data:
            st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

    def _render_new_product_form(self):
        """Render the form for creating a new product category."""
        st.markdown("### Create New Product")
        if st.button("← Back", type="secondary"):
            st.session_state.editing_mode = 'select'
            st.rerun()

        with st.container(border=True):
            product_category = st.text_input(
                "Product Category Name:", 
                value=st.session_state.get('new_product_category', ''),
                key="new_product_category_input"
            )
            
            if st.button("Create and Configure Steps", type="primary", use_container_width=True):
                product_category_stripped = product_category.strip()
                if not product_category_stripped:
                    st.error("Product name cannot be empty.")
                elif product_category_stripped in st.session_state.product_config_display:
                    st.error(f"Product category '{product_category_stripped}' already exists.")
                else:
                    new_product = Product(product_category=product_category_stripped, processing_steps=[])
                    st.session_state.product_config_display[product_category_stripped] = new_product
                    st.session_state.selected_product_for_editing = product_category_stripped
                    st.session_state.editing_mode = 'edit_product'
                    st.session_state.editing_step_id = None
                    st.session_state.reordering_mode = False
                    self._initialize_step_order(product_category_stripped)
                    st.success(f"Product '{product_category_stripped}' created. You can now add steps.")
                    st.rerun()
    
    def _render_product_editor(self):
        """Render the main editor for a selected product, including step summary and forms."""
        product_category = st.session_state.selected_product_for_editing
        product = st.session_state.product_config_display.get(product_category)

        if not product:
            st.error("Product not found. Returning to selection.")
            st.session_state.editing_mode = 'select'
            st.rerun()

        # Header and Back Button
        if st.button("← Back to Product Selection", type="secondary"):
            st.session_state.editing_mode = 'select'
            st.session_state.selected_product_for_editing = None
            st.session_state.editing_step_id = None
            st.session_state.reordering_mode = False
            st.rerun()
        
        st.markdown(f"### Editing Product: **{product_category}**")
        
        # Display steps summary and the selection dropdown/button
        if st.session_state.reordering_mode:
            self._render_step_reordering_interface(product)
        else:
            self._render_steps_summary_and_selection(product)

            # Conditionally display the step edit form
            if st.session_state.editing_step_id:
                self._render_step_edit_form(product)

        # Main Save button to persist all changes to the file
        st.markdown("---")
        col1, col2 = st.columns([0.7, 0.3])
        with col2:
             if st.button("💾 Save All Changes to File", type="primary", use_container_width=True):
                self._save_all_products()
                st.success(f"All changes for product '{product_category}' saved to file.")

    def _render_steps_summary_and_selection(self, product: Product):
        """Displays a summary of steps and provides UI to select/add a step."""
        # Header with reorder button
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown("#### Current Processing Steps")
        with col2:
            if len(product.processing_steps) > 1:
                if st.button("🔄 Reorder", use_container_width=True):
                    st.session_state.reordering_mode = True
                    st.session_state.editing_step_id = None  # Exit step editing mode
                    st.rerun()
        
        summary_data = []
        ordered_steps = self._get_ordered_steps(product)

        for i, step in enumerate(ordered_steps):
            resource_types_str = " / ".join(sorted(list(set(req.resource_type.value for req in step.requirements))))
            summary_data.append({
                "Order": i + 1,
                "Step ID": step.step_id,
                "Step Name": step.name,
                'Process Type': step.process_type, 
                "Resource Types": resource_types_str if resource_types_str else "None",
                "Duration (min)": step.duration_minutes,
                "Min Capacity (L)": step.min_capacity_required_liters,
                "CIP After?": "✔️" if step.requires_CIP_after else "❌",
            })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        else:
            st.info("This product has no processing steps. Add a new step below.")

        st.markdown("---")
        
        # Step selection UI
        step_options = {step.step_id: f"{step.step_id} - {step.name}" for step in ordered_steps}
        options = ["➕ Add New Step"] + list(step_options.keys())
        
        selected_step_id = st.selectbox(
            "Select a step to edit, or add a new one:",
            options=options,
            format_func=lambda x: "➕ Add New Step" if x == "➕ Add New Step" else step_options.get(x, x),
            key=f"step_selector_{product.product_category}",
            index=0 
        )

        if st.button("Configure Selected Step", use_container_width=True):
            if selected_step_id == "➕ Add New Step":
                st.session_state.editing_step_id = 'new'
            else:
                st.session_state.editing_step_id = selected_step_id
            st.rerun()

    def _render_step_reordering_interface(self, product: Product):
        """Renders the interface for reordering processing steps using up/down buttons."""
        st.markdown("#### 🔄 Reorder Processing Steps")
        st.markdown("*Use the ⬆️ and ⬇️ buttons to move steps up or down in the sequence*")
        
        ordered_steps = self._get_ordered_steps(product)
        product_category = product.product_category
        
        if not ordered_steps:
            st.info("No steps to reorder.")
            if st.button("← Back to Step Management", use_container_width=True):
                st.session_state.reordering_mode = False
                st.rerun()
            return
        
        # Display steps with move buttons
        for i, step in enumerate(ordered_steps):
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([0.1, 0.1, 0.6, 0.2])
                
                with col1:
                    # Move Up button
                    if i > 0:
                        if st.button("⬆️", key=f"up_{step.step_id}", help="Move up"):
                            self._move_step_up(product_category, i)
                            st.rerun()
                    else:
                        st.write("")  # Empty space for alignment
                
                with col2:
                    # Move Down button
                    if i < len(ordered_steps) - 1:
                        if st.button("⬇️", key=f"down_{step.step_id}", help="Move down"):
                            self._move_step_down(product_category, i)
                            st.rerun()
                    else:
                        st.write("")  # Empty space for alignment
                
                with col3:
                    # Step info
                    resource_types_str = " / ".join(sorted(list(set(req.resource_type.value for req in step.requirements))))
                    st.markdown(f"**{i+1}. {step.step_id}** - {step.name}")
                    st.caption(f"Resources: {resource_types_str if resource_types_str else 'None'} | Duration: {step.duration_minutes} min")
                
                with col4:
                    st.markdown(f"**Order: {i+1}**")
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Done Reordering", type="primary", use_container_width=True):
                st.session_state.reordering_mode = False
                st.success("Step order updated! Remember to save changes to file.")
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                # Reset to original order (sorted by step_id)
                self._initialize_step_order(product_category)
                st.session_state.reordering_mode = False
                st.rerun()

    def _move_step_up(self, product_category: str, step_index: int):
        """Move a step up in the order."""
        if step_index > 0:
            step_order = st.session_state.step_orders[product_category]
            step_order[step_index], step_order[step_index - 1] = step_order[step_index - 1], step_order[step_index]

    def _move_step_down(self, product_category: str, step_index: int):
        """Move a step down in the order."""
        step_order = st.session_state.step_orders[product_category]
        if step_index < len(step_order) - 1:
            step_order[step_index], step_order[step_index + 1] = step_order[step_index + 1], step_order[step_index]

    def _render_step_edit_form(self, product: Product):
        """Renders the detailed form for adding or editing a single processing step."""
        step_id_to_edit = st.session_state.editing_step_id
        is_new_step = step_id_to_edit == 'new'
        
        # Find existing step data or prepare for a new one
        if is_new_step:
            form_title = "Adding New Step"
            # Default data for a new step
            step_data = {
                'step_id': '', 'name': '', 'duration_minutes': 0.0, 'process_type': ProcessType.PROCESSING,
                'min_capacity_required_liters': 0.0, 'requires_CIP_after': True,
                'requirements': [{'Resource Type': ResourceType.EQUIPMENT.value, 'Compatible Resource IDs': ''}]
            }
        else:
            form_title = f"Editing Step: **{step_id_to_edit}**"
            step_to_edit = next((s for s in product.processing_steps if s.step_id == step_id_to_edit), None)
            if not step_to_edit:
                st.error("Step not found.")
                st.session_state.editing_step_id = None
                return

            req_list = [{'Resource Type': req.resource_type.value, 'Compatible Resource IDs': ','.join(req.compatible_ids)} for req in step_to_edit.requirements]
            if not req_list:
                 req_list.append({'Resource Type': ResourceType.EQUIPMENT.value, 'Compatible Resource IDs': ''})
            
            step_data = step_to_edit.__dict__.copy()
            step_data['requirements'] = req_list

        with st.container(border=True):
            st.markdown(f"#### {form_title}")
            
            with st.form(key=f"step_form_{step_id_to_edit}"):
                c1, c2 = st.columns(2)
                with c1:
                    step_id = st.text_input("Step ID*", value=step_data['step_id'], disabled=not is_new_step)
                    duration = st.number_input("Duration (min/batch)", value=float(step_data['duration_minutes']), min_value=0.0)
                    process_type = st.selectbox('Process Type', options= [t.value for t in ProcessType])
                with c2:
                    name = st.text_input("Step Name", value=step_data['name'])
                    capacity = st.number_input("Min Capacity (Liters)", value=float(step_data['min_capacity_required_liters']), min_value=0.0)
                    cip = st.checkbox("Requires CIP After", value=bool(step_data['requires_CIP_after']))

                st.markdown("##### Resource Requirements*")
                requirements_df = pd.DataFrame(step_data['requirements'])
                edited_req_df = st.data_editor(
                    requirements_df, num_rows="dynamic", use_container_width=True,
                    column_config={
                        "Resource Type": st.column_config.SelectboxColumn("Resource Type", options=[rt.value for rt in ResourceType], required=True),
                        "Compatible Resource IDs": st.column_config.TextColumn("Compatible IDs (comma-separated)")
                    }
                )

                # Form submission buttons
                submit_c1, submit_c2, submit_c3 = st.columns(3)
                with submit_c1:
                    submitted = st.form_submit_button("✅ Save Step", type="primary", use_container_width=True)
                with submit_c2:
                    if not is_new_step:
                        if st.form_submit_button("🗑️ Delete Step", use_container_width=True):
                            self._handle_delete_step(product, step_id_to_edit)
                with submit_c3:
                    if st.form_submit_button("❌ Cancel", use_container_width=True):
                        st.session_state.editing_step_id = None
                        st.rerun()

            if submitted:
                form_data = {
                    "step_id": step_id, "name": name, "duration_minutes": duration, 'process_type': process_type,
                    "min_capacity_required_liters": capacity, "requires_CIP_after": cip,
                    "requirements_df": edited_req_df
                }
                self._handle_save_step(product, step_id_to_edit, form_data)

    def _handle_save_step(self, product: Product, original_step_id: str, form_data: dict):
        """Validates and saves the data from the step form."""
        new_step_id = form_data['step_id'].strip()
        is_new_step = original_step_id == 'new'

        # --- Validation ---
        if not new_step_id:
            st.error("Step ID cannot be empty.")
            return

        if is_new_step and any(s.step_id == new_step_id for s in product.processing_steps):
            st.error(f"Step ID '{new_step_id}' already exists in this product.")
            return

        requirements = []
        for _, row in form_data['requirements_df'].iterrows():
            if pd.isna(row['Resource Type']) or not row['Resource Type']:
                continue
            
            res_type = ResourceType(row['Resource Type'])
            comp_ids_str = row['Compatible Resource IDs']
            comp_ids = [s.strip() for s in str(comp_ids_str).split(',') if s.strip()] if pd.notna(comp_ids_str) else []
            requirements.append(ResourceRequirement(resource_type=res_type, compatible_ids=comp_ids))

        if not requirements:
            st.error("At least one valid resource requirement must be specified.")
            return
            
        # --- Persistence (in-memory) ---
        new_step = ProcessingStep(
            step_id=new_step_id, name=form_data['name'],
            duration_minutes=form_data['duration_minutes'],
            min_capacity_required_liters=form_data['min_capacity_required_liters'],
            requires_CIP_after=form_data['requires_CIP_after'],
            process_type= form_data['process_type'],
            requirements=requirements
        )

        if is_new_step:
            product.processing_steps.append(new_step)
            # Add to step order
            product_category = product.product_category
            if product_category not in st.session_state.step_orders:
                st.session_state.step_orders[product_category] = []
            st.session_state.step_orders[product_category].append(new_step_id)
        else:
            step_index = next((i for i, s in enumerate(product.processing_steps) if s.step_id == original_step_id), -1)
            if step_index != -1:
                product.processing_steps[step_index] = new_step
                # Update step order if step_id changed
                if original_step_id != new_step_id:
                    product_category = product.product_category
                    step_order = st.session_state.step_orders.get(product_category, [])
                    if original_step_id in step_order:
                        idx = step_order.index(original_step_id)
                        step_order[idx] = new_step_id
        
        st.success(f"Step '{new_step_id}' saved to memory. Press 'Save All Changes to File' to persist.")
        st.session_state.editing_step_id = None
        st.rerun()

    def _handle_delete_step(self, product: Product, step_id_to_delete: str):
        """Deletes a step from the product's list of steps."""
        product.processing_steps = [s for s in product.processing_steps if s.step_id != step_id_to_delete]
        
        # Remove from step order
        product_category = product.product_category
        if product_category in st.session_state.step_orders:
            step_order = st.session_state.step_orders[product_category]
            if step_id_to_delete in step_order:
                step_order.remove(step_id_to_delete)
        
        st.success(f"Step '{step_id_to_delete}' deleted from memory. Press 'Save All Changes' to persist.")
        st.session_state.editing_step_id = None
        st.rerun()

    def _start_new_product(self, product_category: str):
        """Initiates the new product creation workflow."""
        product_category = product_category.strip()
        if not product_category:
            st.warning("Please enter a product category name.")
            return
        
        if product_category in st.session_state.product_config_display:
            st.warning(f"Product category '{product_category}' already exists.")
            return
        
        st.session_state.new_product_category = product_category
        st.session_state.editing_mode = 'new_product'
        st.rerun()
    
    def _save_all_products(self):
        """Saves the current state of all products from session state to the CSV file."""
        # Update the processing_steps order for each product based on step_orders
        for product_category, product in st.session_state.product_config_display.items():
            if product_category in st.session_state.step_orders:
                ordered_steps = self._get_ordered_steps(product)
                product.processing_steps = ordered_steps
        
        all_rows = []
        for product in st.session_state.product_config_display.values():
            if not product.processing_steps:
                # Empty product - create one row with empty step data
                all_rows.append({
                    "Product_Category": product.product_category,
                    "Step_ID": None,
                    "Step_Name": None,
                    "Resource_Type": None,
                    "Duration_Minutes_Per_Batch": None,
                    "Min_Capacity_Required_Liters": None,
                    "Compatible_Resource_IDs": None,
                    "Requires_Setup": None,
                    "Requires_CIP_After": None,
                    "Max_Batch_Size": product.max_batch_size
                })
            else:
                # For each step, create one row per resource requirement
                for step in product.processing_steps:
                    if not step.requirements:
                        # Step with no requirements
                        all_rows.append({
                            "Product_Category": product.product_category,
                            "Step_ID": step.step_id,
                            "Step_Name": step.name,
                            "Process_Type": step.process_type,
                            "Resource_Type": None,
                            "Duration_Minutes_Per_Batch": step.duration_minutes,
                            "Min_Capacity_Required_Liters": step.min_capacity_required_liters,
                            "Compatible_Resource_IDs": None,
                            "Requires_Setup": False,  # Default value
                            "Requires_CIP_After": step.requires_CIP_after,
                            "Max_Batch_Size": product.max_batch_size
                        })
                    else:
                        # One row per resource requirement
                        for req in step.requirements:
                            all_rows.append({
                                "Product_Category": product.product_category,
                                "Step_ID": step.step_id,
                                "Step_Name": step.name,
                                "Resource_Type": req.resource_type.value,
                                "Process_Type": step.process_type,
                                "Duration_Minutes_Per_Batch": step.duration_minutes,
                                "Min_Capacity_Required_Liters": step.min_capacity_required_liters,
                                "Compatible_Resource_IDs": ",".join(req.compatible_ids),
                                "Requires_Setup": False,  # Default value - you may want to add this to ResourceRequirement
                                "Requires_CIP_After": step.requires_CIP_after,
                                "Max_Batch_Size": product.max_batch_size
                            })
        
        if not all_rows:
            df_to_save = pd.DataFrame(columns=[
                "Product_Category", "Step_ID", "Step_Name", "Process_Type","Resource_Type",
                "Duration_Minutes_Per_Batch", "Min_Capacity_Required_Liters",
                "Compatible_Resource_IDs", "Requires_Setup", "Requires_CIP_After",
                "Max_Batch_Size"
            ])
        else:
            df_to_save = pd.DataFrame(all_rows)
        
        file_path = self.data_loader.data_dir / "product_config.csv"
        df_to_save.to_csv(file_path, index=False)
        self._reload_products(show_success=False)

    def _reload_products(self, show_success=True):
        """Reloads products from the data source into config and session state."""
        config.PRODUCTS.clear()
        config.PRODUCTS.update(self.data_loader.load_products_with_fallback())
        st.session_state.product_config_display = {
            p.product_category: p for p in config.PRODUCTS.values()
        }
        # Reinitialize step orders
        for product_category in st.session_state.product_config_display.keys():
            self._initialize_step_order(product_category)
            
        if show_success:
            st.success("Product configurations reloaded from file.")
        st.session_state.editing_mode = 'select'
        st.session_state.selected_product_for_editing = None
        st.session_state.editing_step_id = None
        st.session_state.reordering_mode = False
        st.rerun()

def render_product_manager(data_loader: DataLoader):
    """Main function to render the Product Manager UI."""
    product_manager = ProductManagerUI(data_loader)
    product_manager.render()