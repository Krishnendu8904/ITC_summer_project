import streamlit as st
import pandas as pd
import config
from models.data_models import Product, ProcessingStep, ResourceType # Import Product, ProcessingStep, and ResourceType
from data_loader import DataLoader
import numpy as np # Import numpy to handle NaN values from data editor
from typing import List, Dict, Any

def render_product_manager(data_loader: DataLoader):
    """
    Renders the Product configuration UI for editing and saving Products and their processing steps.
    Each product is displayed as a subsection with its steps in a data editor.
    """
    st.subheader("📦 Product Configuration")
    st.markdown("Define product categories and their sequential processing steps.")

    # Initialize a session state for products, which will be the source for UI display
    # This state will be updated by adding/deleting products from the UI
    if 'product_config_display' not in st.session_state:
        st.session_state.product_config_display = {
            product.product_category: product for product in config.PRODUCTS.values()
        }

    # Section to add new products
    with st.container(border=True):
        st.markdown("#### Add New Product Category")
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            new_product_category_name = st.text_input("New Product Category Name", key="new_prod_cat_name_input")
        with col2:
            st.markdown("##") # Space for alignment
            if st.button("➕ Add New Product", type="secondary", use_container_width=True):
                new_cat_name_stripped = new_product_category_name.strip()
                if new_cat_name_stripped and new_cat_name_stripped not in st.session_state.product_config_display:
                    st.session_state.product_config_display[new_cat_name_stripped] = Product(product_category=new_cat_name_stripped, processing_steps=[])
                    st.success(f"Product '{new_cat_name_stripped}' added. You can now define its steps below.")
                    st.rerun() # Rerun to display the new product expander
                elif new_cat_name_stripped and new_cat_name_stripped in st.session_state.product_config_display:
                    st.warning(f"Product category '{new_cat_name_stripped}' already exists.")
                else:
                    st.warning("Please enter a name for the new product category.")

    st.markdown("---")

    # List to collect product categories marked for deletion
    products_to_delete = []

    # Display and edit existing products
    # Sort keys to maintain consistent order in UI
    product_categories_sorted = sorted(st.session_state.product_config_display.keys())

    for product_category in product_categories_sorted:
        product = st.session_state.product_config_display[product_category]

        # Use an expander for each product to group its steps
        with st.expander(f"**Product Category: {product.product_category}**", expanded=True):
            st.markdown(f"**Define Processing Steps for '{product.product_category}':**")
            
            # Prepare steps for data_editor
            current_steps_dicts = product._to_dicts()
            
            # If a product has no actual steps, _to_dicts returns a single dict with None for Step_ID.
            # Convert this to an empty DataFrame for the editor to display as empty.
            if len(current_steps_dicts) == 1 and current_steps_dicts[0].get("Step_ID") is None:
                df_steps_display = pd.DataFrame(columns=[
                    "Step_ID", "Step_Name", "Resource_Type", "Duration_Minutes_Per_Batch",
                    "Min_Capacity_Required_Liters", "Compatible_Resource_IDs", "Requires_Setup", "Requires_CIP_After"
                ])
            else:
                # Create DataFrame, dropping "Product_Category" as it's implicit for this editor
                df_steps_display = pd.DataFrame(current_steps_dicts).drop(columns=["Product_Category"])
            
            # Get available ResourceType values for selectbox column
            resource_type_options = [rt.value for rt in ResourceType]

            # Data editor for the steps of the current product
            edited_steps_df = st.data_editor(
                df_steps_display,
                num_rows="dynamic", # Allow adding/deleting rows within this editor
                key=f"steps_editor_{product.product_category}", # Unique key for each product's editor
                use_container_width=True,
                column_config={
                    "Step_ID": st.column_config.TextColumn("Step ID (Unique per Product)", required=True),
                    "Step_Name": st.column_config.TextColumn("Step Name"),
                    "Resource_Type": st.column_config.SelectboxColumn("Required Resource Type", options=resource_type_options, required=True),
                    "Duration_Minutes_Per_Batch": st.column_config.NumberColumn("Duration (Min/Batch)", min_value=0.0),
                    "Min_Capacity_Required_Liters": st.column_config.NumberColumn("Min Capacity Req (Liters)", min_value=0.0),
                    "Compatible_Resource_IDs": st.column_config.TextColumn("Compatible Resource IDs (comma-separated)"),
                    "Requires_Setup": st.column_config.CheckboxColumn("Requires Setup?"),
                    "Requires_CIP_After": st.column_config.CheckboxColumn("Requires CIP After?"),
                }
            )
            # Store the edited DataFrame for this product's steps in session state for later saving
            st.session_state[f"edited_steps_df_{product.product_category}"] = edited_steps_df

            st.markdown("---")
            # Button to delete the entire product
            col_prod_actions_1, col_prod_actions_2 = st.columns([0.8, 0.2])
            with col_prod_actions_2:
                if st.button(f"🗑️ Delete Product: {product.product_category}", key=f"delete_prod_{product.product_category}", type="secondary", use_container_width=True):
                    products_to_delete.append(product.product_category)
            st.markdown("<br>", unsafe_allow_html=True) # Add some space

    # Process deletions after iterating to avoid modifying dict during iteration
    for prod_cat_to_delete in products_to_delete:
        if prod_cat_to_delete in st.session_state.product_config_display:
            del st.session_state.product_config_display[prod_cat_to_delete]
            # Also clear its edited_steps_df from session state
            if f"edited_steps_df_{prod_cat_to_delete}" in st.session_state:
                del st.session_state[f"edited_steps_df_{prod_cat_to_delete}"]
            st.success(f"Product '{prod_cat_to_delete}' deleted.")
    if products_to_delete:
        st.rerun() # Rerun to remove deleted products from display

    st.markdown("---")
    if st.button("💾 Save All Products to CSV", use_container_width=True, type="primary"):
        with st.spinner("Saving Products..."):
            try:
                updated_products_to_save: Dict[str, Product] = {}

                for product_category_key in product_categories_sorted:
                    # Only process products that are still in the display state (not deleted by user)
                    if product_category_key in st.session_state.product_config_display:
                        # Retrieve the edited steps DataFrame for this product
                        edited_steps_df = st.session_state.get(f"edited_steps_df_{product_category_key}")

                        new_processing_steps_for_product: List[ProcessingStep] = []
                        product_step_id_tracker = set() # To check unique step IDs within this product

                        if edited_steps_df is not None:
                            for index, row in edited_steps_df.iterrows():
                                step_id = str(row["Step_ID"]).strip() if pd.notna(row["Step_ID"]) else None
                                
                                # Check if the row is effectively empty (all relevant step fields are empty or default)
                                # This handles rows added by "dynamic" editor that are not filled out
                                is_empty_step_row = (
                                    (step_id is None or step_id == "") and
                                    (pd.isna(row["Step_Name"]) or str(row["Step_Name"]).strip() == "") and
                                    (pd.isna(row["Resource_Type"]) or str(row["Resource_Type"]).strip() == "") and
                                    (pd.isna(row["Duration_Minutes_Per_Batch"]) or row["Duration_Minutes_Per_Batch"] == 0) and
                                    (pd.isna(row["Min_Capacity_Required_Liters"]) or row["Min_Capacity_Required_Liters"] == 0) and
                                    (pd.isna(row["Compatible_Resource_IDs"]) or str(row["Compatible_Resource_IDs"]).strip() == "") and
                                    (pd.isna(row["Requires_Setup"]) or not bool(row["Requires_Setup"])) and # False is default for checkbox
                                    (pd.isna(row["Requires_CIP_After"]) or not bool(row["Requires_CIP_After"])) # False is default for checkbox
                                )

                                if not is_empty_step_row:
                                    if not step_id:
                                        st.error(f"Product '{product_category_key}', Row {index+1}: Step ID cannot be empty. Please correct and save again.")
                                        return
                                    if step_id in product_step_id_tracker:
                                        st.error(f"Product '{product_category_key}', Row {index+1}: Duplicate Step ID '{step_id}'. Step IDs must be unique per product. Please correct and save again.")
                                        return
                                    product_step_id_tracker.add(step_id)

                                    try:
                                        resource_type_val = row["Resource_Type"]
                                        if pd.isna(resource_type_val) or str(resource_type_val).strip() == "":
                                            st.error(f"Product '{product_category_key}', Row {index+1}: Resource Type cannot be empty. Please select a valid type.")
                                            return
                                        resource_type_enum = ResourceType(resource_type_val)
                                    except ValueError:
                                        st.error(f"Product '{product_category_key}', Row {index+1}: Invalid Resource Type '{row['Resource_Type']}'. Please select a valid type.")
                                        return
                                    
                                    compatible_resource_ids = [
                                        res.strip() for res in str(row["Compatible_Resource_IDs"]).split(',') 
                                        if pd.notna(row["Compatible_Resource_IDs"]) and res.strip()
                                    ]

                                    new_processing_steps_for_product.append(ProcessingStep(
                                        step_id=step_id,
                                        name=str(row["Step_Name"]) if pd.notna(row["Step_Name"]) else "",
                                        resource_type=resource_type_enum,
                                        duration_minutes_per_batch=float(row["Duration_Minutes_Per_Batch"]) if pd.notna(row["Duration_Minutes_Per_Batch"]) else 0.0,
                                        min_capacity_required_liters=float(row["Min_Capacity_Required_Liters"]) if pd.notna(row["Min_Capacity_Required_Liters"]) else 0.0,
                                        compatible_resource_ids=compatible_resource_ids,
                                        requires_setup=bool(row["Requires_Setup"]) if pd.notna(row["Requires_Setup"]) else False,
                                        requires_cip_after=bool(row["Requires_CIP_After"]) if pd.notna(row["Requires_CIP_After"]) else False,
                                    ))
                        
                        # Add product to the save list with its (potentially updated) steps
                        updated_products_to_save[product_category_key] = Product(
                            product_category=product_category_key,
                            processing_steps=new_processing_steps_for_product
                        )
                
                # Convert updated_products_to_save into the flat DataFrame format for CSV
                df_to_save_list = []
                for product in updated_products_to_save.values():
                    df_to_save_list.extend(product._to_dicts())
                
                if not df_to_save_list:
                    # If all products were deleted or none were added, create an empty DataFrame with correct columns
                    df_to_save = pd.DataFrame(columns=[
                        "Product_Category", "Step_ID", "Step_Name", "Resource_Type",
                        "Duration_Minutes_Per_Batch", "Min_Capacity_Required_Liters",
                        "Compatible_Resource_IDs", "Requires_Setup", "Requires_CIP_After"
                    ])
                else:
                    df_to_save = pd.DataFrame(df_to_save_list)
                    # Ensure all expected columns are present, even if some products/steps don't use them
                    final_columns = [
                        "Product_Category", "Step_ID", "Step_Name", "Resource_Type",
                        "Duration_Minutes_Per_Batch", "Min_Capacity_Required_Liters",
                        "Compatible_Resource_IDs", "Requires_Setup", "Requires_CIP_After"
                    ]
                    for col in final_columns:
                        if col not in df_to_save.columns:
                            df_to_save[col] = None
                    df_to_save = df_to_save[final_columns]
                
                # Get file path and save
                file_path = data_loader.data_dir / "product_config.csv"
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ Product configuration saved successfully!")

                # Reload data to reflect changes
                config.PRODUCTS = data_loader.load_products_with_fallback()
                # Update the display state from the reloaded config
                st.session_state.product_config_display = {
                    product.product_category: product for product in config.PRODUCTS.values()
                }
                st.rerun() # Rerun to refresh the UI with saved data

            except Exception as e:
                st.error(f"❌ Error saving Product data: {e}")
                st.exception(e) # Display full traceback for debugging