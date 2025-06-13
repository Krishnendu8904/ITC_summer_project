import streamlit as st
import pandas as pd
import config
from models.data_models import SKU
from data_loader import DataLoader

def render_sku_manager(data_loader: DataLoader):
    """
    Renders the SKU configuration UI for editing and saving SKUs.
    """
    st.subheader("🥛 SKU Configuration")
    st.markdown("Define product SKUs, their properties, and associated production parameters.")

    # Use session state for interactive editing
    if 'sku_config' not in st.session_state:
        st.session_state.sku_config = config.SKUS.copy()

    try:
        df_display = pd.DataFrame([sku._to_dict() for sku in st.session_state.sku_config.values()])
    except Exception as e:
        st.error(f"Error preparing SKU data for display: {e}")
        st.error("Please ensure the data models and CSV headers are consistent.")
        return

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        key="sku_editor",
        use_container_width=True,
        column_config={
            "SKU_ID": st.column_config.TextColumn("SKU ID", required=True),
            "Product_Category": st.column_config.TextColumn("Product Category", required=True),
            "Variant": st.column_config.TextColumn("Variant"),
            "Inventory_Size": st.column_config.NumberColumn("Inventory Size (L/Unit)", required=True, min_value=0.01),
            "Setup_Time": st.column_config.NumberColumn("Setup Time (Min)", required=True, min_value=0),
            "Inventory_Units": st.column_config.NumberColumn("EIU (Units/kg)", required=True, min_value=0.01)
        }
    )

    if st.button("💾 Save SKUs to CSV", use_container_width=True, type="primary"):
        with st.spinner("Saving SKUs..."):
            try:
                # Convert edited DF back to a dictionary of SKU objects for validation
                updated_skus = {}
                for _, row in edited_df.iterrows():
                    sku_id = row["SKU_ID"]
                    if not sku_id:
                        st.error("SKU ID cannot be empty. Please correct and save again.")
                        return
                    if sku_id in updated_skus:
                        st.error(f"Duplicate SKU ID '{sku_id}' found. IDs must be unique.")
                        return
                    
                    updated_skus[sku_id] = SKU(
                        sku_id=str(row["SKU_ID"]),
                        product_category=str(row["Product_Category"]),
                        variant=str(row["Variant"]),
                        inventory_size=float(row["Inventory_Size"]),
                    )

                # Get file path and save
                file_path = data_loader.data_dir / "sku_config.csv"
                df_to_save = pd.DataFrame([sku._to_dict() for sku in updated_skus.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ SKU configuration saved successfully!")

                # Reload data to reflect changes
                config.SKUS = data_loader.load_skus_with_fallback()
                st.session_state.sku_config = config.SKUS.copy()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving SKU data: {e}")