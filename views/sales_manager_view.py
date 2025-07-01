import streamlit as st
import datetime
import pandas as pd
import numpy as np
import time 

# --- Local Imports ---
import config
from utils.data_models import UserIndent, Priority
from utils.data_loader import DataLoader

def save_indents_to_csv(data_loader, indents_to_save: dict):
    """
    Saves the current state of the in-memory indents dictionary to user_indent.csv,
    overwriting the file completely.
    """
    try:
        file_path = data_loader.data_dir / "user_indent.csv"
        
        if not indents_to_save:
            # If there are no indents, save an empty file with headers
            df_to_save = pd.DataFrame(columns=['Order_Number', 'SKU_ID', 'Qty_Required_Liters', 'Priority', 'Due_Date'])
        else:
            # Convert the dictionary of UserIndent objects to a list of dicts
            new_indents_data = [indent._to_dict() for indent in indents_to_save.values()]
            df_to_save = pd.DataFrame(new_indents_data)

        # Overwrite the existing file
        df_to_save.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"❌ Error saving indents to CSV: {e}")
        return False

def render():
    """
    Renders the UI for the Sales Manager role, including create and delete functionality.
    """
    st.markdown('<h2 class="section-header">Sales & Order Management</h2>', unsafe_allow_html=True)

    # --- Main Layout: Two Columns ---
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📝 Create New Indent</h3>', unsafe_allow_html=True)
        
        with st.form("new_indent_form"):
            if not config.SKUS:
                dataloader = DataLoader()
                config.SKUS.update(dataloader.load_skus_with_fallback())
            available_skus = list(config.SKUS.keys()) if config.SKUS else []
            
            # Form fields
            sku_id = st.selectbox("Select SKU", options=available_skus, disabled=not available_skus, key="new_order_sku")
            qty = st.number_input("Quantity (Liters)", min_value=100, value=1000, step=100)
            
            col_date, col_time = st.columns(2)
            with col_date:
                due_date = st.date_input("Due Date", value=datetime.datetime.now().date() + datetime.timedelta(days=3))
            with col_time:
                due_time = st.time_input("Due Time", value=datetime.time(17, 0))
            
            priority = st.selectbox("Priority", options=[p.name for p in Priority], index=1)
            
            submitted = st.form_submit_button("➕ Submit Production Indent", use_container_width=True)

            if submitted:
                if not available_skus:
                    st.error("Cannot submit indent. No SKUs are configured in the system.")
                else:
                    # --- Creative Order Name Generation ---
                    order_no = f"{sku_id}-{due_date.strftime('%d')}-{due_date.strftime('%m')}-{due_date.strftime('%y')}"
                    
                    full_due_date = datetime.datetime.combine(due_date, due_time)
                    new_indent = UserIndent(
                        order_no=order_no,
                        sku_id=sku_id,
                        qty_required_liters=float(qty),
                        due_date=full_due_date,
                        priority=Priority[priority]
                    )
                    
                    config.USER_INDENTS[order_no] = new_indent
                    
                    if save_indents_to_csv(st.session_state.data_loader, config.USER_INDENTS):
                        st.success(f"✅ Indent '{order_no}' submitted successfully!")
                    st.rerun()

    with col2:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📋 Live Production Indents</h3>', unsafe_allow_html=True)
        
        if config.USER_INDENTS:
            sorted_indents = sorted(config.USER_INDENTS.values(), key=lambda x: x.due_date)
            indent_data = [{
                "Order Number": i.order_no, "SKU ID": i.sku_id, "Quantity (L)": i.qty_required_liters,
                "Due Date": i.due_date.strftime("%Y-%m-%d %H:%M"), "Priority": i.priority.name
            } for i in sorted_indents]
            
            indents_df = pd.DataFrame(indent_data)
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(indents_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Delete Order Functionality ---
            st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:1.5rem;">🗑️ Delete an Indent</h3>', unsafe_allow_html=True)
            order_to_delete = st.selectbox("Select an indent to delete", options=[""] + list(config.USER_INDENTS.keys()))

            if st.button("❌ Delete Selected Indent", use_container_width=True, disabled=not order_to_delete):
                if order_to_delete in config.USER_INDENTS:
                    del config.USER_INDENTS[order_to_delete]
                    if save_indents_to_csv(st.session_state.data_loader, config.USER_INDENTS):
                        st.success(f"🗑️ Indent '{order_to_delete}' deleted successfully!")
                    st.rerun()

        else:
            st.info("No active indents. Create a new indent to get started.")
            
        # Placeholder for future Stockout Analysis feature
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:1.5rem;">🚨 Stockout Analysis (Coming Soon)</h3>', unsafe_allow_html=True)
        st.info("An interface to check for potential stockouts will be available here.")

