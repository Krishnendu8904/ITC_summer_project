import streamlit as st
import pandas as pd
from datetime import datetime

def render():
    """
    Renders the UI for the Procurement Team to log incoming raw materials.
    """
    st.markdown('<h2 class="section-header">Procurement & Raw Material Intake</h2>', unsafe_allow_html=True)

    # Initialize session state for storing procurement data if it doesn't exist
    if 'procurement_data' not in st.session_state:
        st.session_state.procurement_data = []

    # --- Main Layout: Two Columns ---
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📝 Log New Shipment</h3>', unsafe_allow_html=True)
        
        with st.form("new_shipment_form"):
            st.text_input("Truck Number", key="truck_no")
            
            st.markdown("---")
            
            # Mixed Milk Inputs
            st.write("**Mixed Milk Details**")
            mm_qty = st.number_input("Mixed Milk (L)", min_value=0.0, step=100.0, key="mm_qty")
            mm_fat = st.number_input("Fat %", min_value=0.0, max_value=100.0, step=0.1, key="mm_fat", format="%.2f")
            mm_snf = st.number_input("SNF %", min_value=0.0, max_value=100.0, step=0.1, key="mm_snf", format="%.2f")
            
            st.markdown("---")

            # Cow Milk Inputs
            st.write("**Cow Milk Details**")
            cm_qty = st.number_input("Cow Milk (L)", min_value=0.0, step=100.0, key="cm_qty")
            cm_fat = st.number_input("Fat % ", min_value=0.0, max_value=100.0, step=0.1, key="cm_fat", format="%.2f") # Space to avoid key conflict
            cm_snf = st.number_input("SNF % ", min_value=0.0, max_value=100.0, step=0.1, key="cm_snf", format="%.2f") # Space to avoid key conflict

            submitted = st.form_submit_button("📥 Log Shipment", use_container_width=True)

            if submitted:
                # Basic validation
                if not st.session_state.truck_no:
                    st.warning("Please enter a Truck Number.")
                elif mm_qty == 0 and cm_qty == 0:
                    st.warning("Please enter a quantity for at least one milk type.")
                else:
                    # Append new data to session state
                    new_entry = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Truck Number": st.session_state.truck_no,
                        "Mixed Milk (L)": mm_qty,
                        "MM Fat %": mm_fat,
                        "MM SNF %": mm_snf,
                        "Cow Milk (L)": cm_qty,
                        "CM Fat %": cm_fat,
                        "CM SNF %": cm_snf,
                    }
                    st.session_state.procurement_data.append(new_entry)
                    st.success(f"✅ Shipment from Truck '{st.session_state.truck_no}' logged successfully!")
                    # Note: In a real app, this would also save to a persistent database or file.

    with col2:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📋 Recent Shipments Log</h3>', unsafe_allow_html=True)
        
        if st.session_state.procurement_data:
            # Display data in descending order (most recent first)
            log_df = pd.DataFrame(st.session_state.procurement_data).iloc[::-1]
            
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No shipments have been logged yet.")

