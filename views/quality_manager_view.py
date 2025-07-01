import streamlit as st

# --- Local Imports ---
from managers.gantt_chart import create_production_gantt
import config

def render():
    """
    Renders the UI for the Quality Manager role.
    For now, it's a placeholder but is structured to hold future features.
    """
    st.markdown('<h2 class="section-header">Quality Assurance Dashboard</h2>', unsafe_allow_html=True)

    # --- Information Box ---
    st.markdown("""
    <div class="info-box" style="border-left-color: #F39C12;">
    This dashboard will provide tools for monitoring production quality at various stages. 
    Key features are currently under development.
    </div>
    """, unsafe_allow_html=True)

    # --- Future Feature: Gantt Chart View ---
    st.markdown('<h3 class="section-header" style="font-size:1.4rem;">Production Schedule Overview</h3>', unsafe_allow_html=True)

    result = st.session_state.get('last_schedule_result')

    if result:
        st.info("Displaying the latest generated production schedule for quality monitoring points.")
        prod_gantt = create_production_gantt(result)
        if prod_gantt:
            st.plotly_chart(prod_gantt, use_container_width=True)
        else:
            st.warning("Could not generate the Production Gantt chart from the latest schedule.")
    else:
        st.warning("No production schedule has been generated yet. The Gantt chart will appear here once a schedule is created by the Production Manager.")

    # --- Other Placeholder Sections ---
    st.markdown("---")
    st.markdown('<h3 class="section-header" style="font-size:1.4rem;">Quality Control Checks (Under Construction)</h3>', unsafe_allow_html=True)
    st.info("A section for logging and tracking QC checks at different production steps will be available here.")

    st.markdown('<h3 class="section-header" style="font-size:1.4rem;">Batch Traceability (Under Construction)</h3>', unsafe_allow_html=True)
    st.info("Tools for tracing product batches from raw materials to final product will be available here.")

