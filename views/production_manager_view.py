import streamlit as st
from datetime import datetime
import pandas as pd
import copy

# --- Local Imports ---
# Use the new gantt chart view
from managers.gantt_chart_view import create_enhanced_gantt_chart 
from managers.gantt_chart import create_summary_dashboard # Keep using the old summary
from managers.summary_dashboard_view import create_summary_dashboard_data, render_summary_dashboard
from heuristic_scheduler import HeuristicScheduler
from optimiser_engine import ProductionOptimizer
import config

# --- Helper Functions for Display Logic (display_feasibility_results, display_override_modal remain the same) ---
def display_feasibility_results():
    """Renders the UI for displaying feasibility check results."""
    result = st.session_state.feasibility_result
    if result['status'] == 'Success':
        st.success("✅ **Feasibility Check Passed:** The current indent list is fully schedulable.")
        if st.button("🚀 Generate Schedule", type="primary", use_container_width=True):
             with st.spinner("Running Heuristic Scheduler..."):
                scheduler = HeuristicScheduler(
                    indents=config.USER_INDENTS, skus=config.SKUS, products=config.PRODUCTS,
                    lines=config.LINES, tanks=config.TANKS, equipments=config.EQUIPMENTS, shifts=config.SHIFTS
                )
                schedule_result = scheduler.run_heuristic_scheduler()
                schedule_result.is_manual_override = False
                st.session_state.last_schedule_result = schedule_result
                st.session_state.feasibility_result = None # Reset state
                st.rerun()
    else:
        st.warning("⚠️ **Feasibility Check Failed:** The current indent list is not schedulable with available capacity.")
        st.markdown("The optimizer suggests the following feasible production mix:")
        
        suggested_indents = result.get('feasible_indents', {})
        if suggested_indents:
            comparison_data = []
            for order_no, original_indent in config.USER_INDENTS.items():
                suggested_indent = suggested_indents.get(order_no)
                original_qty = original_indent.qty_required_liters
                suggested_qty = suggested_indent.qty_required_liters if suggested_indent else 0
                comparison_data.append({
                    "Order Number": order_no, "SKU ID": original_indent.sku_id,
                    "Original Qty (L)": f"{original_qty:,.0f}", 
                    "Suggested Qty (L)": f"{suggested_qty:,.0f}",
                    "Change (L)": f"{suggested_qty - original_qty:,.0f}"
                })
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

            if st.button("🚨 Override & Use Suggestion", use_container_width=True):
                st.session_state.show_override_modal = True
                st.rerun()

def display_override_modal():
    """Renders the confirmation modal for a manual override."""
    with st.container(border=True):
        st.error("### Manual Override Confirmation")
        st.markdown("""
        You are about to override the original indent list with the optimizer's suggestion. 
        This action will be logged and will generate a schedule based on the reduced quantities.
        **This may result in under-production for some orders.**
        """)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm Override", use_container_width=True, type="primary"):
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": "Production Manager",
                    "action": "Manual Override Confirmed",
                    "details": "Accepted optimizer's suggestion for a feasible schedule."
                }
                if 'manual_override_log' not in st.session_state:
                    st.session_state.manual_override_log = []
                st.session_state.manual_override_log.append(log_entry)
                
                with st.spinner("Generating schedule with suggested quantities..."):
                    suggested_indents = st.session_state.feasibility_result.get('feasible_indents', {})
                    scheduler = HeuristicScheduler(
                        indents=suggested_indents, skus=config.SKUS, products=config.PRODUCTS,
                        lines=config.LINES, tanks=config.TANKS, equipments=config.EQUIPMENTS, shifts=config.SHIFTS
                    )
                    schedule_result = scheduler.run_heuristic_scheduler()
                    schedule_result.is_manual_override = True
                    st.session_state.last_schedule_result = schedule_result
                
                st.session_state.show_override_modal = False
                st.session_state.feasibility_result = None
                st.success("Override successful. A new schedule has been generated.")
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_override_modal = False
                st.rerun()

# --- Updated Analytics Display ---
def display_analytics():
    """Renders the analytics dashboard with the enhanced Gantt chart."""
    st.markdown('<h2 class="section-header">Schedule Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    analytics_result = st.session_state.get('last_schedule_result')
    if getattr(analytics_result, 'is_manual_override', False):
        st.warning("🚨 **Manual Override Active:** This schedule was generated from a feasible suggestion, not the original indents.")

    summary_data = create_summary_dashboard_data(analytics_result, config.USER_INDENTS)
    summary_html = render_summary_dashboard(summary_data)
    if summary_html: 
        st.markdown(summary_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Gantt Chart Visualization")

    # Add options for the user to customize the chart
    color_option = st.radio(
        "Color-code Gantt chart by:",
        ('Production Line', 'Product'),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    color_by = 'line' if color_option == 'Production Line' else 'product'

    gantt_chart = create_enhanced_gantt_chart(analytics_result, color_by=color_by)
    if gantt_chart:
        st.plotly_chart(gantt_chart, use_container_width=True)
    else:
        st.info("The generated schedule has no tasks to display.")

# --- Main Render Function (no changes needed here) ---
def render():
    """
    Renders the UI for the Production Manager, with corrected state handling.
    """
    # Initialize state variables
    for key, default_val in [('feasibility_result', None), ('show_override_modal', False), 
                             ('manual_override_log', []), ('indents_loaded', False),
                             ('last_schedule_result', None)]:
        if key not in st.session_state:
            st.session_state[key] = default_val

    st.markdown('<h2 class="section-header">Scheduling Workbench</h2>', unsafe_allow_html=True)
    
    # --- Action Buttons ---
    st.markdown('<div class="action-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Load Production Indents", use_container_width=True):
            # Reset all states when loading new data
            st.session_state.feasibility_result = None
            st.session_state.show_override_modal = False
            st.session_state.last_schedule_result = None
            
            data_loader = st.session_state.data_loader
            config.USER_INDENTS.clear()
            config.USER_INDENTS.update(data_loader.load_user_indents_with_fallback())
            st.session_state.indents_loaded = True
            st.success("Indents loaded. Ready for feasibility check.")
            # No rerun here, let Streamlit handle the update

    with col2:
        if st.button("🔬 Check Feasibility", use_container_width=True, disabled=not st.session_state.indents_loaded):
            with st.spinner("Running feasibility analysis... This may take a moment."):
                base_config = {
                    "skus": config.SKUS, "products": config.PRODUCTS, "lines": config.LINES,
                    "tanks": config.TANKS, "equipments": config.EQUIPMENTS, "shifts": config.SHIFTS
                }
                optimizer = ProductionOptimizer(base_config)
                result = optimizer.find_feasible_baseline(copy.deepcopy(config.USER_INDENTS))
                st.session_state.feasibility_result = result
                st.session_state.last_schedule_result = None # Clear old schedule
                # No rerun here
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Main Display Logic ---
    # This structure ensures only one primary state is shown at a time.
    if st.session_state.show_override_modal:
        display_override_modal()
    elif st.session_state.feasibility_result:
        display_feasibility_results()
    elif st.session_state.indents_loaded:
        st.markdown('<h3 class="section-header" style="margin-top:0; font-size:1.4rem;">📋 Pending Indents Queue</h3>', unsafe_allow_html=True)
        if config.USER_INDENTS:
            sorted_indents = sorted(config.USER_INDENTS.values(), key=lambda x: (x.due_date, x.priority.value))
            indent_data = [{"Order Number": i.order_no, "SKU ID": i.sku_id, "Quantity (Liters)": i.qty_required_liters,
                            "Due Date": i.due_date.strftime("%Y-%m-%d %H:%M"), "Priority": i.priority.name} for i in sorted_indents]
            st.dataframe(pd.DataFrame(indent_data), use_container_width=True, hide_index=True)
        else:
            st.info("No indents were found in the loaded data.")
    else:
        st.info("Click 'Load Production Indents' to begin the scheduling process.")

    # --- Analytics Display ---
    if st.session_state.last_schedule_result:
        st.markdown("---")
        display_analytics()