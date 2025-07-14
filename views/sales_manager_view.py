import streamlit as st
import datetime
import pandas as pd
import json
import time
from pathlib import Path

# --- Local Imports ---
import config
from utils.data_models import UserIndent, Priority
from utils.data_loader import DataLoader
from projections import get_projections_and_plots, SKU_TO_CATEGORY_MAP
# --- INTEGRATION: Import the CapacityAnalyzer ---
from capacity_analyzer import CapacityAnalyzer

# --- INTEGRATION: Helper for logging overrides ---
def log_manual_override(user, indent_data, report):
    """Logs the details of a manual override to a file."""
    log_file = Path("manual_overrides.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"--- MANUAL OVERRIDE LOG ---\n"
        f"Timestamp: {timestamp}\n"
        f"User: {user}\n"
        f"Reason: Infeasible plan was manually overridden.\n"
        f"Submitted Indents: {json.dumps(indent_data, indent=2, default=str)}\n"
        f"Feasibility Report: {json.dumps(report, indent=2)}\n"
        f"---------------------------\n\n"
    )
    with open(log_file, "a") as f:
        f.write(log_entry)

# --- State Management ---
def initialize_state():
    """Initializes all required session state variables."""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
        st.session_state.data_loader.load_sample_data()

    if 'capacity_analyzer' not in st.session_state:
        st.session_state.capacity_analyzer = CapacityAnalyzer(
            products=config.PRODUCTS, equipment=config.EQUIPMENTS, lines=config.LINES,
            tanks=config.TANKS, skus=config.SKUS, rooms=config.ROOMS
        )

    if 'projection_results' not in st.session_state:
        st.session_state.projection_results = None

    if 'user_indents_loaded' not in st.session_state:
        config.USER_INDENTS = st.session_state.data_loader.load_user_indents_with_fallback()
        st.session_state.user_indents_loaded = True
    
    if 'feasibility_report_for_plan' not in st.session_state:
        st.session_state.feasibility_report_for_plan = None


# --- Indent Management Functions ---
def save_indents_to_csv(data_loader, indents_to_save: dict):
    """Saves the current state of indents to user_indent.csv."""
    try:
        file_path = data_loader.data_dir / "user_indent.csv"
        if not indents_to_save:
            df_to_save = pd.DataFrame(columns=['Order_Number', 'SKU_ID', 'Qty_Required_Liters', 'Priority', 'Due_Date'])
        else:
            new_indents_data = [indent._to_dict() for indent in indents_to_save.values()]
            df_to_save = pd.DataFrame(new_indents_data)
        df_to_save.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"❌ Error saving indents to CSV: {e}")
        return False

# --- REWRITTEN: Indent Manager with new workflow ---
def render_indent_manager():
    """Renders the UI for managing a list of indents and checking feasibility of the entire plan."""
    analyzer = st.session_state.capacity_analyzer

    col1, col2 = st.columns([1.2, 1.5], gap="large")
    with col1:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">➕ Add New Indent</h3>', unsafe_allow_html=True)
        
        with st.form("add_indent_form", clear_on_submit=True):
            available_skus = list(config.SKUS.keys())
            sku_id = st.selectbox("Select SKU", options=available_skus)
            qty = st.number_input("Quantity (kg)", min_value=100, value=1000, step=100)
            due_date = st.date_input("Due Date", value=datetime.datetime.now().date() + datetime.timedelta(days=3))
            priority = st.selectbox("Priority", options=[p.name for p in Priority], index=1)
            
            submitted = st.form_submit_button("Add to Plan", use_container_width=True)
            if submitted:
                full_due_date = datetime.datetime.combine(due_date, datetime.time(17, 0))
                order_no = f"{sku_id}-{full_due_date.strftime('%d%m%y%H%M%S')}"
                new_indent = UserIndent(order_no=order_no, sku_id=sku_id, qty_required_liters=float(qty), due_date=full_due_date, priority=Priority[priority])
                config.USER_INDENTS[order_no] = new_indent
                # Clear feasibility results as the plan has changed
                st.session_state.feasibility_report_for_plan = None
                st.rerun()

    with col2:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📋 Current Production Plan</h3>', unsafe_allow_html=True)
        if config.USER_INDENTS:
            sorted_indents = sorted(config.USER_INDENTS.values(), key=lambda x: x.due_date)
            indent_data = [{"Order Number": i.order_no, "SKU ID": i.sku_id, "Quantity (kg)": i.qty_required_liters, "Due Date": i.due_date.strftime("%Y-%m-%d"), "Priority": i.priority.name} for i in sorted_indents]
            st.dataframe(pd.DataFrame(indent_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            # --- Feasibility Check for the entire plan ---
            if st.button("Check Feasibility of Full Plan", use_container_width=True, type="primary"):
                with st.spinner("Checking full plan feasibility..."):
                    # Treat all current indents as hard requirements for the check
                    production_plan = [{"sku_id": i.sku_id, "quantity_kg": i.qty_required_liters, "type": "hard"} for i in config.USER_INDENTS.values()]
                    report = analyzer.check_feasibility(production_plan, optimize_hard_constraints=True)
                    st.session_state.feasibility_report_for_plan = report
                st.rerun()
        else:
            st.info("No active indents. Add an indent to get started.")

    # --- Display Feasibility Results and Actions ---
    if st.session_state.feasibility_report_for_plan:
        report = st.session_state.feasibility_report_for_plan
        status = report.get("overall_status")
        
        st.markdown("---")
        st.markdown("#### Full Plan Feasibility Report")

        if status == "FEASIBLE":
            st.success("✅ The entire production plan is feasible!")
            if st.button("Confirm and Save All Indents to CSV", use_container_width=True):
                save_indents_to_csv(st.session_state.data_loader, config.USER_INDENTS)
                st.toast("Plan saved successfully!", icon="🎉")
                st.session_state.feasibility_report_for_plan = None
                time.sleep(1)
                st.rerun()
        else:
            if status == "INFEASIBLE":
                st.error("❌ This plan is not feasible as requested.")
            else: # FEASIBLE_WITH_ADJUSTMENTS
                st.warning("⚠️ This plan is feasible, but with adjustments.")
            
            st.dataframe(pd.DataFrame(report.get("analysis", [])), use_container_width=True, hide_index=True)
            st.info(f"The limiting factor is the **{report.get('system_bottleneck_for_this_plan')}** stage.")
            st.markdown("Please consider adjusting the plan or using Stock Transfer (STO).")

            if st.button("Manual Override & Save Plan Anyway", use_container_width=True):
                # Log the entire set of indents that were overridden
                indents_for_log = [i._to_dict() for i in config.USER_INDENTS.values()]
                log_manual_override("SalesManager", indents_for_log, report)
                save_indents_to_csv(st.session_state.data_loader, config.USER_INDENTS)
                st.success("✅ Plan saved with manual override.")
                st.session_state.feasibility_report_for_plan = None
                time.sleep(2)
                st.rerun()

# --- MODIFIED: Forecasting View with Caching Control ---
def render_forecasting_view():
    st.markdown("""<div class="info-box" style="border-left-color: #007bff;">Run the simulation to generate demand forecasts. Use the checkbox to force a new calculation, ignoring any saved results.</div>""", unsafe_allow_html=True)
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns([3, 1])
    with col1:
        run_button = st.button("📈 Generate / Refresh Projections", type="primary", use_container_width=True)
    with col2:
        force_rerun_check = st.checkbox("Force Re-run", value=False, help="If checked, the simulation will run from scratch, ignoring any cached data.")

    if run_button:
        with st.spinner("⏳ Running simulation and generating plots..."):
            # The checkbox value now controls whether to force a re-run
            st.session_state.projection_results = get_projections_and_plots(force_rerun=force_rerun_check)
        st.success("✅ Simulation complete! View results below.")

    if st.session_state.projection_results is None:
        st.info("Click the button above to generate projections.")
        return
        
    st.markdown("---")
    sku_tab, category_tab = st.tabs(["**SKU Forecast**", "**Product Category Forecast**"])
    results = st.session_state.projection_results
    with sku_tab:
        available_skus = sorted(list(results['sku_summary_figs'].keys()))
        selected_skus = st.multiselect("Select one or more SKUs:", options=available_skus)
        for sku in selected_skus:
            st.plotly_chart(results['sku_summary_figs'].get(sku), use_container_width=True)
    with category_tab:
        available_categories = sorted(list(results['category_summary_figs'].keys()))
        selected_categories = st.multiselect("Select one or more Product Categories:", options=available_categories)
        for cat in selected_categories:
            st.plotly_chart(results['category_summary_figs'].get(cat), use_container_width=True)

def render_production_plan_view():
    st.markdown("""<div class="info-box" style="border-left-color: #28a745;">Visualize the suggested production plan generated by the simulation.</div>""", unsafe_allow_html=True)
    if st.session_state.projection_results is None:
        st.warning("Please generate projections on the 'Demand Forecasting' tab first.")
        return
    results = st.session_state.projection_results
    sku_tab, category_tab = st.tabs(["**SKU Production Plan**", "**Category Production Plan**"])
    with sku_tab:
        available_skus = sorted(list(results['sku_production_figs'].keys()))
        selected_skus = st.multiselect("Select SKUs to view their production plan:", options=available_skus, key="prod_sku_multiselect")
        for sku in selected_skus:
            st.plotly_chart(results['sku_production_figs'].get(sku), use_container_width=True)
    with category_tab:
        available_categories = sorted(list(results['category_production_figs'].keys()))
        selected_categories = st.multiselect("Select Categories to view their aggregated production plan:", options=available_categories, key="prod_cat_multiselect")
        for cat in selected_categories:
            st.plotly_chart(results['category_production_figs'].get(cat), use_container_width=True)

# --- Main Render Function ---
def render():
    """Renders the main UI for the Sales Manager, with integrated tabs."""
    initialize_state()
    st.markdown('<h2 class="section-header">Sales & Production Dashboard</h2>', unsafe_allow_html=True)
    indent_tab, forecast_tab, production_tab = st.tabs([
        "**📝 Indent Management**",
        "**📈 Demand Forecasting**",
        "**📊 Production Plan**"
    ])
    with indent_tab:
        render_indent_manager()
    with forecast_tab:
        render_forecasting_view()
    with production_tab:
        render_production_plan_view()