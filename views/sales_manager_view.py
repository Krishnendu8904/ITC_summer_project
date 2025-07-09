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
        f"Submitted Indent: {indent_data}\n"
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
        # --- INTEGRATION: Load all data needed for analyzer ---
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
    
    # State for feasibility check results
    if 'feasibility_result' not in st.session_state:
        st.session_state.feasibility_result = None
    if 'new_indent_data' not in st.session_state:
        st.session_state.new_indent_data = None


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

# --- INTEGRATION: Refactored Indent Manager with Feasibility Check ---
def render_indent_manager():
    """Renders the UI for creating and deleting indents, with a new feasibility check workflow."""
    analyzer = st.session_state.capacity_analyzer

    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📝 Create New Indent</h3>', unsafe_allow_html=True)
        
        # --- Step 1: Input Form ---
        with st.form("new_indent_form"):
            available_skus = list(config.SKUS.keys())
            sku_id = st.selectbox("Select SKU", options=available_skus)
            qty = st.number_input("Quantity (kg)", min_value=100, value=1000, step=100)
            due_date = st.date_input("Due Date", value=datetime.datetime.now().date() + datetime.timedelta(days=3))
            due_time = st.time_input("Due Time", value=datetime.time(17, 0))
            priority = st.selectbox("Priority", options=[p.name for p in Priority], index=1)
            
            submitted = st.form_submit_button("Check Feasibility & Submit", use_container_width=True, type="primary")
            if submitted:
                st.session_state.feasibility_result = None # Clear old results
                full_due_date = datetime.datetime.combine(due_date, due_time)
                # Store new indent data temporarily
                st.session_state.new_indent_data = {
                    "sku_id": sku_id, "quantity_kg": float(qty), "due_date": full_due_date, "priority": priority
                }
                
                # Construct the full plan to check
                production_plan = []
                # Add existing indents as "hard" constraints
                for indent in config.USER_INDENTS.values():
                    production_plan.append({"sku_id": indent.sku_id, "quantity_kg": indent.qty_required_liters, "type": "hard"})
                # Add the new indent as a "soft" constraint for the check
                production_plan.append({"sku_id": sku_id, "quantity_kg": float(qty), "type": "soft"})

                with st.spinner("Checking plan feasibility..."):
                    # Always try to optimize hard constraints for sales view
                    report = analyzer.check_feasibility(production_plan, optimize_hard_constraints=True)
                    st.session_state.feasibility_result = report
                st.rerun()

        # --- Step 2: Display Results and Handle Actions ---
        if st.session_state.feasibility_result:
            report = st.session_state.feasibility_result
            status = report.get("overall_status")
            new_indent_info = st.session_state.new_indent_data

            if status == "FEASIBLE":
                st.success("✅ Plan is Feasible! Submitting indent.")
                # Add to main indent list and save
                order_no = f"{new_indent_info['sku_id']}-{new_indent_info['due_date'].strftime('%d%m%y')}"
                new_indent = UserIndent(order_no=order_no, sku_id=new_indent_info['sku_id'], qty_required_liters=new_indent_info['quantity_kg'], due_date=new_indent_info['due_date'], priority=Priority[new_indent_info['priority']])
                config.USER_INDENTS[order_no] = new_indent
                save_indents_to_csv(st.session_state.data_loader, config.USER_INDENTS)
                # Clean up state
                st.session_state.feasibility_result = None
                st.session_state.new_indent_data = None
                time.sleep(1) # Give user time to read message
                st.rerun()

            elif status in ["FEASIBLE_WITH_ADJUSTMENTS", "INFEASIBLE"]:
                st.error("⚠️ Plan Not Fully Feasible")
                st.markdown("The current production plan cannot accommodate the full request.")
                
                # Find the analysis for the new indent (it was the last 'soft' item)
                new_indent_analysis = next((item for item in reversed(report.get("analysis", [])) if item['type'] == 'soft'), None)

                if new_indent_analysis:
                    requested = new_indent_analysis['requested_kg']
                    achieved = new_indent_analysis['achievable_kg']
                    shortfall = requested - achieved
                    st.warning(f"Your request for **{requested} kg** of `{new_indent_analysis['sku_id']}` can only be fulfilled up to **{achieved} kg** (Shortfall: {shortfall} kg).")
                    st.info(f"The limiting factor is the **{report.get('system_bottleneck_for_this_plan')}** stage.")

                st.markdown("Please consider placing a smaller order or using Stock Transfer (STO).")

                if st.button("Manual Override & Submit Anyway", use_container_width=True):
                    log_manual_override("SalesManager", new_indent_info, report)
                    # Add to main indent list and save
                    order_no = f"{new_indent_info['sku_id']}-{new_indent_info['due_date'].strftime('%d%m%y')}"
                    new_indent = UserIndent(order_no=order_no, sku_id=new_indent_info['sku_id'], qty_required_liters=new_indent_info['quantity_kg'], due_date=new_indent_info['due_date'], priority=Priority[new_indent_info['priority']])
                    config.USER_INDENTS[order_no] = new_indent
                    save_indents_to_csv(st.session_state.data_loader, config.USER_INDENTS)
                    st.success("✅ Indent submitted with manual override.")
                    # Clean up state
                    st.session_state.feasibility_result = None
                    st.session_state.new_indent_data = None
                    time.sleep(2)
                    st.rerun()

    with col2:
        st.markdown('<h3 class="section-header" style="font-size:1.4rem; margin-top:0;">📋 Live Production Indents</h3>', unsafe_allow_html=True)
        if config.USER_INDENTS:
            sorted_indents = sorted(config.USER_INDENTS.values(), key=lambda x: x.due_date)
            indent_data = [{"Order Number": i.order_no, "SKU ID": i.sku_id, "Quantity (L)": i.qty_required_liters, "Due Date": i.due_date.strftime("%Y-%m-%d %H:%M"), "Priority": i.priority.name} for i in sorted_indents]
            st.dataframe(pd.DataFrame(indent_data), use_container_width=True, hide_index=True)
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

# --- Forecasting and Plan Views (No changes) ---
def render_forecasting_view():
    st.markdown("""<div class="info-box" style="border-left-color: #007bff;">Run the complete simulation to generate demand forecasts.</div>""", unsafe_allow_html=True)
    if st.button("📈 Generate Projections", type="primary", use_container_width=True):
        with st.spinner("⏳ Running full simulation and generating plots..."):
            st.session_state.projection_results = get_projections_and_plots(force_rerun=True)
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
