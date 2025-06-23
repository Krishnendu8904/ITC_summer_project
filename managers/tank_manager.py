import streamlit as st
import pandas as pd
import config
from utils.data_models import Tank, ResourceStatus, TankType
from utils.data_loader import DataLoader

def render_tank_manager(data_loader: DataLoader):
    """
    Renders the Tank configuration UI for editing and saving production tanks.
    """
    

    # Use session state for interactive editing
    if 'tank_config' not in st.session_state:
        st.session_state.tank_config = config.TANKS.copy()

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        st.subheader("🛢️ Production Tank Configuration")
        st.markdown("Manage properties of each production tank, including capacity, type, and compatibility.")
    with col2:
        if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_tank_btn"):
            try:
                config.TANKS.update(data_loader.load_tanks_with_fallback())
                st.session_state.tank_config = config.TANKS.copy()
            except Exception as e:
                st.error(f"❌ Error reloading data: {e}")
                st.exception(e)

    # Create list of tank IDs for selectbox
    tank_ids = list(st.session_state.tank_config.keys())
    selectbox_options = ["-- Add New Tank --"] + tank_ids

    # Tank selection dropdown
    selected_option = st.selectbox(
        "Select Tank to Edit or Add New:",
        options=selectbox_options,
        key="tank_selector"
    )

    # Determine if we're adding new or editing existing
    is_new_tank = selected_option == "-- Add New Tank --"
    selected_tank = None if is_new_tank else st.session_state.tank_config.get(selected_option)

    # Get available options for dropdowns
    status_options = [s.value for s in ResourceStatus]
    tank_type_options = [t.value for t in TankType]

    # Tank editing form
    with st.form(key="tank_form"):
        st.markdown("### Tank Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tank_id = st.text_input(
                "Tank ID",
                value="" if is_new_tank else selected_tank.tank_id,
                disabled=not is_new_tank,  # Disable editing ID for existing tanks
                help="Unique identifier for the tank"
            )
            
            capacity_liters = st.number_input(
                "Capacity (Liters)",
                min_value=0.0,
                value=0.0 if is_new_tank else selected_tank.capacity_liters,
                step=100.0,
                help="Total capacity of the tank in liters"
            )
            
            tank_type = st.selectbox(
                "Tank Type",
                options=tank_type_options,
                index=0 if is_new_tank else tank_type_options.index(selected_tank.tank_type.value),
                help="The functional type of the tank"
            )
            
            status = st.selectbox(
                "Status",
                options=status_options,
                index=0 if is_new_tank else status_options.index(selected_tank.status.value),
                help="Current operational status of the tank"
            )

        with col2:
            current_volume_liters = st.number_input(
                "Current Volume (Liters)",
                min_value=0.0,
                value=0.0 if is_new_tank else selected_tank.current_volume_liters,
                step=10.0,
                help="Current volume of product in the tank"
            )
            
            current_product_category = st.text_input(
                "Current Product Category",
                value="" if is_new_tank else (selected_tank.current_product_category or ""),
                help="The product category currently in the tank"
            )
            
            cip_duration_min = st.number_input(
                "CIP Duration (Minutes)",
                min_value=0,
                value=0 if is_new_tank else selected_tank.CIP_duration_minutes,
                step=5,
                help="Time required for a full Clean-In-Place cycle"
            )

        # Compatible products input
        compatible_products_str = st.text_area(
            "Compatible Product Categories (comma-separated)",
            value="" if is_new_tank else ", ".join(selected_tank.compatible_product_categories),
            help="List of product categories that can be processed in this tank, separated by commas (e.g., CURD, MILK, CHEESE)"
        )

        # Form submission buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if is_new_tank:
                save_button = st.form_submit_button("🆕 Create New Tank", type="primary")
            else:
                save_button = st.form_submit_button("💾 Save Changes", type="primary")

        # Handle form submission
        if save_button:
            # Validation
            if not tank_id or not tank_id.strip():
                st.error("Tank ID cannot be empty.")
            elif is_new_tank and tank_id in st.session_state.tank_config:
                st.error(f"Tank ID '{tank_id}' already exists. Please choose a different ID.")
            elif current_volume_liters > capacity_liters:
                st.error("Current volume cannot exceed tank capacity.")
            else:
                try:
                    # Process compatible product categories
                    compatible_cats = []
                    if compatible_products_str.strip():
                        compatible_cats = [cat.strip() for cat in compatible_products_str.split(',') if cat.strip()]
                    
                    # Convert enums
                    status_enum = ResourceStatus(status)
                    tank_type_enum = TankType(tank_type)
                    
                    # Handle optional current product category
                    current_product = current_product_category.strip() if current_product_category.strip() else None
                    
                    # Create/update tank object
                    tank_obj = Tank(
                        tank_id=tank_id,
                        capacity_liters=float(capacity_liters),
                        compatible_product_categories=compatible_cats,
                        status=status_enum,
                        tank_type=tank_type_enum,
                        current_product_category=current_product,
                        current_volume_liters=float(current_volume_liters),
                        CIP_duration_minutes=int(cip_duration_min)
                    )
                    
                    # Update session state
                    st.session_state.tank_config[tank_id] = tank_obj
                    
                    if is_new_tank:
                        st.success(f"✅ Tank '{tank_id}' created successfully!")
                    else:
                        st.success(f"✅ Tank '{tank_id}' updated successfully!")
                    
                    st.rerun()
                    
                except ValueError as e:
                    st.error(f"Invalid value: {e}")
                except Exception as e:
                    st.error(f"Error saving tank: {e}")

    # Delete button (outside form to avoid form submission conflicts)
    if not is_new_tank and selected_tank:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Delete Selected Tank", type="secondary", key='delete_tank_button'):
                # Use a confirmation dialog
                st.session_state.show_delete_confirmation = True
        
        # Handle delete confirmation
        if st.session_state.get('show_delete_confirmation', False):
            st.warning(f"Are you sure you want to delete tank '{selected_option}'?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary", key = 'confirmation'):
                    del st.session_state.tank_config[selected_option]
                    st.session_state.show_delete_confirmation = False
                    st.success(f"Tank '{selected_option}' deleted successfully!")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key='cancel_confirmation'):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()

    # Display current tanks summary
    if st.session_state.tank_config:
        st.markdown("---")
        st.markdown("### Current Tanks Summary")
        
        # Create summary dataframe
        summary_data = []
        for tank_id, tank in st.session_state.tank_config.items():
            # Calculate utilization percentage
            utilization = (tank.current_volume_liters / tank.capacity_liters * 100) if tank.capacity_liters > 0 else 0
            
            summary_data.append({
                "Tank ID": tank.tank_id,
                "Type": tank.tank_type.value,
                "Capacity (L)": f"{tank.capacity_liters:,.0f}",
                "Current Volume (L)": f"{tank.current_volume_liters:,.0f}",
                "Utilization (%)": f"{utilization:.1f}%",
                "Status": tank.status.value,
                "Current Product": tank.current_product_category or "Empty",
                "Compatible Products": tank.compatible_product_categories,
                "CIP Duration (min)": tank.CIP_duration_minutes
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    # Save to CSV button
    st.markdown("---")
    if st.button("💾 Save All Changes to CSV", use_container_width=True, type="primary", key= 'save_all_tank_config'):
        with st.spinner("Saving Tanks to CSV..."):
            try:
                # Validate all tanks before saving
                for tank_id, tank in st.session_state.tank_config.items():
                    if not tank_id:
                        st.error("Found tank with empty ID. Please correct before saving.")
                        return
                    if tank.current_volume_liters > tank.capacity_liters:
                        st.error(f"Tank '{tank_id}' current volume exceeds capacity. Please correct before saving.")
                        return

                # Save to CSV
                file_path = data_loader.data_dir / "tank_config.csv"
                df_to_save = pd.DataFrame([tank._to_dict() for tank in st.session_state.tank_config.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ All tank configurations saved to CSV successfully!")

                # Reload all data to ensure consistency across the app
                data_loader.load_all_data()
                st.session_state.tank_config = config.TANKS.copy()

            except Exception as e:
                st.error(f"❌ Error saving tank data to CSV: {e}")

    # Show current count and summary stats
    if st.session_state.tank_config:
        col1, col2, col3, col4 = st.columns(4)
        
        total_tanks = len(st.session_state.tank_config)
        total_capacity = sum(tank.capacity_liters for tank in st.session_state.tank_config.values())
        total_current_volume = sum(tank.current_volume_liters for tank in st.session_state.tank_config.values())
        avg_utilization = (total_current_volume / total_capacity * 100) if total_capacity > 0 else 0
        
        with col1:
            st.metric("Total Tanks", total_tanks)
        with col2:
            st.metric("Total Capacity", f"{total_capacity:,.0f} L")
        with col3:
            st.metric("Current Volume", f"{total_current_volume:,.0f} L")
        with col4:
            st.metric("Avg Utilization", f"{avg_utilization:.1f}%")