import streamlit as st
import pandas as pd
import config
from utils.data_models import Shift
from utils.data_loader import DataLoader
from datetime import datetime, time

def convert_to_time(val):
    if isinstance(val, time):
        return val
    try:
        return datetime.strptime(val, "%H:%M").time()
    except:
        return None

def render_shift_manager(data_loader: DataLoader):
    """
    Renders the Shift configuration UI for editing and saving Shifts.
    """
    

    # Use session state for interactive editing
    if 'shift_config' not in st.session_state:
        st.session_state.shift_config = config.SHIFTS.copy()

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        st.subheader("⏰ Shift Configuration")
        st.markdown("Define production shifts, their properties, and active status.")
    with col2:
        if st.button("🔃 RELOAD", use_container_width=True, type='primary', key="reload_shift_btn"):
            try:
                config.SHIFTS.update(data_loader.load_shifts_with_fallback())
                st.session_state.shift_config = config.SHIFTS.copy()
            except Exception as e:
                st.error(f"❌ Error reloading data: {e}")
                st.exception(e)

    # Create list of shift IDs for selectbox
    shift_ids = list(st.session_state.shift_config.keys())
    selectbox_options = ["-- Add New Shift --"] + shift_ids

    # Shift selection dropdown
    selected_option = st.selectbox(
        "Select Shift to Edit or Add New:",
        options=selectbox_options,
        key="shift_selector"
    )

    # Determine if we're adding new or editing existing
    is_new_shift = selected_option == "-- Add New Shift --"
    selected_shift = None if is_new_shift else st.session_state.shift_config.get(selected_option)

    # Shift editing form
    with st.form(key="shift_form"):
        st.markdown("### Shift Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            shift_id = st.text_input(
                "Shift ID",
                value="" if is_new_shift else selected_shift.shift_id,
                disabled=not is_new_shift,  # Disable editing ID for existing shifts
                help="Unique identifier for the shift (e.g., DAY, NIGHT, MORNING)"
            )
            
            start_time = st.time_input(
                "Start Time",
                value=time(9, 0) if is_new_shift else selected_shift.start_time,
                help="When the shift begins"
            )

        with col2:
            end_time = st.time_input(
                "End Time",
                value=time(17, 0) if is_new_shift else selected_shift.end_time,
                help="When the shift ends"
            )
            
            is_active = st.checkbox(
                "Is Active",
                value=True if is_new_shift else selected_shift.is_active,
                help="Whether this shift is currently active/available for scheduling"
            )

        # Shift duration calculation and display
        if start_time and end_time:
            # Calculate duration (handle overnight shifts)
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            
            if end_minutes <= start_minutes:
                # Overnight shift
                duration_minutes = (24 * 60) - start_minutes + end_minutes
                shift_type = "Overnight Shift"
            else:
                # Same day shift
                duration_minutes = end_minutes - start_minutes
                shift_type = "Same Day Shift"
            
            duration_hours = duration_minutes / 60
            
            st.info(f"**{shift_type}** - Duration: {duration_hours:.1f} hours ({duration_minutes} minutes)")

        # Form submission buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if is_new_shift:
                save_button = st.form_submit_button("🆕 Create New Shift", type="primary")
            else:
                save_button = st.form_submit_button("💾 Save Changes", type="primary")

        # Handle form submission
        if save_button:
            # Validation
            if not shift_id or not shift_id.strip():
                st.error("Shift ID cannot be empty.")
            elif is_new_shift and shift_id in st.session_state.shift_config:
                st.error(f"Shift ID '{shift_id}' already exists. Please choose a different ID.")
            elif not start_time or not end_time:
                st.error("Both start time and end time must be specified.")
            elif start_time == end_time:
                st.error("Start time and end time cannot be the same.")
            else:
                try:
                    # Create/update shift object
                    shift_obj = Shift(
                        shift_id=shift_id,
                        start_time=start_time,
                        end_time=end_time,
                        is_active=is_active
                    )
                    
                    # Update session state
                    st.session_state.shift_config[shift_id] = shift_obj
                    
                    if is_new_shift:
                        st.success(f"✅ Shift '{shift_id}' created successfully!")
                    else:
                        st.success(f"✅ Shift '{shift_id}' updated successfully!")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error saving shift: {e}")

    # Delete button (outside form to avoid form submission conflicts)
    if not is_new_shift and selected_shift:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Delete Selected Shift", type="secondary", key= 'delete_shift'):
                # Use a confirmation dialog
                st.session_state.show_delete_confirmation = True
        
        # Handle delete confirmation
        if st.session_state.get('show_delete_confirmation', False):
            st.warning(f"Are you sure you want to delete shift '{selected_option}'?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary", key= 'confirm_delete_shift'):
                    del st.session_state.shift_config[selected_option]
                    st.session_state.show_delete_confirmation = False
                    st.success(f"Shift '{selected_option}' deleted successfully!")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key= 'cancel_delete'):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()

    # Display current shifts summary
    if st.session_state.shift_config:
        st.markdown("---")
        st.markdown("### Current Shifts Summary")
        
        # Create summary dataframe
        summary_data = []
        for shift_id, shift in st.session_state.shift_config.items():
            # Calculate duration
            start_minutes = shift.start_time.hour * 60 + shift.start_time.minute
            end_minutes = shift.end_time.hour * 60 + shift.end_time.minute
            
            if end_minutes <= start_minutes:
                # Overnight shift
                duration_minutes = (24 * 60) - start_minutes + end_minutes
                shift_type = "Overnight"
            else:
                # Same day shift
                duration_minutes = end_minutes - start_minutes
                shift_type = "Same Day"
            
            duration_hours = duration_minutes / 60
            
            summary_data.append({
                "Shift ID": shift.shift_id,
                "Start Time": shift.start_time.strftime("%H:%M"),
                "End Time": shift.end_time.strftime("%H:%M"),
                "Duration (hrs)": f"{duration_hours:.1f}",
                "Type": shift_type,
                "Status": "🟢 Active" if shift.is_active else "🔴 Inactive"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    # Show shift coverage analysis
    if st.session_state.shift_config:
        st.markdown("### Shift Coverage Analysis")
        
        active_shifts = [shift for shift in st.session_state.shift_config.values() if shift.is_active]
        
        if active_shifts:
            # Create 24-hour coverage map
            coverage = [0] * 24  # Hours 0-23
            
            for shift in active_shifts:
                start_hour = shift.start_time.hour
                end_hour = shift.end_time.hour
                
                if end_hour <= start_hour:  # Overnight shift
                    # Cover from start_hour to 23
                    for hour in range(start_hour, 24):
                        coverage[hour] += 1
                    # Cover from 0 to end_hour
                    for hour in range(0, end_hour):
                        coverage[hour] += 1
                else:  # Same day shift
                    for hour in range(start_hour, end_hour):
                        coverage[hour] += 1
            
            # Display coverage summary
            col1, col2, col3 = st.columns(3)
            
            hours_covered = sum(1 for c in coverage if c > 0)
            max_coverage = max(coverage) if coverage else 0
            avg_coverage = sum(coverage) / 24 if coverage else 0
            
            with col1:
                st.metric("Hours Covered", f"{hours_covered}/24")
            with col2:
                st.metric("Max Overlaps", max_coverage)
            with col3:
                st.metric("Avg Coverage", f"{avg_coverage:.1f}")
            
            # Show gaps in coverage
            gaps = []
            for hour in range(24):
                if coverage[hour] == 0:
                    gaps.append(f"{hour:02d}:00-{(hour+1):02d}:00")
            
            if gaps:
                st.warning(f"⚠️ **Coverage Gaps:** {', '.join(gaps)}")
            else:
                st.success("✅ **24-hour coverage achieved!**")

    # Save to CSV button
    st.markdown("---")
    if st.button("💾 Save All Changes to CSV", use_container_width=True, type="primary", key= 'save_all_shift_config'):
        with st.spinner("Saving Shifts to CSV..."):
            try:
                # Validate all shifts before saving
                for shift_id, shift in st.session_state.shift_config.items():
                    if not shift_id:
                        st.error("Found shift with empty ID. Please correct before saving.")
                        return
                    if not shift.start_time or not shift.end_time:
                        st.error(f"Shift '{shift_id}' is missing start or end time. Please correct before saving.")
                        return
                    if shift.start_time == shift.end_time:
                        st.error(f"Shift '{shift_id}' has identical start and end times. Please correct before saving.")
                        return

                # Save to CSV
                file_path = data_loader.data_dir / "shift_config.csv"
                df_to_save = pd.DataFrame([shift._to_dict() for shift in st.session_state.shift_config.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ All shift configurations saved to CSV successfully!")

                # Reload data to reflect changes
                config.SHIFTS = data_loader.load_shifts_with_fallback()
                st.session_state.shift_config = config.SHIFTS.copy()

            except Exception as e:
                st.error(f"❌ Error saving shift data to CSV: {e}")

    # Show current count and summary stats
    if st.session_state.shift_config:
        col1, col2, col3, col4 = st.columns(4)
        
        total_shifts = len(st.session_state.shift_config)
        active_shifts_count = sum(1 for shift in st.session_state.shift_config.values() if shift.is_active)
        inactive_shifts_count = total_shifts - active_shifts_count
        
        # Calculate total active hours per day
        total_active_hours = 0
        for shift in st.session_state.shift_config.values():
            if shift.is_active:
                start_minutes = shift.start_time.hour * 60 + shift.start_time.minute
                end_minutes = shift.end_time.hour * 60 + shift.end_time.minute
                
                if end_minutes <= start_minutes:
                    duration_minutes = (24 * 60) - start_minutes + end_minutes
                else:
                    duration_minutes = end_minutes - start_minutes
                
                total_active_hours += duration_minutes / 60
        
        with col1:
            st.metric("Total Shifts", total_shifts)
        with col2:
            st.metric("Active Shifts", active_shifts_count)
        with col3:
            st.metric("Inactive Shifts", inactive_shifts_count)
        with col4:
            st.metric("Total Active Hours/Day", f"{total_active_hours:.1f}")