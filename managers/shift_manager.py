import streamlit as st
import pandas as pd
import config
from models.data_models import Shift
from data_loader import DataLoader
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
    st.subheader("⏰ Shift Configuration")
    st.markdown("Define production shifts, their properties, and active status.")

    if 'shift_config' not in st.session_state:
        st.session_state.shift_config = config.SHIFTS.copy()

    try:
        df_display = pd.DataFrame([shift._to_dict() for shift in st.session_state.shift_config.values()])
    except Exception as e:
        st.error(f"Error preparing Shift data for display: {e}")
        st.error("Please ensure the data models and CSV headers are consistent.")
        return
    
    df_display["Start_Time"] = df_display["Start_Time"].apply(convert_to_time)
    df_display["End_Time"] = df_display["End_Time"].apply(convert_to_time)
    edited_df = st.data_editor(
        df_display,
        
        num_rows="dynamic",
        key="shift_editor",
        use_container_width=True,
        column_config={
            "Shift_ID": st.column_config.TextColumn("Shift ID", required=True),
            "Start_Time": st.column_config.TimeColumn("Start Time", required=True, format="HH:mm"),
            "End_Time": st.column_config.TimeColumn("End Time", required=True, format="HH:mm"),
            "Is_Active": st.column_config.CheckboxColumn("Is Active?", default=True)
        }
    )

    if st.button("💾 Save Shifts to CSV", use_container_width=True, type="primary"):
        with st.spinner("Saving Shifts..."):
            try:
                updated_shifts = {}
                for _, row in edited_df.iterrows():
                    shift_id = row["Shift_ID"]
                    if not shift_id:
                        st.error("Shift ID cannot be empty. Please correct and save again.")
                        return
                    if shift_id in updated_shifts:
                        st.error(f"Duplicate Shift ID '{shift_id}' found. IDs must be unique.")
                        return

                    # Ensure Start_Time and End_Time are datetime.time objects
                    start_time = row["Start_Time"]
                    end_time = row["End_Time"]

                    if pd.isna(start_time) or pd.isna(end_time):
                        st.error(f"Start/End time missing for Shift ID '{shift_id}'.")
                        return

                    if not isinstance(start_time, time) or not isinstance(end_time, time):
                        st.error(f"Invalid time format for Shift ID '{shift_id}'. Please use HH:MM.")
                        return

                    updated_shifts[shift_id] = Shift(
                        shift_id=shift_id,
                        start_time=start_time,
                        end_time=end_time,
                        is_active=bool(row["Is_Active"])
                    )

                file_path = data_loader.data_dir / "shift_config.csv"
                df_to_save = pd.DataFrame([shift._to_dict() for shift in updated_shifts.values()])
                df_to_save.to_csv(file_path, index=False)

                st.success("✅ Shift configuration saved successfully!")
                config.SHIFTS = data_loader.load_shifts_with_fallback()
                st.session_state.shift_config = config.SHIFTS.copy()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving Shift data: {e}")
