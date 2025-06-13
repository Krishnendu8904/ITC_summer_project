import streamlit as st
import pandas as pd
import config
from models.data_models import Tank, ResourceStatus, TankType
from data_loader import DataLoader

def render_tank_manager(data_loader: DataLoader):
    """
    Renders the Tank configuration UI for editing and saving production tanks.
    """
    st.subheader("🛢️ Production Tank Configuration")
    st.markdown("Manage properties of each production tank, including capacity, type, and compatibility.")

    if 'tank_config' not in st.session_state:
        st.session_state.tank_config = config.TANKS.copy()

    try:
        df_display = pd.DataFrame([tank._to_dict() for tank in st.session_state.tank_config.values()])
    except Exception as e:
        st.error(f"Error preparing Tank data for display: {e}")
        return

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        key="tank_editor",
        use_container_width=True,
        column_config={
            "Tank_ID": st.column_config.TextColumn("Tank ID", help="Unique identifier for the tank.", required=True),
            "Capacity_Liters": st.column_config.NumberColumn("Capacity (Liters)", help="Total capacity of the tank in liters.", min_value=0.0, required=True),
            "Compatible_Product_Categories": st.column_config.TextColumn("Compatible Products", help="Comma-separated list of product categories (e.g., CURD,MILK)."),
            "Status": st.column_config.SelectboxColumn("Status", help="Current status of the tank.", options=[s.value for s in ResourceStatus], required=True),
            "Tank_Type": st.column_config.SelectboxColumn("Tank Type", help="The functional type of the tank.", options=[t.value for t in TankType], required=True),
            "Current_Product_Category": st.column_config.TextColumn("Current Product", help="The product category currently in the tank."),
            "Current_Volume_Liters": st.column_config.NumberColumn("Current Volume (Liters)", help="Current volume of product in the tank.", min_value=0.0),
            "CIP_Duration_Min": st.column_config.NumberColumn("CIP Duration (Min)", help="Time required for a full Clean-In-Place cycle.", min_value=0),
            "CIP_Circuit": st.column_config.TextColumn("CIP Circuit", help="The CIP circuit this tank is connected to."),
        }
    )

    if st.button("💾 Save Tanks to CSV", use_container_width=True, type="primary"):
        with st.spinner("Saving Tanks..."):
            try:
                # Convert DF to objects for validation
                updated_tanks = {}
                for _, row in edited_df.iterrows():
                    tank_id = row["Tank_ID"]
                    if not tank_id:
                        st.error("Tank ID cannot be empty.")
                        return
                    if tank_id in updated_tanks:
                        st.error(f"Duplicate Tank ID '{tank_id}'. IDs must be unique.")
                        return
                    
                    # Handle comma-separated list for compatible products
                    compat_cats = []
                    if pd.notna(row["Compatible_Product_Categories"]) and row["Compatible_Product_Categories"]:
                        compat_cats = [cat.strip() for cat in str(row["Compatible_Product_Categories"]).split(',')]

                    updated_tanks[tank_id] = Tank(
                        tank_id=str(row["Tank_ID"]),
                        capacity_liters=float(row["Capacity_Liters"]),
                        compatible_product_categories=compat_cats,
                        status=ResourceStatus(row["Status"]),
                        tank_type=TankType(row["Tank_Type"]),
                        current_product_category=str(row["Current_Product_Category"]) if pd.notna(row["Current_Product_Category"]) else None,
                        current_volume_liters=float(row["Current_Volume_Liters"]),
                        cip_duration_minutes=int(row["CIP_Duration_Min"]),
                        cip_circuit=str(row["CIP_Circuit"]) if pd.notna(row["CIP_Circuit"]) else None,
                    )
                
                # Get file path and save
                file_path = data_loader.data_dir / "tank_config.csv"
                df_to_save = pd.DataFrame([tank._to_dict() for tank in updated_tanks.values()])
                df_to_save.to_csv(file_path, index=False)
                
                st.success("✅ Tank configuration saved successfully!")

                # Reload all data to ensure consistency across the app
                data_loader.load_all_data()
                st.session_state.tank_config = config.TANKS.copy()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving Tank data: {e}")