"""
Data Loader for Dairy Scheduler MVP
Handles reading and validating all input CSV/Excel files
Updated to work with new comprehensive data models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from utils.data_models import *
import config
from config import *

logger = logging.getLogger(__name__)



class DataLoader:
    """Loads and validates all input data files"""
    
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = data_dir
        self.validation_errors = []

    def clear_all_data(self):
        config.LINES.clear()
        config.SKUS.clear()
        config.SHIFTS.clear()
        config.TANKS.clear()
        config.EQUIPMENTS.clear()
        config.PRODUCTS.clear()
        config.ROOMS.clear()
        config.CIP_CIRCUIT.clear()

    def load_all_data(self):
        """Load all required data files and return structured data"""
        try:
            logger.info("Loading all production data...")
            
            # Load basic resources
            self.clear_all_data()

            config.PRODUCTS.update(self.load_products_with_fallback())
            config.SKUS.update(self.load_skus_with_fallback())
            config.LINES.update(self.load_lines_with_fallback())
            config.TANKS.update(self.load_tanks_with_fallback())
            config.SHIFTS.update(self.load_shifts_with_fallback())
            config.USER_INDENTS.update(self.load_user_indents_with_fallback())
            
            # Load new resources
            config.EQUIPMENTS.update(self.load_equipment_with_fallback())
            config.ROOMS.update(self.load_rooms_with_fallback())
            config.CIP_CIRCUIT.update(self.load_CIP_circuits_with_fallback())
            
            
            if self.validation_errors:
                logger.warning(f"Data validation warnings: {self.validation_errors}")

            logger.info("All data loaded successfully")
            return

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def load_factory_config(self):
        config.LINES.clear()
        config.SKUS.clear()
        config.SHIFTS.clear()
        config.TANKS.clear()
        config.EQUIPMENTS.clear()
        config.PRODUCTS.clear()
        config.ROOMS.clear()
        config.CIP_CIRCUIT.clear()
        config.PRODUCTS.update(self.load_products_with_fallback())
        config.SKUS.update(self.load_skus_with_fallback())
        config.LINES.update(self.load_lines_with_fallback())
        config.TANKS.update(self.load_tanks_with_fallback())
        config.SHIFTS.update(self.load_shifts_with_fallback())
        config.EQUIPMENTS.update(self.load_equipment_with_fallback())
        config.ROOMS.update(self.load_rooms_with_fallback())
        config.CIP_CIRCUIT.update(self.load_CIP_circuits_with_fallback())

    def _str_to_dict_with_validation(self, s: str, valid_skus: Dict[str, Any], entity_id: str = "N/A") -> Dict[str, float]:
        """
        Converts a semicolon-separated string of SKU:Rate pairs to a dictionary.
        Performs validation for format, SKU existence, and rate validity.
        """
        result = {}
        errors = []
        if not s:
            return result

        pairs = s.split(';')
        for pair in pairs:
            if not pair.strip():
                continue
            parts = pair.split(':', 1)
            if len(parts) != 2:
                errors.append(f"Invalid format '{pair}'. Expected 'SKU_ID:Rate'.")
                continue
            
            
            sku_id = parts[0].strip()
            rate_str = parts[1].strip()

            if not sku_id:
                errors.append(f"Empty SKU ID in pair '{pair}'.")
                continue
            if sku_id not in valid_skus:
                errors.append(f"SKU ID '{sku_id}' not found in SKU configurations.")
                
            try:
                rate = float(rate_str)
                if rate <= 0:
                    errors.append(f"Production rate for SKU '{sku_id}' must be positive (got {rate}).")
            except ValueError:
                errors.append(f"Invalid production rate '{rate_str}' for SKU '{sku_id}'. Expected a number.")
                continue # Don't add to result if rate is invalid

            if sku_id in result:
                errors.append(f"Duplicate SKU ID '{sku_id}' in compatible SKUs list.")
                continue

            # Only add to result if all checks pass
            if not errors: # If no errors for this specific pair
                result[sku_id] = rate
        
        if errors:
            self.validation_errors.append(f"Errors for '{entity_id}' Compatible SKUs Max Production: {'; '.join(errors)}")
        return result
    
    def load_sample_data(self):
        """Load all Sample Data and store in config variables"""
        logger.info("Loading Sample Data... clearing all previous data")
        
        # Clear existing data
        self.clear_all_data()
        
        # Load sample data
        config.SKUS.update(self._create_sample_skus())
        config.LINES.update(self._create_sample_lines())
        config.TANKS.update(self._create_sample_tanks())
        config.SHIFTS.update(self._create_sample_shifts())
        config.USER_INDENTS.update(self._create_sample_indents())
        config.EQUIPMENTS = self._create_sample_equipment()
        config.ROOMS = self._create_sample_rooms()
        config.PRODUCTS = self._create_sample_products()
        print(f"DataLoader: config.PRODUCTS updated. Total products in config: {len(config.PRODUCTS)}") # ADD THIS
        config.CIP_CIRCUIT = self._create_sample_CIP_circuits()
        
        logger.info("Sample Data Loaded Successfully")

    def load_skus_with_fallback(self) -> Dict[str, SKU]:
        config.SKUS.clear()
        """Load SKUs with updated structure"""
        df = self._get_csv_or_warn("sku_config.csv")
        if df is None or df.empty:
            logger.warning("sku_config.csv not found, returning sample data")
            return self._create_sample_skus()
        
        skus = {}
        for _, row in df.iterrows():
            try:
                sku = SKU(
                    sku_id=str(row['SKU_ID']).strip(),
                    product_category=str(row.get('Product_Category', 'CURD')).strip(),
                    inventory_size=float(row.get('Inventory_Size', 1.0))
                )
                
                if sku.sku_id in skus:
                    self.validation_errors.append(f'Duplicate SKU {sku.sku_id} found, latest updated')
                skus[sku.sku_id] = sku

            except Exception as e:
                self.validation_errors.append(f"Invalid SKU row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(skus)} SKUs")
        return skus

    def load_lines_with_fallback(self) -> Dict[str, Line]:
        """Load production lines with updated structure"""
        df = self._get_csv_or_warn("line_config.csv")
        if df is None or df.empty:
            return self._create_sample_lines()
        
        lines = {}
        for _, row in df.iterrows():
            try:
                line_id = str(row["Line_ID"])
                compatible_skus_max_production_str = str(row.get("Compatible SKUs and Max Production", "")).strip()
                # Use the helper function with validation
                compatible_skus_max_production = self._str_to_dict_with_validation(
                    compatible_skus_max_production_str, config.SKUS, line_id
                )
                line = Line(
                    line_id=line_id,
                    compatible_skus_max_production=compatible_skus_max_production, 
                    CIP_duration_minutes=int(row.get('CIP_Duration_Min', 60)),
                    status=ResourceStatus(row.get('Status', 'IDLE')),
                    setup_time_minutes=int(row.get('Setup_Time_Min', 30)),
                    current_sku=str(row.get('Current_SKU', '')) if pd.notna(row.get('Current_SKU')) else None,
                    current_product_category=str(row.get('Current_Product_Category', '')) if pd.notna(row.get('Current_Product_Category')) else None
                )
                
                if line.line_id in lines:
                    self.validation_errors.append(f'Duplicate Line {line.line_id} found, latest taken')
                lines[line.line_id] = line

            except Exception as e:
                self.validation_errors.append(f"Invalid Line row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(lines)} production lines")
        return lines

    def load_tanks_with_fallback(self) -> Dict[str, Tank]:
        """Load storage tanks with updated structure"""
        df = self._get_csv_or_warn("tank_config.csv")
        if df is None or df.empty:
            return self._create_sample_tanks()
        
        tanks = {}
        for _, row in df.iterrows():
            try:
                # Parse compatible product categories
                compatible_cats = []
                if pd.notna(row.get('Compatible_Product_Categories')):
                    compatible_cats = str(row['Compatible_Product_Categories']).split(',')
                    compatible_cats = [cat.strip() for cat in compatible_cats]
                
                tank = Tank(
                    tank_id=str(row['Tank_ID']),
                    capacity_liters=float(row.get('Capacity_Liters', 1000)),
                    compatible_product_categories=compatible_cats,
                    status=ResourceStatus(row.get('Status', 'IDLE')),
                    tank_type=TankType(row.get('Tank_Type', 'PROCESSING')),
                    current_product_category=str(row.get('Current_Product_Category', '')) if pd.notna(row.get('Current_Product_Category')) else None,
                    current_volume_liters=float(row.get('Current_Volume_Liters', 0.0)),
                    CIP_duration_minutes=int(row.get('CIP_Duration_Min', 60))
                )
                
                if tank.tank_id in tanks:
                    self.validation_errors.append(f'Duplicate tank {tank.tank_id} found, recent updated')
                tanks[tank.tank_id] = tank
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Tank row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(tanks)} storage tanks")
        return tanks

    def load_shifts_with_fallback(self) -> Dict[str, Shift]:
        """Load work shifts with updated structure"""
        df = self._get_csv_or_warn("shift_config.csv")
        if df is None or df.empty:
            return self._create_sample_shifts()
        
        shifts = {}
        for _, row in df.iterrows():
            try:
                start_time = self._parse_time(row.get('Start_Time', '08:00'))
                end_time = self._parse_time(row.get('End_Time', '16:00'))
                
                shift = Shift(
                    shift_id=str(row['Shift_ID']),
                    start_time=start_time,
                    end_time=end_time,
                    is_active=bool(row.get('Is_Active', True))
                )
                
                if shift.shift_id in shifts:
                    self.validation_errors.append(f'Duplicate shift data found for {shift.shift_id}, recent updated')
                shifts[shift.shift_id] = shift
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Shift row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(shifts)} work shifts")
        return shifts
    
    def load_equipment_with_fallback(self) -> Dict[str, Equipment]:
        """Load equipment with new structure"""
        df = self._get_csv_or_warn("equipment_config.csv")
        if df is None or df.empty:
            return self._create_sample_equipment()
        
        equipment = {}
        for _, row in df.iterrows():
            try:
                # Parse supported product categories
                supported_prods = []
                if pd.notna(row.get('compatible_product_categories')):
                    supported_p = str(row['compatible_product_categories']).split(',')
                    supported_prods = [cat.strip() for cat in supported_p]
                
                equip = Equipment(
                    equipment_id=str(row['Equipment_ID']),
                    processing_speed= float(row['Processing Speed'] if pd.notna(row["Processing Speed"]) else 0.0),
                    compatible_product_categories=supported_prods,
                    CIP_duration_minutes=int(row.get('CIP_Duration_Min', 0)),
                    status=ResourceStatus(row.get('Status', 'IDLE')),
                    setup_time_minutes=int(row.get('Setup_Time_Min', 0)),
                    current_product_category=str(row.get('Current_Product_Category', '')) if pd.notna(row.get('Current_Product_Category')) else None,
                    capacity_type=int(row.get('Capacity_Time', 1))
                )
                
                equipment[equip.equipment_id] = equip
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Equipment row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(equipment)} equipment items")
        return equipment

    def load_rooms_with_fallback(self) -> Dict[str, Room]:
        """Load rooms with new structure"""
        df = self._get_csv_or_warn("room_config.csv")
        if df is None or df.empty:
            return self._create_sample_rooms()
        
        rooms = {}
        for _, row in df.iterrows():
            try:
                # Parse supported SKUs
                supported_skus = []
                if pd.notna(row.get('Supported_SKUs')):
                    supported_skus = str(row['Supported_SKUs']).split(',')
                    supported_skus = [sku.strip() for sku in supported_skus]
                
                room = Room(
                    room_id=str(row['Room_ID']),
                    capacity_units=float(row.get('Capacity_Units', 100)),
                    supported_skus=supported_skus,
                    room_type=RoomType(row.get('Room_Type', 'STORAGE')),
                    current_occupancy_units=float(row.get('Current_Occupancy_Units', 0.0)),
                    status=ResourceStatus(row.get('Status', 'ACTIVE')),
                )
                
                rooms[room.room_id] = room
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Room row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(rooms)} rooms")
        return rooms

    def load_products_with_fallback(self) -> Dict[str, Product]:
        """Load product definitions with processing steps"""
        df = self._get_csv_or_warn("product_config.csv")
        print(df)
        if df is None or df.empty:
            return self._create_sample_products()
        
        products = {}
        
        for _, row in df.iterrows():
            try:
                product_category = str(row['Product_Category'])
                
                # Initialize product if not exists
                if product_category not in products:
                    max_batch_size = row.get('Max_Batch_Size', 1000.0)  # Default if not specified
                    products[product_category] = Product(
                        product_category=product_category,
                        max_batch_size=float(max_batch_size) if pd.notna(max_batch_size) else 1000.0
                    )
                
                # If there are processing steps defined in the row
                if pd.notna(row.get('Step_ID')):
                    step_id = str(row['Step_ID'])
                    
                    # Check if this step already exists in the product
                    existing_step = None
                    for step in products[product_category].processing_steps:
                        if step.step_id == step_id:
                            existing_step = step
                            break
                    
                    # Create resource requirement from this row
                    resource_requirement = None
                    if pd.notna(row.get('Resource_Type')):
                        resource_type = ResourceType(row.get('Resource_Type'))
                        compatible_ids_str = row.get('Compatible_Resource_IDs', '')
                        compatible_ids = []
                        if pd.notna(compatible_ids_str) and str(compatible_ids_str).strip():
                            compatible_ids = [id.strip() for id in str(compatible_ids_str).split(',') if id.strip()]
                        
                        resource_requirement = ResourceRequirement(
                            resource_type=resource_type,
                            compatible_ids=compatible_ids
                        )
                    
                    if existing_step:
                        # Add resource requirement to existing step
                        if resource_requirement and resource_requirement not in existing_step.requirements:
                            existing_step.requirements.append(resource_requirement)
                    else:
                        # Create new step
                        requirements = [resource_requirement] if resource_requirement else []
                        
                        step = ProcessingStep(
                            step_id=step_id,
                            name=str(row.get('Step_Name', '')),
                            duration_minutes=float(row.get('Duration_Minutes_Per_Batch', 60)),
                            min_capacity_required_liters=float(row.get('Min_Capacity_Required_Liters', 0.0)),
                            requires_CIP_after=bool(row.get('Requires_CIP_After', True)),
                            process_type= row.get('ProcessType', ProcessType.PROCESSING.value), 
                            requirements=requirements
                        )
                        products[product_category].processing_steps.append(step)
                        
            except Exception as e:
                self.validation_errors.append(f"Invalid Product row: {row.to_dict()} - {str(e)}")
                logger.error(f"Error processing product row: {e}")
        
        logger.info(f"Loaded {len(products)} product definitions")
        print(products)
        return products
    
    def load_CIP_circuits_with_fallback(self) -> Dict[str, CIP_circuit]:
        """Load CIP circuits"""
        df = self._get_csv_or_warn("CIP_circuit_config.csv")
        if df is None or df.empty:
            return self._create_sample_CIP_circuits()
        
        circuits = {}
        for _, row in df.iterrows():
            try:
                # Parse connected resource IDs
                connected_resources = []
                if pd.notna(row.get('Connected_Resource_IDs')):
                    connected_resources = str(row['Connected_Resource_IDs']).split(',')
                    connected_resources = [res.strip() for res in connected_resources]
                
                circuit = CIP_circuit(
                    circuit_id=str(row['Circuit_ID']),
                    connected_resource_ids=connected_resources,
                    is_available=bool(row.get('Is_Available', True)),
                )
                
                circuits[circuit.circuit_id] = circuit
                
            except Exception as e:
                self.validation_errors.append(f"Invalid CIP Circuit row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(circuits)} CIP circuits")
        return circuits


    def load_constraints(self) -> Dict[str, Any]:
        """Load special constraints"""
        file_path = self.data_dir / "spl_constraints.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if not df:
                pass
            return df.to_dict('records')
        return {}

    def load_user_indents_with_fallback(self) -> Dict[str, UserIndent]:
        """Load user indents with updated structure"""
        update = False
        df = self._get_csv_or_warn("user_indent.csv")
        if df is None or df.empty:
            return self._create_sample_indents()
        
        indents = {}
        for idx, row in df.iterrows():
            try:
                if "Due_Date" in row and pd.notna(row['Due_Date']):
                    due_date = pd.to_datetime(row['Due_Date'])
                else:
                    due_date = datetime.today()
                    df.at[idx, "Due_Date"] = due_date
                    update = True
                
                indent = UserIndent(
                    order_no=str(row.get('Order_Number', '')).strip(),
                    sku_id=str(row['SKU_ID']).strip(),
                    qty_required_liters=float(row['Qty_Required_Liters']),
                    priority=Priority(int(row.get('Priority'))),
                    due_date=due_date
                )
                
                if indent.order_no in indents:
                    self.validation_errors.append(f'Duplicate indent for order no {indent.order_no}, updated to recent')
                indents[indent.order_no] = indent
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Indent row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(indents)} user indents")
        if update:
            df.to_csv(self._get_csv_or_warn("user_indent.csv", write=True), index=False)
        return indents

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime object"""
        try:
            base_date = datetime.now().date()
            time_obj = datetime.strptime(time_str.strip(), '%H:%M').time()
            return datetime.combine(base_date, time_obj)
        except ValueError:
            return pd.to_datetime(time_str)

    # Sample data creation methods
    def _create_sample_skus(self) -> Dict[str, SKU]:
        return {
        "SEL-BKT-1KG": SKU(sku_id="SEL-BKT-1KG", product_category="SELECT-CURD", inventory_size=0.521),
        "SEL-BKT-2KG": SKU(sku_id="SEL-BKT-2KG", product_category="SELECT-CURD", inventory_size=0.694),
        "SEL-BKT-5KG": SKU(sku_id="SEL-BKT-5KG", product_category="SELECT-CURD", inventory_size=1.428),
        "SEL-BKT-15KG": SKU(sku_id="SEL-BKT-15KG", product_category="SELECT-CURD", inventory_size=1.33),
        "SEL-CUP-200G": SKU(sku_id="SEL-CUP-200G", product_category="SELECT-CURD", inventory_size=0.595),
        "SEL-CUP-400G": SKU(sku_id="SEL-CUP-400G", product_category="SELECT-CURD", inventory_size=0.595),
        "LFT-BKT-1KG": SKU(sku_id="LFT-BKT-1KG", product_category="LOW-FAT-CURD", inventory_size=1.0),
        "LFT-BKT-2KG": SKU(sku_id="LFT-BKT-2KG", product_category="LOW-FAT-CURD", inventory_size=1.0),
        "LFT-BKT-5KG": SKU(sku_id="LFT-BKT-5KG", product_category="LOW-FAT-CURD", inventory_size=1.0),
        "LFT-BKT-15KG": SKU(sku_id="LFT-BKT-15KG", product_category="LOW-FAT-CURD", inventory_size=1.0),
        "LFT-CUP-200G": SKU(sku_id="LFT-CUP-200G", product_category="LOW-FAT-CURD", inventory_size=1.0),
        "LFT-CUP-400G": SKU(sku_id="LFT-CUP-400G", product_category="LOW-FAT-CURD", inventory_size=1.0),
        "LFT-PCH-CRD-200G": SKU(sku_id="LFT-PCH-CRD-200G", product_category="LFT-POUCH-CURD", inventory_size=1.0),
        "LFT-PCH-CRD-400G": SKU(sku_id="LFT-PCH-CRD-400G", product_category="LFT-POUCH-CURD", inventory_size=1.0),
        "LFT-PCH-CRD-1KG": SKU(sku_id="LFT-PCH-CRD-1KG", product_category="LFT-POUCH-CURD", inventory_size=1.0),
        "PLN-PCH-CRD-200G": SKU(sku_id="PLN-PCH-CRD-200G", product_category="PLN-POUCH-CURD", inventory_size=0.595),
        "PLN-PCH-CRD-400G": SKU(sku_id="PLN-PCH-CRD-400G", product_category="PLN-POUCH-CURD", inventory_size=0.595),
        "PLN-PCH-CRD-1KG": SKU(sku_id="PLN-PCH-CRD-1KG", product_category="PLN-POUCH-CURD", inventory_size=0.714),
        "ROS-LSI-170G": SKU(sku_id="ROS-LSI-170G", product_category="ROSE-LASSI", inventory_size=1.0),
        "MNG-LSI-170G": SKU(sku_id="MNG-LSI-170G", product_category="MANGO-LASSI", inventory_size=1.0),
        "SHI-LSI-170G": SKU(sku_id="SHI-LSI-170G", product_category="SHAHI-LASSI", inventory_size=1.0)
    }

    
    def _create_sample_lines(self) -> Dict[str, Line]:
        config.LINES.clear()
        return {
                "BUCKET-LINE-1": Line(
                    line_id="BUCKET-LINE-1",
                    compatible_skus_max_production={
                        "SEL-BKT-1KG": 10.35,
                        "SEL-BKT-2KG": 20.2,
                        "LFT-BKT-1KG": 10.35,
                        "LFT-BKT-2KG": 20.2
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                ),
                "BUCKET-LINE-2": Line(
                    line_id="BUCKET-LINE-2",
                    compatible_skus_max_production={
                        "SEL-BKT-5KG": 5.0,
                        "SEL-BKT-15KG": 15.0,
                        "LFT-BKT-5KG": 5.0,
                        "LFT-BKT-15KG": 15.0
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                ),
                "CUP-LINE-1": Line(
                    line_id="CUP-LINE-1",
                    compatible_skus_max_production={
                        "SEL-CUP-200G": 5.75,
                        "SEL-CUP-400G": 10.5,
                        "LFT-CUP-200G": 5.75,
                        "LFT-CUP-400G": 10.5
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                ),
                "LASSI-LINE-1": Line(
                    line_id="LASSI-LINE-1",
                    compatible_skus_max_production={
                        "ROS-LSI-170G": 22.0,
                        "MNG-LSI-170G": 22.0,
                        "SHI-LSI-170G": 22.0
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                ),
                "LASSI-LINE-2": Line(
                    line_id="LASSI-LINE-2",
                    compatible_skus_max_production={
                        "ROS-LSI-170G": 22.0,
                        "MNG-LSI-170G": 22.0,
                        "SHI-LSI-170G": 22.0
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                ),
                "POUCH-LINE-1": Line(
                    line_id="POUCH-LINE-1",
                    compatible_skus_max_production={
                        "PLN-PCH-CRD-1KG": 22.0,
                        "PLN-PCH-CRD-400G": 11.5,
                        "PLN-PCH-CRD-200G": 5.25
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                ),
                "POUCH-LINE-2": Line(
                    line_id="POUCH-LINE-2",
                    compatible_skus_max_production={
                        "PLN-PCH-CRD-1KG": 22.0,
                        "PLN-PCH-CRD-400G": 11.5,
                        "PLN-PCH-CRD-200G": 5.25
                    },
                    setup_time_minutes=30,
                    CIP_duration_minutes=90
                )
            }


    def _create_sample_tanks(self) -> Dict[str, Tank]:
        return {
            "LT-1": Tank(tank_id= "LT-1",capacity_liters= 5000,CIP_duration_minutes= 90),
            "LT-2": Tank(tank_id= "LT-2",capacity_liters= 5000,CIP_duration_minutes= 90),
            "LT-3": Tank(tank_id= "LT-3",capacity_liters= 5000,CIP_duration_minutes= 90),
            "LT-4": Tank(tank_id= "LT-4",capacity_liters= 5000,CIP_duration_minutes= 90),
            "LT-5": Tank(tank_id= "LT-5",capacity_liters= 5000,CIP_duration_minutes= 90),
            "LT-6": Tank(tank_id= "LT-5",capacity_liters= 5000,CIP_duration_minutes= 90),
            "MST-1": Tank(tank_id= "MST-1",capacity_liters= 10000,CIP_duration_minutes= 90),
            "MST-2": Tank(tank_id= "MST-2",capacity_liters= 10000,CIP_duration_minutes= 90),
            "MST-3": Tank(tank_id= "MST-3",capacity_liters= 10000,CIP_duration_minutes= 90)
        }

    def _create_sample_shifts(self) -> Dict[str, Shift]:
        base_date = datetime.now().date()
        return {
            "A": Shift(
                shift_id="A",
                start_time=datetime.strptime('06:00', '%H:%M').time(),
                end_time=datetime.strptime('14:00', '%H:%M').time(),
                is_active=True
            ),
            "B": Shift(
                shift_id="B",
                start_time=datetime.strptime('14:00', '%H:%M').time(),
                end_time=datetime.strptime('22:00', '%H:%M').time(),
                is_active=True
            ),
            "C": Shift(
                shift_id="C",
                start_time=datetime.strptime('22:00', '%H:%M').time(),
                end_time=datetime.strptime('06:00', '%H:%M').time(),
                is_active=True
            )
        }

    def _create_sample_indents(self) -> Dict[str, UserIndent]:
        start_date = datetime.now().replace(minute=0, hour=20, second=0, microsecond=0)

        return {
            # Test Case 1 & 2: Multiple SKUs for one product (Yogurt) needing batching,
            # plus same SKU with different priorities/due dates.
            #"ORD_101": UserIndent(order_no= "ORD_101",sku_id= "SEL-BKT-1KG",qty_required_liters= 0,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            #"ORD_102": UserIndent(order_no= "ORD_102",sku_id= "SEL-BKT-2KG",qty_required_liters= 0,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            "ORD_105": UserIndent(order_no= "ORD_105",sku_id= "SEL-BKT-5KG",qty_required_liters= 2000,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            "ORD_115": UserIndent(order_no= "ORD_115",sku_id= "SEL-BKT-15KG",qty_required_liters= 3000,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            "ORD_122": UserIndent(order_no= "ORD_122",sku_id= "SEL-CUP-200G",qty_required_liters= 1000,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            #"ORD_124": UserIndent(order_no= "ORD_124",sku_id= "SEL-CUP-400G",qty_required_liters= 1000,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            #"ORD_317": UserIndent(order_no= "ORD_317",sku_id= "SHI-LSI-170G",qty_required_liters= 5000,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            "ORD_205": UserIndent(order_no= "ORD_205",sku_id= "PLN-PCH-CRD-500G",qty_required_liters= 3600,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1)),
            "ORD_210": UserIndent(order_no= "ORD_210",sku_id= "PLN-PCH-CRD-1KG",qty_required_liters= 4000,priority= Priority.MEDIUM,due_date= start_date + timedelta(minutes=0,hours=18,days=1))
        }

    def _create_sample_equipment(self) -> Dict[str, Equipment]:
        return {
            "PST-1": Equipment(equipment_id= "PST-1",processing_speed= 85.0, setup_time_minutes= 30, CIP_duration_minutes= 60, capacity_type=CapacityType.BATCH),
            "PST-2": Equipment(equipment_id= "PST-2",processing_speed= 85.0, setup_time_minutes= 30, CIP_duration_minutes= 60, capacity_type=CapacityType.BATCH),
            "CRD-HTR-1": Equipment(equipment_id= "CRD-HTR-1",processing_speed= 25.0, setup_time_minutes= 30, CIP_duration_minutes= 60, capacity_type=CapacityType.SHARED_BY_CATEGORY),
            "THERMISER": Equipment(equipment_id= "THERMISER",processing_speed= 35.0, setup_time_minutes= 30, CIP_duration_minutes= 60, capacity_type=CapacityType.BATCH)
        }

    def _create_sample_rooms(self) -> Dict[str, Room]:
        return { 
            "INCUBATION-1": Room(room_id= "INCUBATION-1",capacity_units= 5000.0, room_type= RoomType.INCUBATOR),
            "INCUBATION-2": Room(room_id= "INCUBATION-2",capacity_units= 5000.0, room_type= RoomType.INCUBATOR),
            "CHILLER-1": Room(room_id= "CHILLER-1",capacity_units= 5000.0, room_type= RoomType.BLAST_CHILLING),
            "CHILLER-2": Room(room_id= "CHILLER-2",capacity_units= 5000.0, room_type= RoomType.BLAST_CHILLING)
                 }

    def _create_sample_products(self) -> Dict[str, Product]:
        return {
    "SELECT-CURD": Product(product_category="SELECT-CURD", processing_steps=[
        ProcessingStep(
            step_id="SEL-CRD-MST-STANDARDISATION",
            name="MST Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=360
        ),
        ProcessingStep(
            step_id="SEL-CRD-PAST",
            name="Pasteurisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60,
            scheduling_rule=SchedulingRule.ZERO_STAGNATION
        ),
        ProcessingStep(
            step_id="SEL-CRD-LT-STANDARDISATION-INNOCULATION",
            name="LT Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-1", "LT-2", "LT-3" ])],
            duration_minutes=180
        ),
        ProcessingStep(
            step_id="SEL-CRD-PACKING",
            name="Packing",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["BUCKET-LINE-1","BUCKET-LINE-2","CUP-LINE-1","CUP-LINE-2"])],
            duration_minutes=60,
            scheduling_rule=SchedulingRule.ZERO_STAGNATION
        ),
        ProcessingStep(
            step_id="SEL-CRD-INCUBATION",
            name="Incubation",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["INCUBATION-1", "INCUBATION-2"])],
            duration_minutes=360,
            scheduling_rule=SchedulingRule.DEFAULT
        ),
        ProcessingStep(
            step_id="SEL-CRD-BLAST-CHILLING",
            name="Chilling",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=120,
            scheduling_rule=SchedulingRule.DEFAULT
        )
    ]),

    "LOW-FAT-CURD": Product(product_category="LOW-FAT-CURD", processing_steps=[
        ProcessingStep(
            step_id="LF-CRD-MST-STANDARDISATION",
            name="MST Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=240
        ),
        ProcessingStep(
            step_id="LF-CRD-PAST",
            name="Pasteurisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="LF-CRD-LT-STANDARDISATION-INNOCULATION",
            name="LT Standardisation",
            process_type=ProcessType.PROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-1", "LT-2",  "LT-3"])],
            duration_minutes=120
        ),
        ProcessingStep(
            step_id="LF-CRD-PACKING",
            name="Packing",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["BUCKET-LINE-1","BUCKET-LINE-2","CUP-LINE-1","CUP-LINE-2"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="LF-CRD-INCUBATION",
            name="Incubation",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["INCUBATION-1", "INCUBATION-2"])],
            duration_minutes=360,
            scheduling_rule=SchedulingRule.DEFAULT
        ),
        ProcessingStep(
            step_id="LF-CRD-BLAST-CHILLING",
            name="Chilling",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=120,
            scheduling_rule=SchedulingRule.DEFAULT
        )
    ]),

    "PLN-POUCH-CURD": Product(product_category="PLN-POUCH-CURD", processing_steps=[
        ProcessingStep(
            step_id="PLN-CRD-MST-STANDARDISATION",
            name="MST Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=240
        ),
        ProcessingStep(
            step_id="PLN-CRD-PAST",
            name="Pasteurisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60,
            scheduling_rule=SchedulingRule.ZERO_STAGNATION
        ),
        ProcessingStep(
            step_id="PLN-CRD-LT-STANDARDISATION-INNOCULATION",
            name="LT Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-1", "LT-2",  "LT-3"])],
            duration_minutes=120
        ),
        ProcessingStep(
            step_id="PLN-CRD-POUCH-PACKING",
            name="Packing",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["POUCH-LINE-1", "POUCH-LINE-2"])],
            duration_minutes=60,
            scheduling_rule=SchedulingRule.ZERO_STAGNATION
        ),
        ProcessingStep(
            step_id="PLN-CRD-POUCH-INCUBATION",
            name="Incubation",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["INCUBATION-1", "INCUBATION-2"])],
            duration_minutes=360,
            scheduling_rule=SchedulingRule.DEFAULT
        ),
        ProcessingStep(
            step_id="PLN-CRD-POUCH-BLAST-CHILLING",
            name="Chilling",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=120,
            scheduling_rule=SchedulingRule.DEFAULT
        )
    ]),

    "LFT-POUCH-CURD": Product(product_category="LFT-POUCH-CURD", processing_steps=[
        ProcessingStep(
            step_id="LFT-CRD-MST-STANDARDISATION",
            name="MST Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=240
        ),
        ProcessingStep(
            step_id="LFT-CRD-PAST",
            name="Pasteurisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="LFT-CRD-LT-STANDARDISATION-INNOCULATION",
            name="LT Standardisation",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-1", "LT-2", ])],
            duration_minutes=120
        ),
        ProcessingStep(
            step_id="LFT-CRD-POUCH-PACKING",
            name="Packing",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["POUCH-LINE-1", "POUCH-LINE-2"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="LFT-CRD-POUCH-INCUBATION",
            name="Incubation",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["INCUBATION-1", "INCUBATION-2"])],
            duration_minutes=360,
            scheduling_rule=SchedulingRule.DEFAULT
        ),
        ProcessingStep(
            step_id="LFT-CRD-POUCH-BLAST-CHILLING",
            name="Chilling",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.ROOM, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=120,
            scheduling_rule=SchedulingRule.DEFAULT
        )
    ]),

    "ROSE-LASSI": Product(product_category="ROSE-LASSI", processing_steps=[
        ProcessingStep(
            step_id="ROSE-LASSI-MST-STANDARDISATION",
            name="ROSE-LASSI-STD",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=240
        ),
        ProcessingStep(
            step_id="ROSE-LASSI-PST",
            name="ROSE-LASSI-PST",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="ROSE-LASSI-INNOC-INCUB",
            name="ROSE-LASSI-INNOC-INCUB",
            process_type=ProcessType.PROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-4", "LT-5", "LT-6"])],
            duration_minutes=540
        ),
        ProcessingStep(
            step_id="ROSE-LASSI-PACK",
            name="ROSE-LASSI-PACK",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["LASSI-LINE-1"])],
            duration_minutes=300
        ),
        ProcessingStep(
            step_id="ROSE-LASSI-CHILLING",
            name="ROSE-LASSI-CHILLING",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=120
        )
    ]),

    "MANGO-LASSI": Product(product_category="MANGO-LASSI", processing_steps=[
        ProcessingStep(
            step_id="MANGO-LASSI-MST-STANDARDISATION",
            name="MANGO-LASSI-STD",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=240
        ),
        ProcessingStep(
            step_id="MANGO-LASSI-PST",
            name="MANGO-LASSI-PST",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="MANGO-LASSI-INNOC-INCUB",
            name="MANGO-LASSI-INNOC-INCUB",
            process_type=ProcessType.PROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-4", "LT-5", "LT-6"])],
            duration_minutes=540
        ),
        ProcessingStep(
            step_id="MANGO-LASSI-PACK",
            name="MANGO-LASSI-PACK",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["LASSI-LINE-1"])],
            duration_minutes=300
        ),
        ProcessingStep(
            step_id="MANGO-LASSI-CHILLING",
            name="MANGO-LASSI-CHILLING",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=300
        )
    ]),

    "SHAHI-LASSI": Product(product_category="SHAHI-LASSI", processing_steps=[
        ProcessingStep(
            step_id="SHAHI-LASSI-MST-STANDARDISATION",
            name="SHAHI-LASSI-STD",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["MST-1", "MST-2", "MST-3"])],
            duration_minutes=240
        ),
        ProcessingStep(
            step_id="SHAHI-LASSI-PST",
            name="SHAHI-LASSI-PST",
            process_type=ProcessType.PREPROCESSING,
            requirements=[ResourceRequirement(ResourceType.EQUIPMENT, ["PST-1"])],
            duration_minutes=60
        ),
        ProcessingStep(
            step_id="SHAHI-LASSI-INNOC-INCUB",
            name="SHAHI-LASSI-INNOC-INCUB",
            process_type=ProcessType.PROCESSING,
            requirements=[ResourceRequirement(ResourceType.TANK, ["LT-4", "LT-5", "LT-6"])],
            duration_minutes=540
        ),
        ProcessingStep(
            step_id="SHAHI-LASSI-PACK",
            name="SHAHI-LASSI-PACK",
            process_type=ProcessType.PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["LASSI-LINE-1"])],
            duration_minutes=300
        ),
        ProcessingStep(
            step_id="SHAHI-LASSI-CHILLING",
            name="SHAHI-LASSI-CHILLING",
            process_type=ProcessType.POST_PACKAGING,
            requirements=[ResourceRequirement(ResourceType.LINE, ["CHILLER-1", "CHILLER-2"])],
            duration_minutes=120
        )
    ])
}
    
    def _create_sample_CIP_circuits(self) -> Dict[str, CIP_circuit]:
        return {
    "CIP_SYS_1": CIP_circuit(
        circuit_id="CIP_SYS_1",
        connected_resource_ids=[
            "LT-1", "LT-2", "LT-3",
            "BUCKET-LINE-1", "BUCKET-LINE-2",
            "CUP-LINE-1", "LT-4", "LT-5", "LT-6",
            "MST-1", "MST-2", "MST-3"
        ]
    ),
    "CIP_SYS_2": CIP_circuit(
        circuit_id="CIP_SYS_2",
        connected_resource_ids=[
            "CRD-HTR-1", "PST-1", "PST-2",
            "THERMISER", "POUCH-LINE-1", "POUCH-LINE-2",
            "LASSI-LINE-1", "LASSI-LINE-2"
        ]
    )
}

    def _create_sample_compatibility(self) -> pd.DataFrame:
        pass

    def _get_csv_or_warn(self, filename: str, write: bool = False) -> Optional[Union[Path, pd.DataFrame]]:
        """Helper method to load CSV with error handling"""
        file_path = self.data_dir / filename
        
        if write:
            return file_path
            
        if not file_path.exists():
            logger.warning(f"File {filename} not found at {file_path}")
            return None
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logger.warning(f"File {filename} is empty")
                return None
            return df
        except Exception as e:
            logger.error(f"Error reading {filename}: {str(e)}")
            return None