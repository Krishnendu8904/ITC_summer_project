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

from models.data_models import *
import config
from config import *

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and validates all input data files"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.validation_errors = []

    def clear_all_data(self):
        config.LINES.clear()
        config.SKUS.clear()
        config.SHIFTS.clear()
        config.TANKS.clear()
        config.USER_INDENTS.clear()
        config.EQUIPMENTS.clear()
        config.PRODUCTS.clear()
        config.ROOMS.clear()
        config.CIP_CIRCUIT.clear()

    def load_all_data(self):
        """Load all required data files and return structured data"""
        try:
            logger.info("Loading all production data...")
            
            # Load basic resources
            config.SKUS.update(self.load_skus_with_fallback())
            config.LINES.update(self.load_lines_with_fallback())
            config.TANKS.update(self.load_tanks_with_fallback())
            config.SHIFTS.update(self.load_shifts_with_fallback())
            config.USER_INDENTS.update(self.load_user_indents_with_fallback())
            
            # Load new resources
            config.EQUIPMENTS = self.load_equipment_with_fallback()
            config.ROOMS = self.load_rooms_with_fallback()
            config.PRODUCTS = self.load_products_with_fallback()
            config.CIP_CIRCUIT = self.load_cip_circuits_with_fallback()
            
            # Load compatibility and constraints
            self.load_line_sku_compatibility()
            self.load_constraints()
            
            if self.validation_errors:
                logger.warning(f"Data validation warnings: {self.validation_errors}")

            logger.info("All data loaded successfully")
            return

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
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
        config.CIP_CIRCUIT = self._create_sample_cip_circuits()
        
        logger.info("Sample Data Loaded Successfully")

    def load_skus_with_fallback(self) -> Dict[str, SKU]:
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
                    variant=str(row.get('Variant', 'Standard')).strip(),
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
                line = Line(
                    line_id=str(row['Line_ID']),
                    compatible_skus_max_production={},  # Will be populated separately
                    CIP_circuit= str(row.get('CIP_Circuit', '')) if pd.notna(row.get('CIP_Circuit')) else None,
                    cip_duration=int(row.get('CIP_Duration_Min', 60)),
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
                    cip_duration_minutes=int(row.get('CIP_Duration_Min', 60)),
                    cip_circuit=str(row.get('CIP_Circuit', '')) if pd.notna(row.get('CIP_Circuit')) else None
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
                supported_cats = []
                if pd.notna(row.get('Supported_Product_Categories')):
                    supported_cats = str(row['Supported_Product_Categories']).split(',')
                    supported_cats = [cat.strip() for cat in supported_cats]
                
                equip = Equipment(
                    equipment_id=str(row['Equipment_ID']),
                    processing_speed= float(row['Processing Speed'] if pd.notna(row["Processing Speed"]) else 0.0),
                    supported_product_categories=supported_cats,
                    cip_circuit=str(row.get('CIP_Circuit', '')) if pd.notna(row.get('CIP_Circuit')) else None,
                    cip_duration_minutes=int(row.get('CIP_Duration_Min', 0)),
                    status=ResourceStatus(row.get('Status', 'IDLE')),
                    setup_time_minutes=int(row.get('Setup_Time_Min', 0)),
                    current_product_category=str(row.get('Current_Product_Category', '')) if pd.notna(row.get('Current_Product_Category')) else None
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
                    temperature_celsius=float(row.get('Temperature_Celsius')) if pd.notna(row.get('Temperature_Celsius')) else None,
                    humidity_percent=float(row.get('Humidity_Percent')) if pd.notna(row.get('Humidity_Percent')) else None
                )
                
                rooms[room.room_id] = room
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Room row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(rooms)} rooms")
        return rooms

    def load_products_with_fallback(self) -> Dict[str, Product]:
        """Load product definitions with processing steps"""
        df = self._get_csv_or_warn("product_config.csv")
        if df is None or df.empty:
            return self._create_sample_products()
        
        products = {}
        for _, row in df.iterrows():
            try:
                product_category = str(row['Product_Category'])
                if product_category not in products:
                    products[product_category] = Product(product_category=product_category)
                
                # If there are processing steps defined in the row
                if pd.notna(row.get('Step_ID')):
                    step = ProcessingStep(
                        step_id=str(row['Step_ID']),
                        name=str(row.get('Step_Name', '')),
                        resource_type=ResourceType(row.get('Resource_Type', 'EQUIPMENT')),
                        duration_minutes_per_batch=float(row.get('Duration_Minutes_Per_Batch', 60)),
                        min_capacity_required_liters=float(row.get('Min_Capacity_Required_Liters', 0.0)),
                        compatible_resource_ids=str(row.get('Compatible_Resource_IDs', '')).split(',') if pd.notna(row.get('Compatible_Resource_IDs')) else [],
                        requires_setup=bool(row.get('Requires_Setup', False)),
                        requires_cip_after=bool(row.get('Requires_CIP_After', True))
                    )
                    products[product_category].processing_steps.append(step)
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Product row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(products)} product definitions")
        return products

    def load_cip_circuits_with_fallback(self) -> Dict[str, CIP_circuit]:
        """Load CIP circuits"""
        df = self._get_csv_or_warn("cip_circuit_config.csv")
        if df is None or df.empty:
            return self._create_sample_cip_circuits()
        
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
                    standard_cip_duration_minutes=int(row.get('Standard_CIP_Duration_Min', 60))
                )
                
                circuits[circuit.circuit_id] = circuit
                
            except Exception as e:
                self.validation_errors.append(f"Invalid CIP Circuit row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(circuits)} CIP circuits")
        return circuits

    def load_line_sku_compatibility(self):
        """Load line-SKU compatibility with production rates"""
        df = self._get_csv_or_warn("line_sku.csv")
        if df is None or df.empty:
            df = self._create_sample_compatibility()
        
        for _, row in df.iterrows():
            try:
                line_id = str(row['Line_ID']).strip()
                sku_id = str(row['SKU_ID']).strip()
                
                if sku_id not in config.SKUS:
                    self.validation_errors.append(f'SKU ID {sku_id} not found while processing compatibility')
                    continue
                if line_id not in config.LINES:
                    self.validation_errors.append(f'LINE ID {line_id} not found while processing compatibility')
                    continue
    
                max_production_rate = float(row['Max_Production_Rate']) if 'Max_Production_Rate' in row and pd.notna(row['Max_Production_Rate']) else 0.0
                config.LINES[line_id].compatible_skus_max_production[sku_id] = max_production_rate
                
                if max_production_rate == 0:
                    self.validation_errors.append(f'Max Production Rate for {sku_id} for {line_id} is set to 0')
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Line-SKU row: {row.to_dict()} - {str(e)}")
        return

    def load_constraints(self) -> Dict[str, Any]:
        """Load special constraints"""
        file_path = self.data_dir / "spl_constraints.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
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
                    priority=Priority(int(row.get('Priority', DEFAULTS['priority']))),
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
            "CUPSEL200": SKU("CUPSEL200", "CURD", "SELECT_200ML", 0.2),
            "BUCLOF1KG": SKU("BUCLOF1KG", "CURD", "LOW_FAT_1KG", 1.0),
            "PCHSEL400": SKU("PCHSEL400", "CURD", "SELECT_400ML_POUCH", 0.4),
            "CUPMID200": SKU("CUPMID200", "MISHTI_DOI", "TRADITIONAL_200ML", 0.2),
            "CUPSEL400": SKU("CUPSEL400", "CURD", "SELECT_400ML", 0.4)
        }

    def _create_sample_lines(self) -> Dict[str, Line]:
        return {
            "CUPLINE-1": Line(
                line_id="CUPLINE-1",
                compatible_skus_max_production={'CUPSEL200': 5.75, 'CUPSEL400': 11.0},
                cip_duration=90,
                status=ResourceStatus.IDLE,
                setup_time_minutes=45,
                current_product_category="CURD"
            ),
            "MISHTI_LINE-1": Line(
                line_id="MISHTI_LINE-1",
                compatible_skus_max_production={'CUPMID200': 5.25},
                cip_duration=120,
                status=ResourceStatus.IDLE,
                setup_time_minutes=60,
                current_product_category="MISHTI_DOI"
            ),
            "POUCH_PACKING_LINE": Line(
                line_id="POUCH_PACKING_LINE",
                compatible_skus_max_production={'PCHSEL400': 10.0},
                cip_duration=75,
                status=ResourceStatus.IDLE,
                setup_time_minutes=30
            ),
            "BUCKETLINE-1": Line(
                line_id="BUCKETLINE-1",
                compatible_skus_max_production={'BUCLOF1KG': 10.25},
                cip_duration=100,
                status=ResourceStatus.IDLE,
                setup_time_minutes=50,
                current_product_category="CURD"
            )
        }

    def _create_sample_tanks(self) -> Dict[str, Tank]:
        return {
            "LT-1": Tank(
                tank_id="LT-1",
                capacity_liters=5000,
                compatible_product_categories=['CURD'],
                status=ResourceStatus.IDLE,
                tank_type=TankType.PROCESSING,
                cip_duration_minutes=60,
                cip_circuit="CIRCUIT-1"
            ),
            "LT-2": Tank(
                tank_id="LT-2",
                capacity_liters=5000,
                compatible_product_categories=['CURD'],
                status=ResourceStatus.IDLE,
                tank_type=TankType.PROCESSING,
                cip_duration_minutes=60,
                cip_circuit="CIRCUIT-1"
            ),
            "LT-3": Tank(
                tank_id="LT-3",
                capacity_liters=5000,
                compatible_product_categories=['CURD'],
                status=ResourceStatus.IDLE,
                tank_type=TankType.STORAGE,
                cip_duration_minutes=45,
                cip_circuit="CIRCUIT-2"
            ),
            "LT-4": Tank(
                tank_id="LT-4",
                capacity_liters=2000,
                compatible_product_categories=['MISHTI_DOI'],
                status=ResourceStatus.IDLE,
                tank_type=TankType.PROCESSING,
                cip_duration_minutes=90,
                cip_circuit="CIRCUIT-3"
            )
        }

    def _create_sample_shifts(self) -> Dict[str, Shift]:
        base_date = datetime.now().date()
        return {
            "A": Shift(
                shift_id="A",
                start_time=datetime.combine(base_date, datetime.strptime('06:00', '%H:%M').time()),
                end_time=datetime.combine(base_date, datetime.strptime('14:00', '%H:%M').time()),
                is_active=True
            ),
            "B": Shift(
                shift_id="B",
                start_time=datetime.combine(base_date, datetime.strptime('14:00', '%H:%M').time()),
                end_time=datetime.combine(base_date, datetime.strptime('22:00', '%H:%M').time()),
                is_active=True
            ),
            "C": Shift(
                shift_id="C",
                start_time=datetime.combine(base_date, datetime.strptime('22:00', '%H:%M').time()),
                end_time=datetime.combine(base_date + timedelta(1), datetime.strptime('06:00', '%H:%M').time()),
                is_active=True
            )
        }

    def _create_sample_indents(self) -> Dict[str, UserIndent]:
        return {
            "ODR10001": UserIndent(
                order_no="ODR10001",
                sku_id="CUPSEL200",
                qty_required_liters=200.0,  # 1000 units * 0.2L each
                priority=Priority.HIGH,
                due_date=datetime.now() + timedelta(days=1)
            ),
            "ODR10002": UserIndent(
                order_no="ODR10002",
                sku_id="BUCLOF1KG",
                qty_required_liters=3000.0,  # 3000 units * 1.0L each
                priority=Priority.MEDIUM,
                due_date=datetime.now() + timedelta(days=1)
            ),
            "ODR10005": UserIndent(
                order_no="ODR10005",
                sku_id="CUPSEL400",
                qty_required_liters=800.0,  # 2000 units * 0.4L each
                priority=Priority.HIGH,
                due_date=datetime.now() + timedelta(days=1)
            ),
            "ODR10010": UserIndent(
                order_no="ODR10010",
                sku_id="CUPMID200",
                qty_required_liters=300.0,  # 1500 units * 0.2L each
                priority=Priority.LOW,
                due_date=datetime.now() + timedelta(days=1)
            )
        }

    def _create_sample_equipment(self) -> Dict[str, Equipment]:
        return {
            "MIXER-1": Equipment(
                equipment_id="MIXER-1",
                processing_speed= 166.67,
                supported_product_categories=['CURD', 'MISHTI_DOI'],
                cip_circuit="CIRCUIT-1",
                cip_duration_minutes=45,
                status=ResourceStatus.IDLE,
                setup_time_minutes=20
            ),
            "PASTEURIZER-1": Equipment(
                equipment_id="PASTEURIZER-1",
                processing_speed= 166.67,
                supported_product_categories=[],  # Supports all
                cip_circuit="CIRCUIT-2",
                cip_duration_minutes=90,
                status=ResourceStatus.IDLE,
                setup_time_minutes=30
            ),
            "HOMOGENIZER-1": Equipment(
                equipment_id="HOMOGENIZER-1",
                processing_speed= 166.67,
                supported_product_categories=['CURD'],
                cip_circuit="CIRCUIT-1",
                cip_duration_minutes=60,
                status=ResourceStatus.IDLE,
                setup_time_minutes=15
            )
        }

    def _create_sample_rooms(self) -> Dict[str, Room]:
        return {
            "INCUBATOR-1": Room(
                room_id="INCUBATOR-1",
                capacity_units=1000,
                supported_skus=[],  # Supports all
                room_type=RoomType.INCUBATOR,
                status=ResourceStatus.ACTIVE,
                temperature_celsius=42.0,
                humidity_percent=85.0
            ),
            "BLAST_CHILLER-1": Room(
                room_id="BLAST_CHILLER-1",
                capacity_units=500,
                supported_skus=[],
                room_type=RoomType.BLAST_CHILLING,
                status=ResourceStatus.ACTIVE,
                temperature_celsius=2.0
            ),
            "COLD_STORAGE-1": Room(
                room_id="COLD_STORAGE-1",
                capacity_units=2000,
                supported_skus=[],
                room_type=RoomType.STORAGE,
                status=ResourceStatus.ACTIVE,
                temperature_celsius=4.0
            )
        }

    def _create_sample_products(self) -> Dict[str, Product]:
        return {
            "CURD": Product(
                product_category="CURD",
                processing_steps=[
                    ProcessingStep(
                        step_id="CURD_MIXING",
                        name="Milk Mixing",
                        resource_type=ResourceType.EQUIPMENT,
                        duration_minutes_per_batch=30,
                        min_capacity_required_liters=1000,
                        compatible_resource_ids=["MIXER-1"],
                        requires_setup=True,
                        requires_cip_after=True
                    ),
                    ProcessingStep(
                        step_id="CURD_PASTEURIZATION",
                        name="Pasteurization",
                        resource_type=ResourceType.EQUIPMENT,
                        duration_minutes_per_batch=45,
                        min_capacity_required_liters=1000,
                        compatible_resource_ids=["PASTEURIZER-1"],
                        requires_setup=True,
                        requires_cip_after=True
                    ),
                    ProcessingStep(
                        step_id="CURD_INCUBATION",
                        name="Incubation",
                        resource_type=ResourceType.ROOM,
                        duration_minutes_per_batch=480,  # 8 hours
                        compatible_resource_ids=["INCUBATOR-1"],
                        requires_setup=False,
                        requires_cip_after=False
                    ),
                    ProcessingStep(
                        step_id="CURD_PACKAGING",
                        name="Packaging",
                        resource_type=ResourceType.LINE,
                        duration_minutes_per_batch=60,
                        requires_setup=True,
                        requires_cip_after=True
                    )
                ]
            ),
            "MISHTI_DOI": Product(
                product_category="MISHTI_DOI",
                processing_steps=[
                    ProcessingStep(
                        step_id="MISHTI_MIXING",
                        name="Sweetened Milk Mixing",
                        resource_type=ResourceType.EQUIPMENT,
                        duration_minutes_per_batch=45,
                        min_capacity_required_liters=500,
                        compatible_resource_ids=["MIXER-1"],
                        requires_setup=True,
                        requires_cip_after=True
                    ),
                    ProcessingStep(
                        step_id="MISHTI_PASTEURIZATION",
                        name="Pasteurization",
                        resource_type=ResourceType.EQUIPMENT,
                        duration_minutes_per_batch=60,
                        min_capacity_required_liters=500,
                        compatible_resource_ids=["PASTEURIZER-1"],
                        requires_setup=True,
                        requires_cip_after=True
                    ),
                    ProcessingStep(
                        step_id="MISHTI_INCUBATION",
                        name="Sweet Incubation",
                        resource_type=ResourceType.ROOM,
                        duration_minutes_per_batch=360,  # 6 hours
                        compatible_resource_ids=["INCUBATOR-1"],
                        requires_setup=False,
                        requires_cip_after=False
                    ),
                    ProcessingStep(
                        step_id="MISHTI_CHILLING",
                        name="Blast Chilling",
                        resource_type=ResourceType.ROOM,
                        duration_minutes_per_batch=120,  # 2 hours
                        compatible_resource_ids=["BLAST_CHILLER-1"],
                        requires_setup=False,
                        requires_cip_after=False
                    ),
                    ProcessingStep(
                        step_id="MISHTI_PACKAGING",
                        name="Packaging",
                        resource_type=ResourceType.LINE,
                        duration_minutes_per_batch=45,
                        requires_setup=True,
                        requires_cip_after=True
                    )
                ]
            )
        }

    def _create_sample_cip_circuits(self) -> Dict[str, CIP_circuit]:
        return {
            "CIRCUIT-1": CIP_circuit(
                circuit_id="CIRCUIT-1",
                connected_resource_ids=["LT-1", "LT-2", "MIXER-1", "HOMOGENIZER-1"],
                is_available=True,
                standard_cip_duration_minutes=60
            ),
            "CIRCUIT-2": CIP_circuit(
                circuit_id="CIRCUIT-2",
                connected_resource_ids=["LT-3", "PASTEURIZER-1"],
                is_available=True,
                standard_cip_duration_minutes=90
            ),
            "CIRCUIT-3": CIP_circuit(
                circuit_id="CIRCUIT-3",
                connected_resource_ids=["LT-4"],
                is_available=True,
                standard_cip_duration_minutes=75
            )
        }

    def _create_sample_compatibility(self) -> pd.DataFrame:
        """Create sample line-SKU compatibility data"""
        compatibility_data = [
            {"Line_ID": "CUPLINE-1", "SKU_ID": "CUPSEL200", "Max_Production_Rate": 5.75},
            {"Line_ID": "CUPLINE-1", "SKU_ID": "CUPSEL400", "Max_Production_Rate": 11.0},
            {"Line_ID": "MISHTI_LINE-1", "SKU_ID": "CUPMID200", "Max_Production_Rate": 5.25},
            {"Line_ID": "POUCH_PACKING_LINE", "SKU_ID": "PCHSEL400", "Max_Production_Rate": 10.0},
            {"Line_ID": "BUCKETLINE-1", "SKU_ID": "BUCLOF1KG", "Max_Production_Rate": 10.25}
        ]
        return pd.DataFrame(compatibility_data)

    def _get_csv_or_warn(self, filename: str, write: bool = False) -> Optional[pd.DataFrame]:
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