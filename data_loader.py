"""
Data Loader for Dairy Scheduler MVP
Handles reading and validating all input CSV/Excel files
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

    def load_all_data(self):
        """Load all required data files and return structured data"""
        try:
            logger.info("Attempting to load ")
            config.SKUS.update(self.load_skus_with_fallback())
            config.LINES.update(self.load_lines_with_fallback())
            config.TANKS.update(self.load_tanks_with_fallback())
            config.SHIFTS.update(self.load_shifts_with_fallback())
            config.USER_INDENTS.update(self.load_user_indents_with_fallback())
            self.load_line_sku_compatibility()
            self.load_constraints()
            
            if self.validation_errors:
                logger.warning(f"Data validation warnings: {self.validation_errors}")

            return

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def load_sample_data(self):
        '''Load all Sample Data and store in config variables'''
        logger.info("Loading Sample Data ... clearing all previous data")
        config.SKUS.clear()
        config.SKUS.update(self._create_sample_skus())
        config.LINES.clear()
        config.LINES.update(self._create_sample_lines())
        config.TANKS.clear()
        config.TANKS.update(self._create_sample_tanks())
        config.SHIFTS.clear()
        config.SHIFTS.update(self._create_sample_shifts())
        config.USER_INDENTS.clear()
        config.USER_INDENTS.update(self._create_sample_indents())
        logger.info("Sample Data Loaded Successfully")


    def load_skus_with_fallback(self) -> Dict[str, SKU]:
        df = self._get_csv_or_warn("sku_config.csv")
        if df is None or df.empty:
            logger.warning("sku_config.csv not found, returning sample data")
            return self._create_sample_skus()
        skus = {}
        for _, row in df.iterrows():
            try:
                sku = SKU(
                    sku_id=str(row['SKU_ID']).strip(),
                    product_category=ProductTypeRegistry.get_name(row.get('Product_Type', 'Curd')),
                    variant=str(row.get('Variant', 'Select')),
                    setup_time=int(row.get('Setup Time', DEFAULTS['setup_time'])),
                    inventory_units=float(row.get('EUI', 0.0))
                )
                if sku.sku_id in skus:
                    self.validation_errors.append(f'Duplicate SKU {sku.sku_id} found, latest updated')
                skus[sku.sku_id] = sku

            except Exception as e:
                self.validation_errors.append(f"Invalid SKU row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(skus)} SKUs")
        return skus

    def load_lines_with_fallback(self) -> Dict[str, Line]:
        df = self._get_csv_or_warn("line_config.csv")
        if df is None or df.empty:
            return self._create_sample_lines()
        
        lines = {}
        for _, row in df.iterrows():
            try:
                line = Line(
                    line_id=str(row['Line_ID']),
                    cip_circuit=str(row.get('CIP circuit', None)),
                    status=LineStatus(row.get('Active_Status', 'Active'))
                )
                if line.line_id in lines:
                    self.validation_errors.append(f'Duplicate Line {line.line_id} found, latest taken')
                lines[line.line_id] = line

            except Exception as e:
                self.validation_errors.append(f"Invalid Line row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(lines)} production lines")
        return lines

    def load_tanks_with_fallback(self) -> Dict[str, Tank]:
        df = self._get_csv_or_warn("tank_config.csv")
        if df is None or df.empty:
            return self._create_sample_tanks()
        tanks = {}
        for _, row in df.iterrows():
            try:
                tank = Tank(
                    tank_id=str(row['Tank_ID']),
                    capacity_liters=float(row.get('Capacity_Liters', 1000)),
                    available=bool(row.get('Available', True))
                )
                if tank.tank_id in tanks:
                    self.validation_errors.append(f'Duplicate tank {tank.tank_id} found, recent updated')
                tanks[tank.tank_id] = tank
            except Exception as e:
                self.validation_errors.append(f"Invalid Tank row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(tanks)} storage tanks")
        return tanks

    def load_shifts_with_fallback(self) -> Dict[str, Shift]:
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
                    active=bool(row.get('Active', True))
                )
                if shift.shift_id in shifts:
                    self.validation_errors.append(f'Duplicate shift data found for {shift.shift_id}, recent updated')
                shifts[shift.shift_id] = shift
            except Exception as e:
                self.validation_errors.append(f"Invalid Shift row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(shifts)} work shifts")
        return shifts
    
    def load_user_indents_with_fallback(self) -> Dict[str, UserIndent]:
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
                    sku_id=str(row['SKU_ID']).strip(),
                    qty_required=float(row['Qty_Required']),
                    priority=Priority(int(row.get('Priority', DEFAULTS['priority']))),
                    due_date=due_date,
                    order_no=str(row.get('Order_Number', None)).strip()
                )
                if indent.order_no in indents:
                    self.validation_errors.append(f'Duplicate indent for order no {indent.order_no}, updated to recent')
                indents[indent.order_no] = indent
            except Exception as e:
                self.validation_errors.append(f"Invalid Indent row: {row.to_dict()} - {str(e)}")

        logger.info(f"Loaded {len(indents)} user indents")
        if update:
            df.to_csv(self._get_csv_or_warn("user_indent.csv", write= True), index=False)
        return indents

    def load_line_sku_compatibility(self):
        df = self._get_csv_or_warn("line_sku.csv")
        if not df:
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
    
                config.LINES[line_id].compatible_skus_max_production[sku_id] = float(row['Max Production Rate']) if 'Max Production Rate' in row and pd.notna(row['Max Production Rate']) else 0.0
                if config.LINES[line_id].compatible_skus_max_production[sku_id] == 0:
                    self.validation_errors.append(f'Max Production Rate for {sku_id} for {line_id} is set to 0')
                
            except Exception as e:
                self.validation_errors.append(f"Invalid Line-SKU row: {row.to_dict()} - {str(e)}")
        return

    def load_constraints(self) -> Dict[str, Any]:
        file_path = self.data_dir / "spl_constraints.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        return {}

    def _parse_time(self, time_str: str) -> datetime:
        try:
            base_date = datetime.now().date()
            time_obj = datetime.strptime(time_str.strip(), '%H:%M').time()
            return datetime.combine(base_date, time_obj)
        except ValueError:
            return pd.to_datetime(time_str)

    def _create_sample_skus(self) -> Dict[str, SKU]:
        return {
            "CUPSEL200": SKU("CUPSEL200", "CURD", "SELECT", 30, 1),
            "BUCLOF1KG": SKU("BUCLOF1KG", "CURD", "LOW_FAT", 30, 1.5),
            "PCHSEL400": SKU("PCHSEL400", "POUCH_CURD", "SELECT", 30, 1),
            "CUPMID200": SKU("CUPMID200", "MISHTI_DOI", "TRADITIONAL", 30, 1.1),
            "CUPSEL400": SKU("CUPSEL400", "CURD", "SELECT", 30, 1)

        }

    def _create_sample_lines(self) -> Dict[str, Line]:
        return {
            "CUPLINE-1": Line("CUPLINE-1", "CIRCUIT-2", LineStatus.ACTIVE, {'CUPSEL200': 5.75, 'CUPSEL400': 11}),
            "MISHTI_LINE-1": Line("MISHTI_LINE-1", "CIRCUIT-5", LineStatus.ACTIVE, {'CUPMID200': 5.25}),
            "POUCH_PACKING_LINE": Line("POUCH_PACKING_LINE", "CIRCUIT-3", LineStatus.ACTIVE, {'PCHSEL400': 10.0}),
            "BUCKETLINE-1": Line("BUCKETLINE-1", "CIRCUIT-1", LineStatus.ACTIVE, {'BUCLOF1KG': 10.25})
        }

    def _create_sample_tanks(self) -> Dict[str, Tank]:
        return {
            "LT-1": Tank("LT-1", 5000, ['CURD', 'POUCH_CURD'], None, 0, True, None),
            "LT-2": Tank("LT-2", 5000, ['CURD', 'POUCH_CURD'], None, 0, True, None),
            "LT-3": Tank("LT-3", 5000, ['CURD', 'POUCH_CURD'], None, 0, True, None),
            "LT-4": Tank("LT-4", 2000, ['MISHTI_DOI'], None, 0, True, None),
        }

    def _create_sample_shifts(self) -> Dict[str, Shift]:
        base_date = datetime.now().date()
        return {
            "A": Shift("A", datetime.combine(base_date, datetime.strptime('06:00', '%H:%M').time()), datetime.combine(base_date, datetime.strptime('14:00', '%H:%M').time()), True),
            "B": Shift("B", datetime.combine(base_date, datetime.strptime('14:00', '%H:%M').time()), datetime.combine(base_date , datetime.strptime('22:00', '%H:%M').time()), True),
            "C": Shift("C", datetime.combine(base_date, datetime.strptime('22:00', '%H:%M').time()), datetime.combine(base_date + timedelta(1), datetime.strptime('06:00', '%H:%M').time()), True)
        }

    def _create_sample_indents(self) -> Dict[str, UserIndent]:
        return {
            "ODR10001": UserIndent("CUPSEL200", 1000, Priority.HIGH, datetime.now() + timedelta(days=0.5), "ODR10001"),
            "ODR10002": UserIndent("BUCLOF1KG", 3000, Priority.MEDIUM, datetime.now() + timedelta(days=0.5), "ODR10002"),
            "ODR10005": UserIndent("CUPSEL400", 2000, Priority.HIGH, datetime.now() + timedelta(days=0.5), "ODR10005"),
            "ODR10010": UserIndent("CUPMID200", 1500, Priority.LOW, datetime.now() + timedelta(days=0.5), "ODR10010")
        }
    
    def _get_csv_or_warn(self, filename: str, write=False) -> Union[Path, pd.DataFrame, None]:
        path = self.data_dir / filename
        if write:
            return path
        else:
            if not path.exists():
                    self.validation_errors.append(f"{filename} not found.")
                    logger.warning(f"{filename} not found at {path}")
                    return None
            return pd.read_csv(path)
        