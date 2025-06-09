"""
Data Loader for Dairy Scheduler MVP
Handles reading and validating all input CSV/Excel files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from models.data_models import SKU, Line, Tank, Shift, UserIndent, ProductType, Priority, LineStatus
from config import DATA_DIR, REQUIRED_COLUMNS, DEFAULTS

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and validates all input data files"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.validation_errors = []
        
    def load_all_data(self) -> Dict[str, Any]:
        """Load all required data files and return structured data"""
        try:
            data = {
                'skus': self.load_skus(),
                'lines': self.load_lines(),
                'tanks': self.load_tanks(),
                'shifts': self.load_shifts(),
                'user_indents': self.load_user_indents(),
                'line_sku_compatibility': self.load_line_sku_compatibility(),
                'constraints': self.load_constraints()
            }
            
            if self.validation_errors:
                logger.warning(f"Data validation warnings: {self.validation_errors}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def load_skus(self) -> List[SKU]:
        """Load SKU configuration"""
        file_path = self.data_dir / "sku_config.csv"
        
        if not file_path.exists():
            logger.warning(f"SKU config file not found: {file_path}")
            return self._create_sample_skus()
        
        df = pd.read_csv(file_path)
        skus = []
        
        for _, row in df.iterrows():
            try:
                sku = SKU(
                    sku_id=str(row['SKU_ID']),
                    product_type=ProductType(row.get('Product_Type', 'Curd')),
                    variant=str(row.get('Variant', 'Select')),
                    setup_time_minutes= int(row.get('Setup Time'))
                )
                skus.append(sku)
            except Exception as e:
                self.validation_errors.append(f"Invalid SKU row: {row.to_dict()} - {str(e)}")
        
        logger.info(f"Loaded {len(skus)} SKUs")
        return skus
    
    def load_lines(self) -> List[Line]:
        """Load production line configuration"""
        file_path = self.data_dir / "line_config.csv"
        
        if not file_path.exists():
            logger.warning(f"Line config file not found: {file_path}")
            return self._create_sample_lines()
        
        df = pd.read_csv(file_path)
        lines = []
        
        for _, row in df.iterrows():
            try:
                line = Line(
                    line_id=str(row['Line_ID']),
                    max_capacity=float(row.get('Max_Capacity', 500)),
                    current_product=row.get('Current_Product', None),
                    status=LineStatus(row.get('Active_Status', 'Active'))
                )
                lines.append(line)
            except Exception as e:
                self.validation_errors.append(f"Invalid Line row: {row.to_dict()} - {str(e)}")
        
        logger.info(f"Loaded {len(lines)} production lines")
        return lines
    
    def load_tanks(self) -> List[Tank]:
        """Load tank configuration"""
        file_path = self.data_dir / "tank_config.csv"
        
        if not file_path.exists():
            logger.warning(f"Tank config file not found: {file_path}")
            return self._create_sample_tanks()
        
        df = pd.read_csv(file_path)
        tanks = []
        
        for _, row in df.iterrows():
            try:
                tank = Tank(
                    tank_id=str(row['Tank_ID']),
                    capacity_liters=float(row.get('Capacity_Liters', 1000)),
                    current_product=row.get('Current_Product', None),
                    current_volume=float(row.get('Current_Volume', 0)),
                    available=bool(row.get('Available', True))
                )
                tanks.append(tank)
            except Exception as e:
                self.validation_errors.append(f"Invalid Tank row: {row.to_dict()} - {str(e)}")
        
        logger.info(f"Loaded {len(tanks)} storage tanks")
        return tanks
    
    def load_shifts(self) -> List[Shift]:
        """Load shift configuration"""
        file_path = self.data_dir / "shift_config.csv"
        
        if not file_path.exists():
            logger.warning(f"Shift config file not found: {file_path}")
            return self._create_sample_shifts()
        
        df = pd.read_csv(file_path)
        shifts = []
        
        for _, row in df.iterrows():
            try:
                # Parse time strings
                start_time = self._parse_time(row.get('Start_Time', '08:00'))
                end_time = self._parse_time(row.get('End_Time', '16:00'))
                
                shift = Shift(
                    shift_id=str(row['Shift_ID']),
                    start_time=start_time,
                    end_time=end_time,
                    active=bool(row.get('Active', True))
                )
                shifts.append(shift)
            except Exception as e:
                self.validation_errors.append(f"Invalid Shift row: {row.to_dict()} - {str(e)}")
        
        logger.info(f"Loaded {len(shifts)} work shifts")
        return shifts
    
    def load_user_indents(self) -> List[UserIndent]:
        """Load user production indents/demands"""
        file_path = self.data_dir / "user_indent.csv"
        
        if not file_path.exists():
            logger.warning(f"User indent file not found: {file_path}")
            return self._create_sample_indents()
        
        df = pd.read_csv(file_path)
        indents = []
        
        for _, row in df.iterrows():
            try:
                due_date = None
                if 'Due_Date' in row and pd.notna(row['Due_Date']):
                    due_date = pd.to_datetime(row['Due_Date'])
                
                indent = UserIndent(
                    sku_id=str(row['SKU_ID']),
                    qty_required=float(row['Qty_Required']),
                    priority=Priority(int(row.get('Priority', DEFAULTS['priority']))),
                    due_date=due_date,
                    customer_id=row.get('Customer_ID', None)
                )
                indents.append(indent)
            except Exception as e:
                self.validation_errors.append(f"Invalid Indent row: {row.to_dict()} - {str(e)}")
        
        logger.info(f"Loaded {len(indents)} user indents")
        return indents
    
    def load_line_sku_compatibility(self) -> Dict[str, List[str]]:
        """Load line-SKU compatibility matrix"""
        file_path = self.data_dir / "line_sku.csv"
        compatibility = {}
        
        if not file_path.exists():
            logger.warning(f"Line-SKU compatibility file not found: {file_path}")
            return self._create_sample_compatibility()
        
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            line_id = str(row['Line_ID'])
            sku_id = str(row['SKU_ID'])
            base_production_rate = float(row.get('Base_Production_Rate', DEFAULTS['production_rate'])),
            setup_time_minutes = int(row.get('Setup_Time_Min', DEFAULTS['setup_time'])),
            cip_required = bool(row.get('CIP_Required', True)),
            
            if line_id not in compatibility:
                compatibility[line_id] = []
            compatibility[line_id].append(sku_id)
            
        
        logger.info(f"Loaded compatibility for {len(compatibility)} lines")
        return compatibility
    
    def load_constraints(self) -> Dict[str, Any]:
        """Load special constraints"""
        file_path = self.data_dir / "spl_constraints.csv"
        constraints = {}
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Process constraints based on your specific needs
            # This is a placeholder for custom constraint logic
            constraints = df.to_dict('records')
        
        return constraints
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime object"""
        try:
            # Assume today's date with the given time
            base_date = datetime.now().date()
            time_obj = datetime.strptime(time_str.strip(), '%H:%M').time()
            return datetime.combine(base_date, time_obj)
        except ValueError:
            # Fallback to full datetime parsing
            return pd.to_datetime(time_str)
    
    def _create_sample_skus(self) -> List[SKU]:
        """Create sample SKU data if file missing"""
        return [
            SKU("CURD_001", ProductType.CURD, "Plain", 150, 20, True, 100, 1500),
            SKU("CURD_002", ProductType.CURD, "Flavored", 120, 25, True, 100, 1500),
            SKU("MISHTI_001", ProductType.MISHTI_DOI, "Sweet", 100, 30, True, 50, 1000),
            SKU("MISHTI_002", ProductType.MISHTI_DOI, "Traditional", 90, 35, True, 50, 1000)
        ]
    
    def _create_sample_lines(self) -> List[Line]:
        """Create sample line data if file missing"""
        return [
            Line("L001", 200, None, LineStatus.ACTIVE),
            Line("L002", 180, None, LineStatus.ACTIVE),
            Line("L003", 250, None, LineStatus.ACTIVE)
        ]
    
    def _create_sample_tanks(self) -> List[Tank]:
        """Create sample tank data if file missing"""
        return [
            Tank("T001", 1500, None, 0, True),
            Tank("T002", 1200, None, 0, True),
            Tank("T003", 2000, None, 0, True),
            Tank("T004", 1000, None, 0, True)
        ]
    
    def _create_sample_shifts(self) -> List[Shift]:
        """Create sample shift data if file missing"""
        base_date = datetime.now().date()
        return [
            Shift("S001", 
                  datetime.combine(base_date, datetime.strptime('08:00', '%H:%M').time()),
                  datetime.combine(base_date, datetime.strptime('16:00', '%H:%M').time()),
                  True),
            Shift("S002", 
                  datetime.combine(base_date, datetime.strptime('16:00', '%H:%M').time()),
                  datetime.combine(base_date, datetime.strptime('00:00', '%H:%M').time()) + timedelta(days=1),
                  True),
            Shift("S003", 
                  datetime.combine(base_date, datetime.strptime('00:00', '%H:%M').time()),
                  datetime.combine(base_date, datetime.strptime('08:00', '%H:%M').time()),
                  True)
        ]
    
    def _create_sample_indents(self) -> List[UserIndent]:
        """Create sample indent data if file missing"""
        return [
            UserIndent("CURD_001", 500, Priority.HIGH, datetime.now() + timedelta(days=1)),
            UserIndent("CURD_002", 300, Priority.MEDIUM, datetime.now() + timedelta(days=2)),
            UserIndent("MISHTI_001", 200, Priority.HIGH, datetime.now() + timedelta(days=1)),
            UserIndent("MISHTI_002", 150, Priority.LOW, datetime.now() + timedelta(days=3))
        ]
    
    def _create_sample_compatibility(self) -> Dict[str, List[str]]:
        """Create sample line-SKU compatibility"""
        return {
            "L001": ["CURD_001", "CURD_002"],
            "L002": ["CURD_001", "CURD_002", "MISHTI_001"],
            "L003": ["MISHTI_001", "MISHTI_002"]
        }
    
