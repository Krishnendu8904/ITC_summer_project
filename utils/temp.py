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
from models.my_Code import ProductTypeRegistry

from models.my_Code import SKU, Line, Tank, Shift, UserIndent, ProductType, Priority, LineStatus
from config import DATA_DIR, REQUIRED_COLUMNS, DEFAULTS

logger = logging.getLogger(__name__)

class DataLoader:
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
                    product_type= ProductTypeRegistry.get_name(row.get('Product_Type', 'Curd')),
                    variant=str(row.get('Variant', 'Select')),
                    compatible_line=Line(row['Line'])
                    setup_time= int(row.get('Setup Time'))
                )
                skus.append(sku)
            except Exception as e:
                self.validation_errors.append(f"Invalid SKU row: {row.to_dict()} - {str(e)}")
        
        logger.info(f"Loaded {len(skus)} SKUs")
        return skus
    