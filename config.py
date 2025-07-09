"""
Global Configuration for Dairy Scheduler MVP
Contains all constants, penalties, and system parameters
"""
from datetime import timedelta, datetime
from pathlib import Path
import utils.data_models as dm
from utils.data_models import *
from typing import List, Dict

# File Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Production Constants
DEFAULT_SHIFT_HOURS = 8
DEFAULT_CIP_TIME_MINUTES = 45
SETUP_TIME_SAME_VARIANT_MINUTES = 15
SETUP_TIME_DIFFERENT_VARIANT_MINUTES = 30

#Lines list
LINES: Dict[str, Line] = {}
SKUS: Dict[str, SKU] = {}
TANKS: Dict[str, Tank] = {}
SHIFTS: Dict[str, Shift] = {}
USER_INDENTS: Dict[str, UserIndent] = {}
EQUIPMENTS: Dict[str, Equipment] = {}
PRODUCTS: Dict[str, Product] = {}
ROOMS: Dict[str, Room] = {}
CIP_CIRCUIT: Dict[str, CIP_circuit] = {}

    
def get_resource(self, id: str):
    """Return the actual class type associated with this resource type."""
    return {
        ResourceType.TANK: TANKS[id],
        ResourceType.LINE: LINES[id],
        ResourceType.ROOM: ROOMS[id],
        ResourceType.EQUIPMENT: EQUIPMENTS[id]
    }[self]
    

# Scheduling Penalties (for optimization scoring)
PENALTY_WEIGHTS = {
    'unfulfilled_demand': 1000,      # Heavy penalty for not meeting demand
    'line_setup_cost': 50,           # Cost of line changeover
    'tank_CIP_cost': 100,            # Cost of CIP operations
    'shift_overtime_cost': 2000,      # Cost of exceeding shift time
    'efficiency_bonus': -25          # Bonus for efficient line utilization
}

# Constraint Thresholds
MIN_BATCH_SIZE_LITERS = 5000
MAX_BATCH_SIZE_LITERS = 5000
MIN_PRODUCTION_EFFICIENCY = 0.7    # 70% minimum line efficiency

# Time Constants
MINUTES_PER_HOUR = 60
HOURS_PER_SHIFT = 8
MAX_OVERTIME_MINUTES = 60

# Data Validation Rules
REQUIRED_COLUMNS = {
    'user_indent': ['SKU_ID', 'Qty_Required', 'Priority', 'Due_Date'],
    'sku_config': ['SKU_ID', 'Product_Type', 'Variant', 'Base_Production_Rate'],
    'line_config': ['Line_ID', 'Max_Capacity', 'Active_Status'],
    'tank_config': ['Tank_ID', 'Capacity_Liters', 'Current_Product', 'Available'],
    'shift_config': ['Shift_ID', 'Start_Time', 'End_Time', 'Active']
}

# Default Values for Missing Data
DEFAULTS = {
    'priority': 3,              # Medium priority if not specified
    'production_rate': 5.75,     # Liters per minutes default
    'setup_time': 30,          # Minutes
    'CIP_time': 45,             # Minutes
    'base_date': datetime.now() + timedelta(1)
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': "Dairy Production Scheduler",
    'page_icon': "🥛",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Export Formats
SUPPORTED_EXPORT_FORMATS = ['CSV', 'Excel', 'JSON']

CAPACITY_GROUPS = {
    'Lassi': {
        'categories': ['MANGO-LASSI', 'ROSE-LASSI', 'SHAHI-LASSI'],
        'daily_capacity': 20000
    },
    'Curd': {
        'categories': ['SELECT-CURD', 'LOW-FAT-CURD', 'LFT-POUCH-CURD', 'PLN-POUCH-CURD'],
        'daily_capacity': 20000 
    }
}