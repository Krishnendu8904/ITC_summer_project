from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import yaml

class ProductTypeRegistry:
    _product_types = {
        "POUCH_CURD": "Pouch Curd",
        "CURD": "Curd",
        "MISHTI_DOI": "Mishti Doi",
        "MILK": "Milk"
    }

    @classmethod
    def register(cls, code: str, name: str):
        cls._product_types[code] = name

    @classmethod
    def get_name(cls, code: str) -> Optional[str]:
        return cls._product_types.get(code)

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        return dict(cls._product_types)

    @classmethod
    def load_from_dict(cls, product_dict: Dict[str, str]):
        cls._product_types.update(product_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        with open(file_path) as f:
            data = yaml.safe_load(f)
            for product in data.get("products", []):
                cls.register(product["code"], product["name"])

class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class LineStatus(Enum):
    ACTIVE = "Active"
    MAINTENANCE = "Maintenance"
    CIP = "CIP"
    IDLE = "Idle"

@dataclass
class SKU:
    sku_id: str
    product_category: str
    variant: str
    setup_time: int
    inventory_units: Optional[float] = 0.0 

    def __post_init__(self):
        if self.product_category not in ProductTypeRegistry.get_all().keys():
            raise ValueError(f"Unknown product type: {self.product_category}")

@dataclass
class Line:
    line_id: str
    cip_circuit: str
    status: LineStatus = LineStatus.IDLE
    compatible_skus_max_production: Dict[str, float] = field(default_factory=dict) # COMPATIBLE SKUS AND ITS MAX PRODUCTION
    current_sku: Optional[SKU] = None
    last_cip_time: Optional[datetime] = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = LineStatus(self.status)

    def is_available(self) -> bool:
        return self.status in [LineStatus.ACTIVE, LineStatus.IDLE]

    def needs_setup(self, target_sku: SKU) -> bool:
        if not self.current_sku:
            return True
        return self.current_sku.sku_id != target_sku.sku_id

    def needs_cip(self, target_sku: SKU) -> bool:
        if not self.current_sku:
            return False
        return self.current_sku.product_category != target_sku.product_category
        
@dataclass
class Tank:
    tank_id: str
    capacity_liters: float = 5000.0
    compatible_product_categories: List[str] = field(default_factory=list) 
    
    current_product_category: Optional[str] = None # Stores the product category ID 
    current_volume: float = 0.0
    available: bool = True # Can be False if undergoing maintenance, cleaning etc.
    last_cleaned: Optional[datetime] = None

    def get_available_volume(self) -> float:
        """Returns the actual empty space in the tank in liters."""
        return self.capacity_liters - self.current_volume

    def is_compatible_with_category(self, sku: SKU) -> bool:
        """Checks if the tank is configured to store a given product category."""
        if not self.compatible_product_categories: # If list is empty, assume compatible with all
            return True
        return sku.product_category in self.compatible_product_categories

    def can_store(self, volume_to_add: float, sku: SKU) -> bool:
        """
        Determines if the tank can currently store the specified volume of the given SKU.
        This method does NOT modify the tank's state.
        """
        if not self.available:
            return False # Tank is marked unavailable for use

        if not self.is_compatible_with_category(sku):
            return False # SKU's category is not allowed in this tank

        # Check if the tank is empty or contains the same product category
        if self.current_product_category is None:
            # Tank is empty, can store any compatible product category up to its full capacity
            return self.get_available_volume() >= volume_to_add
        elif self.current_product_category == sku.product_category:
            # Tank has the same product category, check remaining capacity
            return self.get_available_volume() >= volume_to_add
        else:
            # Tank has a different product category, cannot store directly (requires CIP first)
            return False

    def needs_cip_for_next_sku(self, next_sku: SKU) -> bool:
        """
        Checks if a Clean-In-Place (CIP) operation is required before the next SKU.
        A CIP is typically needed if the tank is not empty and the next product category
        is different from the current one.
        """
        if self.current_product_category is None:
            return False # Tank is empty, no CIP needed for a category change (unless specific rule)
        
        # If current category is different from next SKU's category
        if self.current_product_category != next_sku.product_category:
            return True
        
        return False # Same category, no CIP needed

    def update_fill_state(self, volume_added: float, sku: SKU):
        """
        Updates the tank's volume and product category after a fill operation.
        This method should be called *after* `can_store` has confirmed capacity.
        """
        if self.current_product_category is None:
            self.current_product_category = sku.product_category
        elif self.current_product_category != sku.product_category:
            # This case should ideally be prevented by a `can_store` check and CIP logic
            raise ValueError(f"Tank {self.tank_id} attempted to add different product category "
                             f"('{sku.product_category}') to existing ('{self.current_product_category}'). "
                             "CIP required first.")
        
        if (self.current_volume + volume_added) > self.capacity_liters:
            raise ValueError(f"Tank {self.tank_id} overflow: adding {volume_added}L exceeds capacity.")
            
        self.current_volume += volume_added
        print(f"DEBUG: Tank {self.tank_id} now has {self.current_volume}L of {self.current_product_category}") # For debugging

    def clean_tank(self):
        """Resets the tank state as if it has undergone a CIP."""
        self.current_product_category = None
        self.current_volume = 0.0
        self.last_cleaned = datetime.now()
        self.available = True # Tank becomes available after cleaning.
        print(f"DEBUG: Tank {self.tank_id} cleaned. Now empty and available.") # For debugging

@dataclass
class Shift:
    shift_id: str
    start_time: datetime
    end_time: datetime
    active: bool = True

    def duration_minutes(self) -> int:
        print(type(staticmethod))
        return int((self.end_time - self.start_time).total_seconds() / 60)

    def is_active(self) -> bool:
        return self.active

    def time_remaining(self, current_time: datetime) -> int:
        if current_time >= self.end_time:
            return 0
        return int((self.end_time - current_time).total_seconds() / 60)

@dataclass
class UserIndent:
    sku_id: str
    qty_required: float
    priority: Priority = Priority.MEDIUM
    due_date: Optional[datetime] = field(default_factory=datetime.today)
    order_no: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.priority, int):
            self.priority = Priority(self.priority)

@dataclass
class ScheduleItem:
    sku: SKU
    line: Line
    tank: Tank
    shift: Shift
    start_time: datetime
    end_time: datetime
    quantity: float
    produced_quantity: Optional[float]
    setup_time_minutes: int = 0
    cip_time_minutes: int = 0
    status: str = "Scheduled"

    def duration_minutes(self) -> int:
        return int((self.end_time - self.start_time).total_seconds() / 60)

    def total_time_minutes(self) -> int:
        return self.duration_minutes() + self.setup_time_minutes + self.cip_time_minutes

    def to_dict(self) -> Dict[str, Any]:
        return {
            'SKU_ID': self.sku.sku_id,
            'Line_ID': self.line_id.line_id,
            'Tank_ID': self.tank_id.tank_id,
            'Shift_ID': self.shift_id.shift_id,
            'Start_Time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'End_Time': self.end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Quantity_Liters': self.quantity,
            'Setup_Time_Min': self.setup_time_minutes,
            'CIP_Time_Min': self.cip_time_minutes,
            'Duration_Min': self.duration_minutes(),
            'Status': self.status
        }

    def efficiency(self) -> float:
        if self.produced_quantity is None or self.quantity == 0:
            return 0.0
        return min(1.0, self.produced_quantity / self.quantity)

@dataclass
class SchedulingResult:
    schedule_items: List[ScheduleItem]
    unfulfilled_indents: List[UserIndent]
    total_production: float
    efficiency_score: float
    warnings: Optional[List[str]]

    def item_efficiencies(self) -> List[float]:
        return [item.efficiency() for item in self.schedule_items]

    def success_rate(self) -> float:
        efficiencies = self.item_efficiencies()
        if not efficiencies:
            return 1.0
        return sum(efficiencies) / len(efficiencies)

@dataclass
class Inventory:
    inventory_id: str
    inventory_space: int
    compatible_skus: List[str]
    skus_in: Dict[str, int] = field(default_factory=dict)
    space_occupied: int = 0
    available: bool = True

    def put_inventory(self, sku: SKU, count: int) -> int:
        if not self.available or sku.sku_id not in self.compatible_skus:
            return count
        remaining_space = self.inventory_space - self.space_occupied
        to_store = min(count, remaining_space)
        self.skus_in[sku.sku_id] = self.skus_in.get(sku.sku_id, 0) + to_store
        self.space_occupied += to_store
        return count - to_store

    def is_full(self) -> bool:
        return self.space_occupied >= self.inventory_space

    def has_space(self) -> bool:
        return self.space_occupied < self.inventory_space

@dataclass
class CIP_circuit:
    circuit_id: str
    connected_units: Dict[str, bool] = field(default_factory=dict)
    available: bool = True
    cip_time: int = 0

    def __post_init__(self):
        if self.available is None:
            self.available = True
