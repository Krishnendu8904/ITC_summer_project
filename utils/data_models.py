from __future__ import annotations # For forward references in type hints (e.g., 'Product')
from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import List, Optional, Dict, Any, Union
from enum import Enum

# Custom StrEnum for consistent string-backed Enums (Python 3.10 and earlier)
# If using Python 3.11+, you can directly use `from enum import StrEnum`

class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

# Unified Priority enum with proper ordering
class Priority(Enum):
    LOW = 1
    MEDIUM = 3
    HIGH = 5
# Consolidated Status for all resources (Line, Tank, Equipment, Room)
class CapacityType(Enum):
    BATCH = 1      # Default: For resources that do one task at a time.
    CUMULATIVE = 0

class ResourceType(StrEnum): # Represents categories of resources
    TANK = "Tank"
    LINE = "Line"
    ROOM = "Room"
    EQUIPMENT = "Equipment"

class ResourceStatus(StrEnum):
    ACTIVE = "Active"
    MAINTENANCE = "Maintenance"
    IDLE = "Idle"
    UNAVAILABLE = "Unavailable"
    CIP = "CIP" # CIP can be a status for any resource undergoing cleaning
    SETUP = "Setup" # SETUP can be a status for any resource undergoing setup

class TankType(StrEnum):
    PREPROCESSING = "MST"
    PROCESSING = "LT"
    FILL_TANK = "HMST"

class RoomType(StrEnum):
    INCUBATOR = "Incubator"
    BLAST_CHILLING = "Blast_Chilling"
    AGING = "Aging"
    STORAGE = "Storage"
    
class ScheduleStatus(StrEnum): # New enum for ScheduleItem status
    SCHEDULED = "Scheduled"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    CANCELED = "Canceled"
    DELAYED = "Delayed"
    UNFULFILLED = "Unfulfilled" # For indents that couldn't be scheduled

class ProcessType(StrEnum):
    INCUBATION = "Incubation"
    CHILLING = "Chilling"
    STORING = "Storing"
    PREPROCESSING = "Preprocessing"
    PROCESSING = "Processing"
    PACKAGING = "Packaging"


def _dict_to_str(d: Dict[str, float]) -> str:
    """Converts a dictionary of string to float to a semicolon-separated string."""
    if not d:
        return ""
    return ";".join([f"{x}:{y}" for x, y in d.items()])

def _str_to_dict(s: str) -> Dict[str, float]:
    """Converts a semicolon-separated string back into a dictionary of str -> float."""
    if not s:
        return {}
    return {k: float(v) for k, v in (item.split(":") for item in s.split(";"))}

@dataclass
class SKU:
    sku_id: str
    product_category: str # Links to Product.product_category
    inventory_size: float  # Size/volume per unit for inventory calculations (e.g., 0.5L per pouch)

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "SKU_ID": self.sku_id,
            "Product_Category": self.product_category,
            "Inventory_Size": self.inventory_size
        }

@dataclass
class Line:
    line_id: str
    compatible_skus_max_production: Dict[str, float] =field(default_factory= dict)  # SKU_ID -> max production rate (kg/minute per SKU)
    CIP_duration_minutes: int = 0  # minutes for this line's full CIP process
    status: ResourceStatus = ResourceStatus.IDLE # Using consolidated status
    setup_time_minutes: int = 30  # minutes for generic setup (e.g., if switching lines, or general setup)
    current_sku: Optional[str] = None  # current sku running on the linex
    current_product_category: Optional[str] = None # current product category running on the line

    resourcetype = ResourceType.LINE

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ResourceStatus(self.status)
        
        if isinstance(self.compatible_skus_max_production, str):
            self.compatible_skus_max_production = _str_to_dict(self.compatible_skus_max_production)

    def _to_dict(self) -> Dict[str, Any]:
        # Compatibility is handled in a separate CSV, so we don't include it here. 
        return {
            "Line_ID": self.line_id,
            "Compatible SKUs and Max Production": _dict_to_str(self.compatible_skus_max_production),
            "CIP_Duration_Min": self.CIP_duration_minutes,
            "Status": self.status.value,
            "Setup_Time_Min": self.setup_time_minutes,
            "Current_SKU": self.current_sku,
            "Current_Product_Category": self.current_product_category,
        }

    def is_available(self) -> bool:
        return self.status in [ResourceStatus.ACTIVE, ResourceStatus.IDLE]

    def needs_setup_for_sku_change(self, target_sku: SKU) -> bool:
        """
        Checks if setup is needed due to SKU or product category change.
        This could be simple (any change = setup) or complex (e.g., product A to B no setup, B to C setup).
        """
        if self.current_product_category is None: # Line is idle, potentially needs initial setup
            return True # Assume initial setup always needed for first run
        
        # If the target product category is different, typically needs setup
        return self.current_sku != target_sku.sku_id

@dataclass
class Tank:
    tank_id: str
    capacity_liters: float  # Capacity in liters
    compatible_product_categories: List[str] = field(default_factory=list) # e.g., ["Curd", "Milk"]
    status: ResourceStatus = ResourceStatus.IDLE # Using consolidated status
    tank_type: TankType = TankType.PROCESSING
    current_product_category: Optional[str] = None  # current product category in tank
    current_volume_liters: float = 0.0 # Current volume in liters
    CIP_duration_minutes: int = 60 # minutes, specific to this tank for CIP

    resource_type = ResourceType.TANK


    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ResourceStatus(self.status)
        if isinstance(self.tank_type, str):
            self.tank_type = TankType(self.tank_type)
        if isinstance(self.compatible_product_categories, str):
            self.compatible_product_categories = [p.strip() for p in self.compatible_product_categories.strip().split(',')]
    
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Tank_ID": self.tank_id,
            "Capacity_Liters": self.capacity_liters,
            "Compatible_Product_Categories": ",".join(self.compatible_product_categories),
            "Status": self.status.value,
            "Tank_Type": self.tank_type.value,
            "Current_Product_Category": self.current_product_category,
            "Current_Volume_Liters": self.current_volume_liters,
            "CIP_Duration_Min": self.CIP_duration_minutes,
        }

    def get_available_capacity_liters(self) -> float:
        return self.capacity_liters - self.current_volume_liters

    def is_compatible_with_product_category(self, product_category: str) -> bool:
        if not self.compatible_product_categories: # If empty list, assume compatible with all
            return True
        return product_category in self.compatible_product_categories

    def can_store(self, volume: float, product_category: str) -> bool:
        if self.status not in [ResourceStatus.ACTIVE, ResourceStatus.IDLE]:
            return False
        if not self.is_compatible_with_product_category(product_category):
            return False
        # If tank already contains a different product category, it can't store without CIP
        if self.current_product_category and self.current_product_category != product_category and self.current_volume_liters != 0:
            return False
        return self.get_available_capacity_liters() >= volume

@dataclass
class Shift:
    shift_id: str
    start_time: time  # datetime.time object
    end_time: time    # datetime.time object
    is_active: bool = True

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Shift_ID": self.shift_id,
            "Start_Time": self.start_time.strftime("%H:%M"),
            "End_Time": self.end_time.strftime("%H:%M"),
            "Is_Active": self.is_active,
        }

    def duration_minutes(self) -> int:
        # Use an arbitrary date to calculate timedelta between times
        today = datetime.today().date()
        start_dt = datetime.combine(today, self.start_time)
        end_dt = datetime.combine(today, self.end_time)

        # Handle overnight shifts
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return int((end_dt - start_dt).total_seconds() / 60)

@dataclass
class Equipment: # General purpose equipment like mixers, pasteurizers, incubators
    equipment_id: str
    processing_speed: float = 0.0 # Processing Rate in l / min or kg / min
    start_up_time: int = 0 # machine start up time
    compatible_product_categories: List[str] = field(default_factory=list) # What product categories this equipment can handle, empty means all
    CIP_duration_minutes: int = 0  # minutes for CIP (could be 0 if no CIP)
    status: ResourceStatus = ResourceStatus.IDLE # Using consolidated status
    setup_time_minutes: int = 0  # minutes, for setup specific to this equipment/product change
    current_product_category: Optional[str] = None  # current product category running on equipment
    resource_type = ResourceType.EQUIPMENT
    capacity_type: CapacityType = CapacityType.BATCH

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ResourceStatus(self.status)

        if isinstance(self.capacity_type, int):
            self.capacity_type = CapacityType(self.capacity_type)

        if isinstance(self.compatible_product_categories, str):
            self.compatible_product_categories  = [p.strip() for p in self.compatible_product_categories.strip().split(',')]

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Equipment_ID": self.equipment_id,
            "Processing Speed": self.processing_speed,
            "Supported_Product_Categories": ",".join(self.compatible_product_categories),
            "CIP_Duration_Min": self.CIP_duration_minutes,
            "Status": self.status.value,
            "Setup_Time_Min": self.setup_time_minutes,
            "Current_Product_Category": self.current_product_category,
        }

    def is_available(self) -> bool:
        return self.status in [ResourceStatus.ACTIVE, ResourceStatus.IDLE]

    def supports_product_category(self, product_category: str) -> bool:
        if not self.compatible_product_categories: # If empty list, assume supports all
            return True
        return product_category in self.compatible_product_categories

@dataclass
class Room:
    room_id: str
    capacity_units: float  # e.g., number in EUIs
    supported_skus: List[str] = field(default_factory=list) # SKU IDs that can be stored/processed here
    room_type: RoomType = RoomType.STORAGE # e.g., Incubator, Storage, Packaging
    current_occupancy_units: float = 0.0
    status: ResourceStatus = ResourceStatus.ACTIVE # Using consolidated status
    resource_type = ResourceType.ROOM

    def __post_init__(self):
        if isinstance(self.room_type, str):
            self.room_type = RoomType(self.room_type)
        if isinstance(self.status, str):
            self.status = ResourceStatus(self.status)
    
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Room_ID": self.room_id,
            "Capacity_Units": self.capacity_units,
            "Supported_SKUs": ",".join(self.supported_skus),
            "Room_Type": self.room_type.value,
            "Current_Occupancy_Units": self.current_occupancy_units,
            "Status": self.status.value,
        }

    def get_available_capacity_units(self) -> float:
        return self.capacity_units - self.current_occupancy_units

    def can_accommodate(self, sku_id: str, quantity: float) -> bool:
        if self.status != ResourceStatus.ACTIVE:
            return False
        if not self.supported_skus: # If empty list, assume supports all
            return True
        if sku_id not in self.supported_skus:
            return False
        return self.get_available_capacity_units() >= quantity

    def is_full(self) -> bool:
        return self.current_occupancy_units >= self.capacity_units

@dataclass
class ResourceRequirement:
    resource_type: ResourceType = ResourceType.EQUIPMENT
    compatible_ids: List[str] = field(default_factory= List)
    
    def __post_init__(self):
        try:
                # Convert the string to a proper ResourceType enum member (e.g., "LINE" -> ResourceType.LINE)
                # .upper() makes it case-insensitive
            self.resource_type = ResourceType[self.resource_type.upper()]
        except KeyError:
            # This handles cases where an invalid string is provided (e.g., "TRUCK")
            raise ValueError(
                f"'{self.resource_type}' is not a valid ResourceType. "
                f"Please use one of {[e.name for e in ResourceType]}."
            )

@dataclass
class ProcessingStep:
    """Defines a single, generic step in a product's overall process."""
    step_id: str # Unique ID for this specific step within a product's process
    name: str  # e.g., "Mixing", "Pasteurization", "Incubation", "Packaging"
    duration_minutes: Optional[float] # Time taken to process ONE batch of product at this step
    min_capacity_required_liters: Optional[float] = 0.0 # Min capacity of resource needed (e.g., 1000L tank for mixing)
    setup_time: Optional[int] = 0 # set-up in minutes
    cool_down: Optional[int] = 0 # Cool down in minutes
    requirements: List[ResourceRequirement] = field(default_factory= list)
    process_type: ProcessType = ProcessType.PACKAGING
    
    # Flags for step-specific requirements for the scheduler
    requires_CIP_after: bool = False # Does this step require CIP on its resource after completion?

    def _post_init(self):
        if isinstance(self.process_type, str):
            self.process_type = ProcessType(self.process_type)


    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Step_ID": self.step_id,
            "Step_Name": self.name,
            "Process_Type": self.process_type,
            "Resource_Type": ' / '.join(resource.resource_type.name for resource in self.requirements),
            "Duration_Minutes_Per_Batch": self.duration_minutes,
            "Min_Capacity_Required_Liters": self.min_capacity_required_liters,
            "Compatible_Resource_IDs": " / ".join((','.join(resource_id for resource_id in resource.compatible_ids)) for resource in self.requirements),
            "Requires_CIP_After": self.requires_CIP_after
        }
    

@dataclass
class Product:
    """Defines a product category and its sequence of processing steps."""
    product_category: str # Unique identifier for the product type (e.g., "CURD", "MILK")
    processing_steps: List[ProcessingStep] = field(default_factory=list)
    max_batch_size: Optional[int] =0

    def _to_dicts(self) -> List[Dict[str, Any]]:
        """Returns a list of dictionaries, one for each processing step, including the product category."""
        if not self.processing_steps:
             return [{
                "Product_Category": self.product_category,
                "Step_ID": None, "Step_Name": None, 
                "Duration_Minutes_Per_Batch": None, "Min_Capacity_Required_Liters": None,
                "Requirements": None,
                "Requires_CIP_After": None, 
                "Max Batch Size": None
            }]
        
        step_dicts = []
        for step in self.processing_steps:
            step_dict = step._to_dict()
            step_dict["Product_Category"] = self.product_category
            step_dict["Max Batch Size"] = self.max_batch_size
            step_dicts.append(step_dict)
        
        return step_dicts
    
    def get_total_processing_time_per_batch(self) -> float:
        """Returns the sum of durations for all processing steps for one batch."""
        return sum(step.duration_minutes for step in self.processing_steps)

    def get_steps_by_resource_type(self, resource_type: ResourceType) -> List[ProcessingStep]:
        """Returns processing steps that require a specific resource type."""
        return [step for step in self.processing_steps if resource_type in step.requirements]
    
@dataclass
class CIP_circuit:
    circuit_id: str
    connected_resource_ids: List[str] = field(default_factory=list) # IDs of connected equipment/lines/tanks
    is_available: bool = True # Is the circuit itself operational

    def _post__init(self):
        if isinstance(self.connected_resource_ids, str):
            self.connected_resource_ids = [p.strip() for p in self.connected_resource_ids.strip()]

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Circuit_ID": self.circuit_id,
            "Connected_Resource_IDs": ",".join(self.connected_resource_ids),
            "Is_Available": self.is_available
        }

@dataclass
class CIPSchedule:
    """Clean-in-Place schedule details"""
    CIP_id: str
    resource_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    preceding_task_id: str
    following_task_id: str
    CIP_type: str = "standard"  # standard, deep, sanitization
    
    def __str__(self) -> str:
        return f"CIP {self.CIP_id}: {self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')} on {self.resource_id}"
# --- Schedule Output Classes ---

@dataclass
class ScheduledProcessingStep:
    """Represents a specific instance of a processing step that has been scheduled."""
    step_definition: ProcessingStep # Reference to the generic step definition
    allocated_resource_id: str # The specific resource (e.g., "Line1", "MixerA") used for this step
    start_time: datetime
    end_time: datetime
    volume_processed_liters: float # Volume processed in this specific instance of the step

@dataclass
class ScheduleItem:
    """Represents a single, complete scheduled production run for an SKU."""
    indent_id: Optional[str] = None # Link back to UserIndent.order_no if available
    sku: Optional[SKU] = None# The specific SKU being produced in this schedule item
    quantity_to_produce_liters: float = 0.0 # Total quantity requested for this item (liters)
    scheduled_activities: List[ScheduledProcessingStep] = field(default_factory=list)
    
    overall_start_time: Optional[datetime] = None # Calculated from scheduled_activities
    overall_end_time: Optional[datetime] = None   # Calculated from scheduled_activities
    produced_quantity_liters: Optional[float] = 0.0 # Actual quantity produced (after execution)
    total_setup_time_minutes: int = 0  # Sum of all setup times incurred for this item
    total_CIP_time_minutes: int = 0    # Sum of all CIP times incurred for this item
    status: ScheduleStatus = ScheduleStatus.SCHEDULED # Using Enum for status

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ScheduleStatus(self.status)
        # Calculate overall times immediately after creation if steps are provided
        if self.scheduled_activities:
            self.calculate_overall_times()

    def calculate_overall_times(self):
        if self.scheduled_activities:
            self.overall_start_time = min(s.start_time for s in self.scheduled_activities)
            self.overall_end_time = max(s.end_time for s in self.scheduled_activities)

    def get_total_duration_minutes(self) -> int:
        if self.overall_start_time and self.overall_end_time:
            return int((self.overall_end_time - self.overall_start_time).total_seconds() / 60)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        self.calculate_overall_times() # Ensure times are updated before conversion
        return {
            'indent_id': self.indent_id,
            'SKU_ID': self.sku.sku_id,
            'Product_Category': self.sku.product_category,
            'Quantity_To_Produce_Liters': self.quantity_to_produce_liters,
            'Produced_Quantity_Liters': self.produced_quantity_liters,
            'Scheduled_Activities': [
                {
                    'step_id': sps.step_definition.step_id,
                    'step_name': sps.step_definition.name,
                    'resource_type': sps.step_definition.resource_type.value,
                    'allocated_resource_id': sps.allocated_resource_id,
                    'start_time': sps.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': sps.end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'volume_processed_liters': sps.volume_processed_liters
                } for sps in self.scheduled_activities
            ],
            'Overall_Start': self.overall_start_time.strftime('%Y-%m-%d %H:%M:%S') if self.overall_start_time else None,
            'Overall_End': self.overall_end_time.strftime('%Y-%m-%d %H:%M:%S') if self.overall_end_time else None,
            'Total_Setup_Time_Min': self.total_setup_time_minutes,
            'Total_CIP_Time_Min': self.total_CIP_time_minutes,
            'Total_Duration_Min': self.get_total_duration_minutes(),
            'Status': self.status.value
        }

    def efficiency(self) -> float:
        if self.produced_quantity_liters is None or self.quantity_to_produce_liters == 0:
            return 0.0
        return min(1.0, self.produced_quantity_liters / self.quantity_to_produce_liters)

@dataclass
class UserIndent:
    order_no: str # Made required for explicit linking
    sku_id: str
    qty_required_liters: float # Renamed for clarity, implies liters
    priority: Priority = Priority.MEDIUM
    due_date: datetime = field(default_factory=lambda: datetime.combine(datetime.now().date() + timedelta(days=2), time(14, 0))) # Corrected default_factory to datetime.now()

    def __post_init__(self):
        if isinstance(self.priority, int):
            self.priority = Priority(self.priority)

        # If loading from JSON/CSV, you might need to convert string to datetime here
        if isinstance(self.due_date, str):
            self.due_date = datetime.fromisoformat(self.due_date) # Example, adjust format as needed
    
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "Order_Number": self.order_no,
            "SKU_ID": self.sku_id,
            "Qty_Required_Liters": self.qty_required_liters,
            "Priority": self.priority.value,
            "Due_Date": self.due_date.strftime("%Y-%m-%d"),
        }

@dataclass
class FlowEdge:
    """Represents an edge in a flow network (e.g., for routing or material flow optimization)"""
    from_node: str
    to_node: str
    capacity: float
    cost: float
    flow: float = 0.0

@dataclass
class ProductionSlot:
    """
    Represents a potential time slot on a resource for a specific production activity.
    This might be a pre-calculated available window or a segment in a scheduling solver.
    """
    resource_type: ResourceType # e.g., LINE, TANK, EQUIPMENT, ROOM
    resource_id: str # The specific ID of the resource (e.g., "Line1", "MixerA")
    product_category: str # What product category this slot is suitable for
    sku_id: Optional[str] = None # Optional: Which specific SKU might be produced/handled
    start_time: datetime = datetime.now()
    end_time: datetime = datetime.now()
    # The capacity this slot can produce/handle (e.g., liters on a line, or space in a room)
    available_capacity_liters: float = 0.0
    
    # Flags if the slot requires initial setup/CIP for a specific product/SKU
    requires_initial_setup: bool = False
    requires_initial_CIP: bool = False

    def __post_init__(self):
        if isinstance(self.resource_type, str):
            self.resource_type = ResourceType(self.resource_type)

# --- Advanced Scheduling Classes ---
@dataclass
class TaskSchedule:
    """Represents a scheduled task with all relevant details"""
    task_id: str
    order_no: str
    sku_id: str
    batch_index: int
    step_id: str
    start_time: datetime
    end_time: datetime
    resource_id: str
    volume: int
    priority: Priority
    setup_time: int = 0
    CIP_required: bool = False
    
    @property
    def duration_minutes(self) -> int:
        """Calculate task duration in minutes"""
        return int((self.end_time - self.start_time).total_seconds() / 60)
    
    def __str__(self) -> str:
        return f"Task {self.task_id}: {self.start_time.strftime('%Y-%m-%d %H:%M')} - {self.end_time.strftime('%Y-%m-%d %H:%M')} on {self.resource_id}"

@dataclass
class ResourceUtilization:
    """Resource utilization details"""
    resource_id: str
    total_scheduled_time: int  # minutes
    utilization_rate: float   # 0.0 to 1.0
    idle_periods: List[tuple] # List of (start_time, end_time) tuples
    scheduled_tasks: List[str] # List of task_ids
    setup_time: int = 0
    CIP_time: int = 0
    
    @property
    def effective_utilization(self) -> float:
        """Utilization excluding setup and CIP time"""
        productive_time = self.total_scheduled_time - self.setup_time - self.CIP_time
        return max(0.0, productive_time / self.total_scheduled_time) if self.total_scheduled_time > 0 else 0.0

@dataclass
class ProductionSummary:
    """Production summary for an order"""
    order_no: str
    sku_id: str
    required_quantity: int
    produced_quantity: int
    underproduction: int
    overproduction: int
    scheduled: bool
    completion_time: Optional[datetime] = None
    due_date: Optional[datetime] = None
    priority: Priority = Priority.MEDIUM
    
    @property
    def fulfillment_rate(self) -> float:
        """Calculate fulfillment rate (0.0 to 1.0+)"""
        if self.required_quantity == 0:
            return 1.0
        return self.produced_quantity / self.required_quantity
    
    @property
    def is_on_time(self) -> bool:
        """Check if order is completed on time"""
        if not self.completion_time or not self.due_date:
            return False
        return self.completion_time <= self.due_date
    
    @property
    def is_otif(self) -> bool:
        """Check if order is On-Time In-Full"""
        return self.is_on_time and self.fulfillment_rate >= 1.0
    
    @property
    def tardiness_minutes(self) -> int:
        """Calculate tardiness in minutes (0 if on time)"""
        if not self.completion_time or not self.due_date or self.is_on_time:
            return 0
        return int((self.completion_time - self.due_date).total_seconds() / 60)

@dataclass
class SchedulingMetrics:
    """Key performance metrics for the schedule"""
    total_orders: int
    scheduled_orders: int
    schedule_efficiency: float  # 0.0 to 1.0
    average_resource_utilization: float
    otif_rate: float  # On-Time In-Full rate
    total_production_volume: int
    total_setup_time: int
    total_CIP_time: int
    total_idle_time: int
    
    @property
    def scheduling_success_rate(self) -> float:
        """Percentage of orders successfully scheduled"""
        if self.total_orders == 0:
            return 0.0
        return self.scheduled_orders / self.total_orders
    
    @property
    def productive_time_ratio(self) -> float:
        """Ratio of productive time vs. total time"""
        total_time = self.total_production_volume + self.total_setup_time + self.total_CIP_time + self.total_idle_time
        if total_time == 0:
            return 0.0
        return self.total_production_volume / total_time
    
@dataclass
class SchedulingResult:
    """Comprehensive scheduling result with all details"""
    status: int  # cp_model status
    objective_value: float
    scheduled_tasks: List[TaskSchedule]
    resource_utilization: Dict[str, List[Dict[str, Any]]]  # resource_id -> list of task details
    production_summary: Dict[str, Dict[str, Any]]  # order_no -> production details
    solve_time: float
    warnings: List[str] = field(default_factory=list)
    CIP_schedules: List[CIPSchedule] = field(default_factory=list)
    metrics: Optional[SchedulingMetrics] = None
    
    def __post_init__(self):
        """Calculate metrics after initialization"""
        if not self.metrics:
            self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> SchedulingMetrics:
        """Calculate comprehensive scheduling metrics"""
        total_orders = len(self.production_summary)
        scheduled_orders = sum(1 for details in self.production_summary.values() if details.get('scheduled', False))
        
        # Calculate resource utilization
        total_utilization = 0.0
        resource_count = len(self.resource_utilization)
        
        total_production_time = 0
        total_setup_time = 0
        total_CIP_time = 0
        
        for resource_id, tasks in self.resource_utilization.items():
            if tasks:
                resource_time = sum(task['end'] - task['start'] for task in tasks)
                total_production_time += resource_time
                # Note: Setup and CIP time would be calculated separately if tracked
        
        avg_utilization = total_utilization / resource_count if resource_count > 0 else 0.0
        
        # Calculate OTIF rate
        otif_count = 0
        for details in self.production_summary.values():
            if details.get('scheduled', False):
                # This would require completion_time and due_date in production_summary
                # For now, simplified calculation
                fulfillment_rate = details.get('produced_quantity', 0) / max(1, details.get('required_quantity', 1))
                if fulfillment_rate >= 1.0:  # Simplified OTIF check
                    otif_count += 1
        
        otif_rate = otif_count / max(1, scheduled_orders)
        
        return SchedulingMetrics(
            total_orders=total_orders,
            scheduled_orders=scheduled_orders,
            schedule_efficiency=self.objective_value / 10000 if self.objective_value > 0 else 0.0,  # Normalized
            average_resource_utilization=avg_utilization,
            otif_rate=otif_rate,
            total_production_volume=sum(details.get('produced_quantity', 0) for details in self.production_summary.values()),
            total_setup_time=total_setup_time,
            total_CIP_time=len(self.CIP_schedules) * 60,  # Simplified
            total_idle_time=0  # Would need more detailed calculation
        )
    
    @property
    def is_optimal(self) -> bool:
        """Check if solution is optimal"""
        return self.status == cp_model.OPTIMAL
    
    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        return self.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
    
    @property
    def status_name(self) -> str:
        """Get human-readable status name"""
        status_names = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID"
        }
        return status_names.get(self.status, "UNKNOWN")
    
    def get_resource_schedule(self, resource_id: str) -> List[TaskSchedule]:
        """Get all tasks scheduled for a specific resource"""
        return [task for task in self.scheduled_tasks if task.resource_id == resource_id]
    
    def get_order_tasks(self, order_no: str) -> List[TaskSchedule]:
        """Get all tasks for a specific order"""
        return [task for task in self.scheduled_tasks if task.order_no == order_no]
    
    def get_production_summary_objects(self) -> List[ProductionSummary]:
        """Get production summary as structured objects"""
        summaries = []
        for order_no, details in self.production_summary.items():
            # Find completion time from tasks
            order_tasks = self.get_order_tasks(order_no)
            completion_time = max(task.end_time for task in order_tasks) if order_tasks else None
            
            summary = ProductionSummary(
                order_no=order_no,
                sku_id=details.get('sku_id', ''),
                required_quantity=details.get('required_quantity', 0),
                produced_quantity=details.get('produced_quantity', 0),
                underproduction=details.get('underproduction', 0),
                overproduction=details.get('overproduction', 0),
                scheduled=details.get('scheduled', False),
                completion_time=completion_time,
                due_date=details.get('due_date'),
                priority=details.get('priority', Priority.MEDIUM)
            )
            summaries.append(summary)
        
        return summaries
    
    def print_summary(self):
        """Print a comprehensive summary of the scheduling results"""
        print(f"\n{'='*60}")
        print(f"PRODUCTION SCHEDULING RESULT")
        print(f"{'='*60}")
        print(f"Status: {self.status_name}")
        print(f"Objective Value: {self.objective_value:.2f}")
        print(f"Solve Time: {self.solve_time:.2f} seconds")
        print(f"Total Tasks Scheduled: {len(self.scheduled_tasks)}")
        
        if self.metrics:
            print(f"\nKEY METRICS:")
            print(f"  Orders Scheduled: {self.metrics.scheduled_orders}/{self.metrics.total_orders} ({self.metrics.scheduling_success_rate*100:.1f}%)")
            print(f"  OTIF Rate: {self.metrics.otif_rate*100:.1f}%")
            print(f"  Avg Resource Utilization: {self.metrics.average_resource_utilization*100:.1f}%")
            print(f"  Total Production Volume: {self.metrics.total_production_volume:,} liters")
        
        print(f"\nRESOURCE UTILIZATION:")
        for resource_id, tasks in self.resource_utilization.items():
            if tasks:
                total_time = sum(task['end'] - task['start'] for task in tasks)
                print(f"  {resource_id}: {len(tasks)} tasks, {total_time} minutes")
        
        if self.warnings:
            print(f"\nWARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print(f"{'='*60}\n")
