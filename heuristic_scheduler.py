import os
from dataclasses import dataclass, field
from re import sub
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, time
from enum import Enum, auto
from collections import defaultdict
import math
import plotly.io as pio
import json
import datetime

from pyparsing import line


# Assume 'config.py' and 'utils/data_models.py' exist in the same directory
# or are accessible in the Python path.
import config
from utils.data_models import *
from utils.data_loader import *
import plotly.express as px
import pandas as pd

# --- Enums for Status and Task Types ---
class ResourceStatus(Enum):
    """Defines the real-time status of a resource."""
    IDLE = auto()      # Clean, empty, and available
    BUSY = auto()      # Currently performing a task
    DIRTY = auto()     # Finished a task, contains product, needs cleaning

@dataclass
class ResourceState:
    """Holds the live state of a single resource."""
    resource_id: str
    status: ResourceStatus = ResourceStatus.IDLE
    current_contents: Optional[str] = None  # e.g., the product_category
    task_id: Optional[str] = None
    dirty_since: Optional[datetime] = None

class ScheduleStatus(Enum):
    """Defines the scheduling status of a task."""
    TENTATIVE = auto()
    BOOKED = auto()
    PENDING = auto()
    FAILED = auto()

class TaskType(Enum):
    """Defines the type of a production task."""
    BULK_PRODUCTION = auto()
    FINISHING = auto()
    CIP = auto() # Clean-In-Place
    LOCKED = auto()

# In scheduler-v2-dummy.py

class TimeManager:
    """
    Manages working shifts and finds valid uninterruptible time slots.
    Now includes logic to buffer the start and end of each shift.
    """
    def __init__(self, shifts: Dict, time_block_minutes: int):
        self.TIME_BLOCK_MINUTES = time_block_minutes
        self.shifts_by_day = defaultdict(list)
        
        START_BUFFER_MINUTES = 30
        END_BUFFER_MINUTES = 15

        for s in shifts.values():
            if s.is_active:
                # To handle time arithmetic, we combine the shift times with a dummy date.
                dummy_date = datetime(2024, 1, 1).date()
                start_dt = datetime.combine(dummy_date, s.start_time)
                end_dt = datetime.combine(dummy_date, s.end_time)

                # Handle overnight shifts by moving the end date to the next day
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)

                # Apply the startup and cooldown buffers
                effective_start_dt = start_dt + timedelta(minutes=START_BUFFER_MINUTES)
                effective_end_dt = end_dt - timedelta(minutes=END_BUFFER_MINUTES)

                # If the buffers make the shift invalid (e.g., too short), skip it.
                if effective_end_dt <= effective_start_dt:
                    continue

                # The rest of the logic uses the new effective times.
                # We store only the time part for the daily check.
                effective_start_time = effective_start_dt.time()
                effective_end_time = effective_end_dt.time()

                # Assume an active shift runs every day of the week
                for day_of_week in range(7):
                    self.shifts_by_day[day_of_week].append((effective_start_time, effective_end_time))

    def _is_in_shift(self, dt_to_check: datetime) -> bool:
        """Checks if a single datetime falls within any active (and buffered) shift."""
        day_of_week = dt_to_check.weekday()
        time_of_day = dt_to_check.time()

        for shift_start, shift_end in self.shifts_by_day.get(day_of_week, []):
            # This logic correctly handles both normal and overnight shifts with the buffered times.
            if shift_end <= shift_start: # Overnight shift
                if time_of_day >= shift_start or time_of_day < shift_end:
                    return True
            else: # Normal shift
                if shift_start <= time_of_day < shift_end:
                    return True
        return False

    def get_shift_end(self, start_dt: datetime) -> Optional[datetime]:
        """
        Finds the shift a datetime belongs to and returns the exact datetime
        when that shift ends (the buffered end time).
        """
        day_of_week = start_dt.weekday()
        time_of_day = start_dt.time()

        for shift_start_time, shift_end_time in self.shifts_by_day.get(day_of_week, []):
            # Normal shift (e.g., 06:00-14:00)
            if shift_start_time < shift_end_time:
                if shift_start_time <= time_of_day < shift_end_time:
                    return datetime.datetime.combine(start_dt.date(), shift_end_time)
            # Overnight shift (e.g., 22:00-06:00)
            else:
                if time_of_day >= shift_start_time:
                    # It's before midnight, so the shift ends on the next day
                    return datetime.datetime.combine(start_dt.date() + datetime.timedelta(days=1), shift_end_time)
                elif time_of_day < shift_end_time:
                     # It's after midnight, so the shift ends on the same day
                    return datetime.datetime.combine(start_dt.date(), shift_end_time)
        
        return None # The given time is not in any active shift

@dataclass
class SubOrder:
    """Represents a piece of a larger order, sized to fit in a batch."""
    parent_order_no: str
    sub_order_id: str
    sku_id: str
    volume: float
    priority: int
    due_date: datetime
    master_batch_id: str

@dataclass
class HeuristicTask:
    """Represents a single, schedulable unit of work for the heuristic model."""
    # Core Identifiers
    task_id: str
    job_id: str      # e.g., Order number or bulk production ID
    sku_id: str
    step: ProcessingStep
    
    
    # Task Characteristics
    task_type: TaskType
    compatible_resources: Dict[ResourceType, List[str]] = field(default_factory=dict)
    base_duration_tokens: int = 60
    volume_liters: float = 0.0
    priority: int = 5 # Default priority, 1 is highest
    total_prereq_duration_tokens: int = 0
    step_idx: int = 0
    batch_idx: int = 0
    
    sub_batch_id: str = "" # New explicit field for batch grouping

    # Relational Links
    previous_task: Optional['HeuristicTask'] = None
    next_task: List['HeuristicTask'] = field(default_factory=list)
    prerequisites: List['HeuristicTask'] = field(default_factory=list)
    is_anchor_task: bool = False
    is_backfilled: bool = False

    # Scheduling Information (to be filled by the heuristic)
    status: ScheduleStatus = ScheduleStatus.PENDING
    assigned_resource_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __repr__(self):
        return f"HeuristicTask(id={self.task_id}, job={self.job_id}, step={self.step.step_id})"

class ResourceManager:
    """
    A stateful manager for all resources in the plant.
    It tracks the real-time status and contents of each resource,
    serving as the single source of truth for the scheduler.
    """
    def __init__(self, tanks: Dict, lines: Dict, equipments: Dict, rooms: Dict, skus: Dict, log_entries: List[str], task_lookup: Dict[str, HeuristicTask]):
        self.resource_states: Dict[str, ResourceState] = {}
        self.timeline: Dict[str, List[Tuple[datetime, datetime, str, float]]] = defaultdict(list)
        self.skus = skus
        self.log_entries = log_entries
        self.task_lookup = task_lookup

        # A combined dictionary to easily look up resource objects by ID
        self.resource_objects = {**tanks, **lines, **equipments, **rooms}

        # Initialize state for all known resources
        for res_id in self.resource_objects.keys():
            self.resource_states[res_id] = ResourceState(resource_id=res_id)
        self.log_entries.append(f"[ResourceManager] Initialized with {len(self.resource_states)} resources.")

    def get_state(self, resource_id: str) -> Optional[ResourceState]:
        """Safely returns the current state of a resource."""
        return self.resource_states.get(resource_id)

    def is_available(self, resource_id: str, start_time: datetime, end_time: datetime) -> bool:
        """Checks if a resource has any booking collisions during a time window."""
        # A small buffer to avoid floating point issues with datetimes
        end_time_exclusive = end_time - timedelta(seconds=1)

        for booked_start, booked_end, task_id, _ in self.timeline.get(resource_id, []):
            if booked_start < end_time and booked_end > start_time:
                self.log_entries.append(f"    [RM-Check] Collision on {resource_id}: requested slot {start_time}-{end_time} conflicts with {task_id} ({booked_start}-{booked_end})")
                return False
        return True

    def is_capacity_available(self, resource_id: str, start_time: datetime, end_time: datetime, required_capacity: float) -> bool:
        """
        Checks if a SHARED resource (like a Room) has enough EUI capacity available
        during a time window.
        """
        room = self.resource_objects.get(resource_id)
        if not room or not hasattr(room, 'capacity_units'):
            self.log_entries.append(f"    [RM-Capacity-Check] ERROR: Resource {resource_id} is not a valid room with capacity.")
            return False # Not a valid room

        total_room_capacity = room.capacity_units
        
        # Find the peak capacity usage during the requested time window
        peak_usage = 0
        
        # We need to check the capacity usage at every point a new task starts or ends within our window
        # to find the moment of maximum concurrent usage.
        
        # Get all overlapping tasks
        overlapping_tasks = [
            (s, e, t, cap) for s, e, t, cap in self.timeline.get(resource_id, [])
            if s < end_time and e > start_time
        ]

        if not overlapping_tasks:
            return total_room_capacity >= required_capacity

        # Find the highest concurrent capacity usage within the interval
        # This is a simplified check for peak concurrency. A more advanced model might check every minute.
        max_concurrent_capacity = 0
        
        # Check at the start time of the new task
        current_usage_at_start = sum(cap for s, e, t, cap in overlapping_tasks if s <= start_time < e)
        max_concurrent_capacity = max(max_concurrent_capacity, current_usage_at_start)
        
        # Check just after each existing task starts within the window
        for s_check, _, _, _ in overlapping_tasks:
            if start_time < s_check < end_time:
                usage_at_this_point = sum(cap for s, e, t, cap in overlapping_tasks if s <= s_check < e)
                max_concurrent_capacity = max(max_concurrent_capacity, usage_at_this_point)
                
        available_capacity = total_room_capacity - max_concurrent_capacity
        
        if available_capacity >= required_capacity:
            return True
        else:
            self.log_entries.append(f"    [RM-Capacity-Check] FAILED on {resource_id}: Required {required_capacity:.2f} EUI, but only {available_capacity:.2f} is available.")
            return False


    
    def get_required_setup_minutes(self, resource_id: str, new_product_category: str, start_time: datetime) -> int:
        """
        Determines the necessary setup or CIP time based on the resource's state.
        Returns duration in minutes. Now includes a check for max continuous runtime.

        Args:
            resource_id (str): The ID of the resource to check.
            new_product_category (str): The category of the product about to be scheduled.
            start_time (datetime): The proposed start time for the new task, used for runtime check.
        """
        state = self.resource_states[resource_id]
        resource_obj = self.resource_objects.get(resource_id)
        cip_duration = getattr(resource_obj, 'CIP_duration_minutes', 90)
        
        # Rule 1: Product Changeover (Highest Priority)
        # If the resource is dirty with a different product, a major clean is always needed.
        if state.status == ResourceStatus.DIRTY:
            if state.current_contents and state.current_contents != new_product_category:
                self.log_entries.append(f"    [RM-Setup] CIP required on {resource_id}: Was '{state.current_contents}', needs '{new_product_category}'. Duration: {cip_duration} min.")
                return cip_duration
                
        # Rule 2: Max Continuous Runtime for Lines and Equipment
        # This rule applies if there is no product changeover.
        if isinstance(resource_obj, (Line, Equipment)):
            MAX_RUNTIME_MINUTES = 480  # 8 hours, hardcoded as requested

            # Find the end time of the last CIP on this resource
            last_cip_end_time = None
            for s, e, task_id, _ in reversed(self.timeline.get(resource_id, [])):
                task = self.task_lookup.get(task_id) # Assumes task_lookup is available
                if task and task.task_type == TaskType.CIP:
                    last_cip_end_time = e
                    break
            
            # If there has never been a CIP, we check from the start of the schedule
            if last_cip_end_time is None:
                # To be safe, let's find the first production task instead of assuming schedule start
                first_prod_task_start = start_time
                for s, e, task_id, _ in self.timeline.get(resource_id, []):
                    task = self.task_lookup.get(task_id)
                    if task and task.task_type != TaskType.CIP:
                        first_prod_task_start = s
                        break
                last_cip_end_time = first_prod_task_start


            # Calculate the continuous runtime up to the proposed start of the new task
            continuous_runtime_minutes = (start_time - last_cip_end_time).total_seconds() / 60

            if continuous_runtime_minutes > MAX_RUNTIME_MINUTES:
                self.log_entries.append(f"    [RM-Setup] Max runtime CIP required on {resource_id}. Has run for {continuous_runtime_minutes:.0f} mins. Duration: {cip_duration} min.")
                return cip_duration

        # If neither of the above rules trigger, no setup/CIP is needed.
        return 0


    def commit_task_to_timeline(self, task: HeuristicTask, resource_id: str, start_time: datetime, end_time: datetime):
        """
        Books a task on a resource's timeline and updates the resource's final state.
        This now includes calculating and storing the capacity consumed.
        """
        # --- START: NEW LOGIC TO BE ADDED ---
        capacity_consumed = 0.0
        resource = self.resource_objects.get(resource_id)
        
        if isinstance(resource, Room):
            sku = self.skus.get(task.sku_id)
            if sku and hasattr(sku, 'inventory_size'):
                # This is the key calculation: volume * EUI_per_kg
                capacity_consumed = task.volume_liters * sku.inventory_size
        elif resource:
            capacity_consumed = 1.0 

        self.timeline[resource_id].append((start_time, end_time, task.task_id, capacity_consumed))
        # --- END KEY LINE CHANGE ---
        
        self.timeline[resource_id].sort()
        self.log_entries.append(f"    [RM-Commit] Booking {task.task_id} on {resource_id} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} consuming {capacity_consumed:.2f} EUI.")

        # The rest of the state update logic remains the same
        state = self.resource_states[resource_id]
        
        if task.task_type == TaskType.CIP:
            state.status = ResourceStatus.IDLE
            state.current_contents = None
            state.dirty_since = None
            self.log_entries.append(f"    [RM-State] Resource {resource_id} is now IDLE and clean.")
        
        elif task.task_type in [TaskType.BULK_PRODUCTION, TaskType.FINISHING]:
            state.status = ResourceStatus.DIRTY
            state.task_id = task.task_id
            product_category = None
            if task.task_type == TaskType.BULK_PRODUCTION:
                product_category = task.sku_id
            elif task.task_type == TaskType.FINISHING:
                if task.sku_id in self.skus:
                    product_category = self.skus[task.sku_id].product_category
            
            state.current_contents = product_category
            state.dirty_since = end_time
            self.log_entries.append(f"    [RM-State] Resource {resource_id} is now DIRTY with {product_category}.")

class HeuristicScheduler:
    def __init__(self, indents, skus, products, lines, tanks, equipments, shifts):
        
        self.indents = indents
        self.skus = skus
        self.products = products
        self.lines = lines
        self.tanks = tanks
        self.equipments = equipments
        self.shifts = shifts
        self.rooms = config.ROOMS
        self.cip_circuits = config.CIP_CIRCUIT
        self.TIME_BLOCK_MINUTES = 15
        self.MAX_MASTER_BATCH_SIZE = 10000
        self.log_entries: List[str] = []
        self.task_lookup: Dict[str, HeuristicTask] = {}
        self.log_entries.append(f'Indent recieved: {indents} ******')


        # --- RESOURCE MANAGER INTEGRATION ---
        self.resource_manager = ResourceManager(
            tanks=self.tanks, 
            lines=self.lines, 
            equipments=self.equipments, 
            rooms = self.rooms,
            skus=self.skus,
            log_entries=self.log_entries,
            task_lookup=self.task_lookup
        )
        self.time_manager = TimeManager(self.shifts, self.TIME_BLOCK_MINUTES)
        self.schedule_start_dt = datetime.combine(datetime.now().date() + timedelta(days=1), time(22, 0))
        self.schedule_end_dt = self.schedule_start_dt + timedelta(days=2)
        self.schedule_start_token = 0
        self.schedule_end_token = self._to_tokens(self.schedule_end_dt)
        
        self.master_task_list: List[HeuristicTask] = []
        
        self.resource_timelines: Dict[str, List[Tuple[datetime, datetime, str]]] = self.resource_manager.timeline
        self.weights = {"priority": 1000, "due_date_urgency": 500}

    def _to_tokens(self, dt: datetime) -> int:
        """Converts a datetime object to an integer token count from the schedule start."""
        if dt < self.schedule_start_dt:
            return 0
        delta = dt - self.schedule_start_dt
        return int(delta.total_seconds() / 60 / self.TIME_BLOCK_MINUTES)

    def _to_datetime(self, tokens: int) -> datetime:
        """Converts an integer token count back to a datetime object."""
        minutes_to_add = tokens * self.TIME_BLOCK_MINUTES
        return self.schedule_start_dt + timedelta(minutes=minutes_to_add)

    def _round_up_duration_to_tokens(self, duration_minutes: int) -> int:
        """Rounds a duration in minutes up to the nearest full time block."""
        if duration_minutes == 0:
            return 0
        return (duration_minutes + self.TIME_BLOCK_MINUTES - 1) // self.TIME_BLOCK_MINUTES
    
    def _split_indents_into_sub_orders(self) -> List[SubOrder]:
        """
        Prepares sub-orders by converting indents directly into SubOrders
        without pre-splitting. Batching is handled separately via draw-down logic.
        """
        sub_orders = []
        for indent in self.indents.values():
            sub_orders.append(SubOrder(
                parent_order_no=indent.order_no,
                sub_order_id=f"{indent.order_no}-part1",
                sku_id=indent.sku_id,
                volume=indent.qty_required_liters,
                priority=indent.priority.value,
                due_date=indent.due_date
            ))
        return sub_orders

    def _calculate_stage_capacities(self) -> float:
        """
        Calculates the bottleneck capacity of the entire system.
        1. Dynamically discovers which resources belong to each stage using index-based boundaries.
        2. For each stage, calculates its bottleneck as min(Volume Capacity, Rate Capacity).
        3. Applies a downstream constraint and returns the final system bottleneck value.
        """
        print("[INFO] Dynamically calculating stage bottleneck capacities using index-based rules...")
        
        # --- 1. Discover resources for each stage using the definitive index-based logic ---
        resources_by_stage = defaultdict(set)
        steps_by_stage = defaultdict(list)

        for product in self.products.values():
            steps = product.processing_steps
            
            last_prep_idx = next((i for i, step in reversed(list(enumerate(steps))) if step.process_type == ProcessType.PREPROCESSING), -1)
            first_pack_idx = next((i for i, step in enumerate(steps) if step.process_type == ProcessType.PACKAGING), -1)

            if last_prep_idx == -1 or first_pack_idx == -1:
                print(f"[WARNING] Product '{product.product_category}' is missing PREPROCESSING or PACKAGING steps. Skipping for capacity calculation.")
                continue

            for i, step in enumerate(steps):
                # Use the new terminology
                stage = None
                if i <= last_prep_idx:
                    stage = 'Preprocessing'
                elif i < first_pack_idx:
                    stage = 'Processing'
                elif i == first_pack_idx:
                    stage = 'Anchor' # Formerly Finishing
                else: # i > first_pack_idx
                    stage = 'Post-processing' # Formerly FIFO

                steps_by_stage[stage].append(step)
                for req in step.requirements:
                    for resource_id in req.compatible_ids:
                        resources_by_stage[stage].add(resource_id)

        # --- 2. Calculate the capacity of each stage (Volume vs. Rate) ---
        stage_capacities = defaultdict(float)
        TOTAL_SHIFT_MINUTES = 24 * 60 # Assume a 24-hour planning horizon

        for stage_name, resource_ids in resources_by_stage.items():
            total_volume_capacity = 0
            total_rate_capacity = 0
            
            # Use an average setup time for the stage for rate calculations
            stage_steps = steps_by_stage[stage_name]
            setup_times = [getattr(s, 'setup_time', 0) for s in stage_steps]
            avg_setup_time = sum(setup_times) / len(setup_times) if setup_times else 0
            usable_time_minutes = TOTAL_SHIFT_MINUTES - avg_setup_time

            for resource_id in resource_ids:
                if resource_id in self.tanks:
                    total_volume_capacity += self.tanks[resource_id].capacity
                
                elif resource_id in self.lines:
                    line = self.lines[resource_id]
                    if line.compatible_skus_max_production:
                        avg_speed = sum(line.compatible_skus_max_production.values()) / len(line.compatible_skus_max_production)
                        total_rate_capacity += (usable_time_minutes * avg_speed)

                elif resource_id in self.equipments:
                    equipment = self.equipments[resource_id]
                    speed = getattr(equipment, 'processing_speed', 0)
                    if speed > 0:
                        total_rate_capacity += (usable_time_minutes * speed)
            
            print(f"[CAPACITY] Stage '{stage_name}' raw Volume: {total_volume_capacity}L, raw Rate: {total_rate_capacity:.0f}L/day")

            # The "faucet and bathtub" principle: the stage's capacity is the minimum of the two.
            if total_volume_capacity > 0 and total_rate_capacity > 0:
                stage_capacities[stage_name] = min(total_volume_capacity, total_rate_capacity)
            elif total_volume_capacity > 0:
                stage_capacities[stage_name] = total_volume_capacity
            else:
                stage_capacities[stage_name] = total_rate_capacity

        # --- 3. Apply Downstream Constraint & Find Final Bottleneck ---
        final_capacities = {}
        # Apply constraint in the order of the process flow
        process_flow_order = ['Preprocessing', 'Processing', 'Anchor', 'Post-processing']
        last_stage_capacity = float('inf')

        for stage in process_flow_order:
            if stage in stage_capacities:
                current_capacity = stage_capacities[stage]
                constrained_capacity = min(last_stage_capacity, current_capacity)
                final_capacities[stage] = constrained_capacity
                last_stage_capacity = constrained_capacity
        
        if not final_capacities: return 0.0

        # Log the final constrained capacities
        for stage, cap in final_capacities.items():
            print(f"[BOTTLENECK] Stage '{stage}' constrained capacity: {cap:.2f}L.")
        
        overall_bottleneck_value = min(final_capacities.values())
        print(f"[BOTTLENECK] FINAL SYSTEM BOTTLENECK identified as {overall_bottleneck_value:.2f}L/day.")

        return overall_bottleneck_value

    def _create_master_bus_plan(self, bottleneck_capacity: float) -> List[Dict]:
        """
        Creates a hierarchical production plan, respecting the overall system bottleneck capacity.
        """
        self.log_entries.append(f"[INFO] Creating allocation plan with a capacity cap of {bottleneck_capacity:.2f}L.")
        master_plan = []

        mst_capacity = 10000
        lt_capacity = 5000

        
        orders_by_product = defaultdict(list)
        for indent in self.indents.values():
            if indent.due_date < self.schedule_start_dt: continue
            sku = self.skus.get(indent.sku_id)
            if sku:
                orders_by_product[sku.product_category].append(indent) # Aggregation of indent by Product Category

        total_volume_planned = 0
        master_bus_counter = 0
        for product_cat, orders in orders_by_product.items():
            
            remaining_demand = {order.order_no: order.qty_required_liters for order in orders}
            
            while sum(remaining_demand.values()) > 0.1 and total_volume_planned < bottleneck_capacity:
                master_bus_id = f"MST-bus-{master_bus_counter}"
                
                # The volume of this master bus is the smaller of demand or MST capacity
                available_capacity = bottleneck_capacity - total_volume_planned
                master_bus_total_volume = min(sum(remaining_demand.values()), mst_capacity)
                volume_allocated_to_master_bus = 0
                
                sub_buses = []
                
                # --- THIS IS THE KEY CHANGE ---
                # Instead of a fixed loop, create LT-sized sub-buses until the master bus volume is filled.
                while volume_allocated_to_master_bus < master_bus_total_volume:
                    
                    # Stop if there's no more demand to fulfill
                    if sum(remaining_demand.values()) < 0.1: break

                    sub_bus_id = f"LT-bus-{chr(ord('A') + len(sub_buses))}-from-{master_bus_id}"
                    
                    # Sub-bus capacity is the smaller of LT capacity or remaining master bus volume
                    sub_bus_capacity = min(lt_capacity, master_bus_total_volume - volume_allocated_to_master_bus)

                    # Allocate order volumes to this specific Sub-Bus
                    allocations = {}
                    current_sub_bus_volume = 0
                    for order_no, rem_vol in remaining_demand.items():
                        if rem_vol <= 0: continue
                        
                        volume_to_take = min(rem_vol, sub_bus_capacity - current_sub_bus_volume)
                        
                        if volume_to_take > 0:
                            allocations[order_no] = allocations.get(order_no, 0) + volume_to_take
                            remaining_demand[order_no] -= volume_to_take
                            current_sub_bus_volume += volume_to_take

                        if current_sub_bus_volume >= sub_bus_capacity: break
                    
                    if current_sub_bus_volume > 0:
                        sub_buses.append({
                            "sub_bus_id": sub_bus_id,
                            "volume": current_sub_bus_volume,
                            "allocations": allocations
                        })
                        volume_allocated_to_master_bus += current_sub_bus_volume
                
                if sub_buses:
                    master_plan.append({
                        "master_bus_id": master_bus_id,
                        "product_category": product_cat,
                        "volume": volume_allocated_to_master_bus,
                        "sub_buses": sub_buses
                    })
                master_bus_counter += 1
                
        return master_plan

    def _group_sub_orders_into_batches(self, sub_orders: List[SubOrder]) -> Dict:
        """
        Groups sub-orders into fixed-size bulk batches using proportional draw-down logic,
        weighted by packing speed and grouped by product category.
        """
        from collections import defaultdict
        batches = defaultdict(lambda: {"total_volume": 0, "sub_orders": []})

        # Group sub-orders by product category
        sub_orders_by_category = defaultdict(list)
        for so in sub_orders:
            category = self.skus[so.sku_id].product_category
            sub_orders_by_category[category].append(so)

        batch_counter = 0
        for category, sos in sub_orders_by_category.items():
            # Aggregate demand per SKU
            demand_by_sku = defaultdict(float)
            speed_by_sku = {}
            for so in sos:
                demand_by_sku[so.sku_id] += so.volume
                if so.sku_id not in speed_by_sku:
                    speed_by_sku[so.sku_id] = self.skus[so.sku_id].packing_speed_kg_per_min

            # Map SKU to its sub-orders (FIFO allocation)
            so_queue_by_sku = defaultdict(list)
            for so in sos:
                so_queue_by_sku[so.sku_id].append(so)

            # Draw-down loop: allocate into fixed-size buses
            while sum(demand_by_sku.values()) > 0:
                remaining_capacity = self.MAX_BATCH_SIZE
                total_speed = sum(speed_by_sku[sku] for sku in demand_by_sku if demand_by_sku[sku] > 0)
                allocation = defaultdict(float)

                # Compute proportional allocation
                for sku, demand in demand_by_sku.items():
                    if demand == 0:
                        continue
                    proportion = speed_by_sku[sku] / total_speed
                    alloc = min(proportion * self.MAX_BATCH_SIZE, demand)
                    allocation[sku] = alloc

                # Apply allocation by pulling sub-orders
                bus_sub_orders = []
                for sku, alloc_volume in allocation.items():
                    demand_by_sku[sku] -= alloc_volume
                    while alloc_volume > 0 and so_queue_by_sku[sku]:
                        so = so_queue_by_sku[sku][0]
                        take_volume = min(so.volume, alloc_volume)

                        # Split if necessary
                        if take_volume < so.volume:
                            remaining_volume = so.volume - take_volume
                            so.volume = take_volume
                            new_so = SubOrder(
                                parent_order_no=so.parent_order_no,
                                sub_order_id=so.sub_order_id + "-split",
                                sku_id=so.sku_id,
                                volume=remaining_volume,
                                priority=so.priority,
                                due_date=so.due_date
                            )
                            so_queue_by_sku[sku][0] = new_so  # Replace with leftover
                        else:
                            so_queue_by_sku[sku].pop(0)  # Fully consumed

                        alloc_volume -= take_volume
                        bus_sub_orders.append(so)

                # Register batch
                batch_id = f"{category}-bus{batch_counter}"
                total_vol = sum(so.volume for so in bus_sub_orders)
                batches[batch_id]["sub_orders"] = bus_sub_orders
                batches[batch_id]["total_volume"] = total_vol
                batch_counter += 1

        return batches

    def run_heuristic_scheduler(self):
            """
            Main engine that implements a forward-chaining, state-aware, transactional
            scheduling model with support for task splitting.
            """
            self.log_entries.append("--- State-Aware Heuristic Scheduler Run Started ---")

            self.log_entries.append("Step 1: Generating all tasks...")
            self.generate_all_tasks()
            self.log_entries.append(f"-> {len(self.master_task_list)} tasks generated.")

            self.log_entries.append("\nStep 2: Finding ready tasks and scheduling them transactionally...")
            
            max_loops = len(self.master_task_list) * 3 # Increased max_loops to allow for splits
            for i in range(max_loops):
                if not any(t.status == ScheduleStatus.PENDING for t in self.master_task_list):
                    self.log_entries.append(f"\n-> All tasks scheduled after {i} iterations.")
                    break

                ready_tasks = self._get_ready_tasks()
                if not ready_tasks:
                    if any(t.status == ScheduleStatus.PENDING for t in self.master_task_list):
                        self.log_entries.append("[ERROR] Deadlock detected. No ready tasks found but pending tasks still exist.")
                    break
                
                task_to_schedule = self._score_and_select_best_task(ready_tasks)
                self.log_entries.append(f"  [{i+1:>3}] Scheduling {task_to_schedule.task_id}...")

                # STRATEGY: First, try to book the entire task.
                placement = self._find_and_book_transaction(task_to_schedule)
                
                # If that fails, and it's a finish task, try splitting it.
                if not placement and task_to_schedule.task_type == TaskType.FINISHING:
                    placement = self._split_and_schedule_first_part(task_to_schedule)

                # If both attempts fail, mark it as failed for this run.
                if not placement and task_to_schedule.status == ScheduleStatus.PENDING:
                    task_to_schedule.status = ScheduleStatus.FAILED
                    self.log_entries.append(f"    -> FAILED: Could not place {task_to_schedule.task_id}, even with splitting.")

            else:
                self.log_entries.append(f"\n[WARNING] Scheduling loop reached max iterations. Check for unresolved dependencies.")
            
            self.log_entries.append("\n--- Heuristic Scheduling Complete ---")
            self.log_entries.append("\nStep 3: Scheduling mandatory cleanup tasks...")
            self._schedule_drained_tank_cip()
            self._schedule_final_equipment_cip()
            self.log_entries.append("-> Cleanup CIP scheduling complete.")
            
            self._report_unscheduled_tasks()
            self.write_schedule_log_file()
            return self._create_scheduling_result_for_export()

    def _find_and_book_transaction(self, task: HeuristicTask) -> Optional[Dict]:
        """
        Finds the BEST placement for a FULL task by checking all compatible resources
        and selecting the one that offers the earliest finish time.
        This is now aware of different resource types and handles the ZERO_STAGNATION
        rule correctly for pipeline sub-tasks.
        """
        earliest_start_dt = self._get_prerequisite_finish_time(task)
        
        if not task.step or not task.step.requirements:
            self.log_entries.append(f"    [FAIL] Task {task.task_id} has no defined step or resource requirements.")
            return None
        primary_req = task.step.requirements[0]
        compatible_resources = task.compatible_resources.get(primary_req.resource_type, [])
        if not compatible_resources:
            self.log_entries.append(f"    [FAIL] Task {task.task_id} has no compatible resources for type {primary_req.resource_type}.")
            return None
        product_cat = self.skus.get(task.sku_id).product_category if task.sku_id in self.skus else task.sku_id
        if not product_cat:
            self.log_entries.append(f"    [FAIL] Could not determine product category for task {task.task_id}.")
            return None

        best_placement = None
        best_finish_time = datetime.max

        for resource_id in compatible_resources:
            self.log_entries.append(f"    - Evaluating resource {resource_id} for task {task.task_id}...")
            
            resource_obj = self.resource_manager.resource_objects.get(resource_id)
            if not resource_obj:
                continue
            
            search_dt = earliest_start_dt
            cip_minutes = self.resource_manager.get_required_setup_minutes(resource_id, product_cat, search_dt)
            cip_duration = timedelta(minutes=cip_minutes)
            task_duration = timedelta(minutes=task.base_duration_tokens * self.TIME_BLOCK_MINUTES)
            
            
            while search_dt < self.schedule_end_dt:
                transaction_start = search_dt
                cip_end = transaction_start + cip_duration
                task_end = cip_end + task_duration

                if not self.time_manager._is_in_shift(cip_end) or not self.time_manager._is_in_shift(task_end - timedelta(seconds=1)):
                    search_dt += timedelta(minutes=self.TIME_BLOCK_MINUTES)
                    continue

                is_slot_available = False
                source_resource_id = None # For Zero Stagnation
                
                # --- START: NEW, SMARTER ZERO_STAGNATION LOGIC ---
                is_zero_stag = getattr(task.step, 'scheduling_rule', SchedulingRule.DEFAULT) == SchedulingRule.ZERO_STAGNATION
                is_pipeline_subtask = "-p" in task.task_id and task.task_type == TaskType.FINISHING

                if is_zero_stag and task.previous_task and task.previous_task.assigned_resource_id:
                    source_resource_id = task.previous_task.assigned_resource_id
                    
                    # Check availability of the main resource (the line)
                    line_is_available = self.resource_manager.is_available(resource_id, cip_end, task_end)
                    
                    if not line_is_available:
                        search_dt += timedelta(minutes=self.TIME_BLOCK_MINUTES)
                        continue

                    # If it's a pipeline sub-task, we only lock the source for the *first* part
                    if is_pipeline_subtask:
                        if task.task_id.endswith("-p1"):
                            # This is the first part. Find all sibling parts to calculate total drain time.
                            job_prefix = task.task_id.rsplit('-', 1)[0]
                            sibling_tasks = [t for t in self.master_task_list if t.task_id.startswith(job_prefix)]
                            total_drain_duration = timedelta(minutes=len(sibling_tasks) * self.TIME_BLOCK_MINUTES)
                            drain_end_time = cip_end + total_drain_duration
                            
                            # Check if the source tank is free for the whole duration
                            is_slot_available = self.resource_manager.is_available(source_resource_id, cip_end, drain_end_time)
                        else:
                            # This is a subsequent part (-p2, -p3...). The source tank is already locked.
                            # We only need to check the line, which we already did.
                            is_slot_available = True
                    else:
                        # Standard Zero Stagnation for non-pipeline tasks
                        is_slot_available = self.resource_manager.is_available(source_resource_id, cip_end, task_end)
                
                elif isinstance(resource_obj, Room):
                    sku = self.skus.get(task.sku_id)
                    if sku and hasattr(sku, 'inventory_size') and sku.inventory_size > 0:
                        required_eui = task.volume_liters * sku.inventory_size
                        is_slot_available = self.resource_manager.is_capacity_available(resource_id, cip_end, task_end, required_eui)
                    else:
                        self.log_entries.append(f"    [FAIL] Cannot calculate EUI for {task.sku_id} on Room {resource_id}.")
                        break
                else:
                    # Standard check for all other resources
                    is_slot_available = self.resource_manager.is_available(resource_id, transaction_start, task_end)
                # --- END: NEW LOGIC ---

                if not is_slot_available:
                    search_dt += timedelta(minutes=self.TIME_BLOCK_MINUTES)
                    continue

                if task_end < best_finish_time:
                    self.log_entries.append(f"    * New best placement found on {resource_id} finishing at {task_end.strftime('%H:%M')}.")
                    best_finish_time = task_end
                    best_placement = {
                        "task": task, "resource_id": resource_id,
                        "transaction_start": transaction_start, "cip_end": cip_end, "task_end": task_end,
                        "cip_duration": cip_duration, "source_resource_id": source_resource_id
                    }
                break
        
        if best_placement:
            bp = best_placement
            task_to_commit = bp['task']
            self.log_entries.append(f"    + Committing best placement for {task_to_commit.task_id} on {bp['resource_id']}.")

            if bp['cip_duration'].total_seconds() > 0:
                cip_task_id = f"CIP-for-{task_to_commit.task_id}"
                cip_task = HeuristicTask(task_id=cip_task_id, job_id=task_to_commit.job_id, sku_id="CIP", step=task_to_commit.step, task_type=TaskType.CIP, batch_idx=task_to_commit.batch_idx)
                self.resource_manager.commit_task_to_timeline(cip_task, bp['resource_id'], bp['transaction_start'], bp['cip_end'])
                self.task_lookup[cip_task_id] = cip_task

            self.resource_manager.commit_task_to_timeline(task_to_commit, bp['resource_id'], bp['cip_end'], bp['task_end'])
            task_to_commit.status = ScheduleStatus.BOOKED
            task_to_commit.assigned_resource_id = bp['resource_id']
            task_to_commit.start_time = bp['cip_end']
            task_to_commit.end_time = bp['task_end']
            
            # --- START: NEW, SMARTER SOURCE LOCKING ---
            if bp['source_resource_id']:
                # For pipeline tasks, only lock the source for the first part (-p1)
                if is_pipeline_subtask and task_to_commit.task_id.endswith("-p1"):
                    job_prefix = task_to_commit.task_id.rsplit('-', 1)[0]
                    sibling_tasks = [t for t in self.master_task_list if t.task_id.startswith(job_prefix)]
                    total_drain_duration = timedelta(minutes=len(sibling_tasks) * self.TIME_BLOCK_MINUTES)
                    drain_end_time = bp['cip_end'] + total_drain_duration
                    
                    locked_task_id = f"LOCKED-for-{job_prefix}"
                    locked_task = HeuristicTask(task_id=locked_task_id, job_id=task_to_commit.job_id, sku_id="LOCKED", step=task_to_commit.step, task_type=TaskType.LOCKED, batch_idx=task_to_commit.batch_idx)
                    self.resource_manager.commit_task_to_timeline(locked_task, bp['source_resource_id'], bp['cip_end'], drain_end_time)
                elif not is_pipeline_subtask:
                    # Standard locking for non-pipeline tasks
                    locked_task_id = f"LOCKED-for-{task_to_commit.task_id}"
                    locked_task = HeuristicTask(task_id=locked_task_id, job_id=task_to_commit.job_id, sku_id="LOCKED", step=task_to_commit.step, task_type=TaskType.LOCKED, batch_idx=task_to_commit.batch_idx)
                    self.resource_manager.commit_task_to_timeline(locked_task, bp['source_resource_id'], bp['cip_end'], bp['task_end'])
            # --- END: NEW SOURCE LOCKING ---
            
            return {"resource_id": bp['resource_id'], "start_time": bp['cip_end'], "end_time": bp['task_end']}
        
        return None
    
    def _find_next_available_slot(self, resource_id: str, search_after: datetime) -> Optional[Tuple[datetime, datetime]]:
        """Finds the next available time slot for a resource after a given time."""
        timeline = self.resource_manager.timeline.get(resource_id, [])
        current_time = search_after
        
        if not timeline:
            if current_time < self.schedule_end_dt:
                return (current_time, self.schedule_end_dt)
            return None

        for i, (start, end, *_) in enumerate(timeline):
            if current_time < end:
                current_time = max(current_time, end)
            
            next_start = timeline[i+1][0] if i + 1 < len(timeline) else self.schedule_end_dt
            if current_time < next_start:
                return (current_time, next_start)

        if current_time < self.schedule_end_dt:
            return (current_time, self.schedule_end_dt)
            
        return None

    def _split_and_schedule_first_part(self, task: HeuristicTask) -> Optional[Dict]:
        """
        Takes a task that failed full placement, finds the first available gap,
        and schedules a 'Part 1' that is sized to fit within a single shift.
        """
        earliest_start_dt = self._get_prerequisite_finish_time(task)
        primary_req = task.step.requirements[0]
        compatible_resources = task.compatible_resources.get(primary_req.resource_type, [])

        self.log_entries.append(f"    [SPLIT] Attempting to split {task.task_id}.")
        
        for resource_id in compatible_resources:
            # Start searching for a valid, shift-compliant slot from its earliest possible start time
            search_dt = earliest_start_dt
            while search_dt < self.schedule_end_dt:
                slot = self._find_next_available_slot(resource_id, search_dt)
                if not slot: break # No more slots on this resource

                slot_start, slot_end = slot
                
                # Find the end of the shift that the slot starts in
                shift_end_dt = self.time_manager.get_shift_end(slot_start)
                if not shift_end_dt: # The slot starts outside of a valid shift, so advance search
                    search_dt = slot_end if slot_end > slot_start else search_dt + timedelta(minutes=self.TIME_BLOCK_MINUTES)
                    continue

                # The maximum end time for our new part is the earlier of the slot end or the shift end
                max_end_dt = min(slot_end, shift_end_dt)
                
                setup_minutes = getattr(task.step, 'setup_time', 0)
                available_duration_minutes = (max_end_dt - slot_start).total_seconds() / 60

                if available_duration_minutes <= setup_minutes:
                    search_dt = max_end_dt # This slot is too small, check after it
                    continue

                processing_time_minutes = available_duration_minutes - setup_minutes
                
                res = self.resource_manager.resource_objects.get(resource_id)
                speed = res.compatible_skus_max_production.get(task.sku_id) if hasattr(res, 'compatible_skus_max_production') else getattr(res, 'processing_speed', 0)
                
                if not speed or speed <= 0: break # Break from while loop, try next resource

                volume_for_part1 = min(task.volume_liters, processing_time_minutes * speed)
                
                # Don't schedule a trivially small part
                if volume_for_part1 <= 1: 
                    search_dt = max_end_dt
                    continue

                part1_duration_minutes = setup_minutes + (volume_for_part1 / speed)
                part1_end_time = slot_start + timedelta(minutes=part1_duration_minutes)

                # Final check to ensure no weird rounding pushed it over the shift boundary
                if part1_end_time > max_end_dt + timedelta(seconds=1):
                     search_dt = max_end_dt
                     continue

                # --- We found a valid, shift-compliant piece. Book it. ---
                part1_task_id = f"{task.task_id}-p1"
                part1_task = HeuristicTask(
                    task_id=part1_task_id, job_id=task.job_id, sku_id=task.sku_id, step=task.step,
                    task_type=task.task_type, volume_liters=volume_for_part1, priority=task.priority,
                    compatible_resources=task.compatible_resources, prerequisites=task.prerequisites
                )

                self.resource_manager.commit_task_to_timeline(part1_task, resource_id, slot_start, part1_end_time)
                part1_task.status = ScheduleStatus.BOOKED
                part1_task.assigned_resource_id = resource_id
                part1_task.start_time = slot_start
                part1_task.end_time = part1_end_time
                self.master_task_list.append(part1_task)
                self.task_lookup[part1_task.task_id] = part1_task
                self.log_entries.append(f"    [SPLIT] Booked {part1_task.task_id} for {volume_for_part1:.0f}L.")
                
                remaining_volume = task.volume_liters - volume_for_part1
                if remaining_volume > 1:
                    part2_task_id = f"{task.task_id}-p2"
                    part2_task = HeuristicTask(
                        task_id=part2_task_id, job_id=task.job_id, sku_id=task.sku_id, step=task.step,
                        task_type=task.task_type, volume_liters=remaining_volume, priority=task.priority,
                        compatible_resources=task.compatible_resources, prerequisites=[part1_task]
                    )
                    self.master_task_list.append(part2_task)
                    self.task_lookup[part2_task.task_id] = part2_task
                    self.log_entries.append(f"    [SPLIT] Created {part2_task.task_id} for remaining {remaining_volume:.0f}L.")

                task.status = ScheduleStatus.FAILED
                self.log_entries.append(f"    [SPLIT] Original task {task.task_id} marked as FAILED and replaced.")
                
                return {"resource_id": resource_id, "start_time": slot_start, "end_time": part1_end_time}

        return None

    def _create_allocation_plan(self, orders: List[UserIndent], campaign_prefix: str) -> List[Dict]:
        """
        Creates a robust production plan for a group of orders, ensuring all demand is met.
        """
        self.log_entries.append(f"[INFO] Creating allocation plan for campaign '{campaign_prefix}'.")
        master_plan = []
        master_batch_counter = 0

        # Create a mutable copy of the demand to track what's left
        remaining_demand = {o.order_no: o.qty_required_liters for o in orders}
        # Sort orders to handle them consistently
        sorted_orders = sorted(orders, key=lambda o: (o.priority.value, o.due_date))

        # Loop until all demand in this group is allocated into batches
        while sum(remaining_demand.values()) > 0.1:
            bus_id = f"{campaign_prefix}-master{master_batch_counter}"
            bus_capacity = self.MAX_MASTER_BATCH_SIZE
            bus_allocations = defaultdict(float)
            allocated_to_bus = 0.0

            # Fill the current batch up to its capacity
            for order in sorted_orders:
                order_no = order.order_no
                if remaining_demand[order_no] > 0:
                    volume_to_take = min(remaining_demand[order_no], bus_capacity - allocated_to_bus)
                    if volume_to_take > 0:
                        bus_allocations[order_no] += volume_to_take
                        allocated_to_bus += volume_to_take
                        remaining_demand[order_no] -= volume_to_take
                
                if allocated_to_bus >= bus_capacity:
                    break
            
            # If the batch has volume, add it to the master plan
            if allocated_to_bus > 0:
                # Find the primary product category for naming/logging purposes
                first_order_no = next(iter(bus_allocations))
                product_cat = self.skus[self.indents[first_order_no].sku_id].product_category

                master_plan.append({
                    "batch_id": bus_id,
                    "product_category": product_cat,
                    "volume": allocated_to_bus,
                    "allocations": dict(bus_allocations)
                })
            master_batch_counter += 1
        
        return master_plan
    
    # Add this new method inside the HeuristicScheduler class

    def _get_compatible_resources_for_step(self, sku_id: str, step: ProcessingStep) -> Dict[ResourceType, List[str]]:
        """
        Finds the specific resources that are compatible with a given SKU for a step.
        This is especially important for filtering lines based on SKU compatibility.
        """
        compatible_resources = defaultdict(list)
        if not step:
            return {}

        for req in step.requirements:
            # If the requirement is for a LINE, we must do a specific check
            if req.resource_type == ResourceType.LINE:
                filtered_line_ids = []
                for line_id in req.compatible_ids:
                    line = self.lines.get(line_id)
                    # The line is compatible only if the specific SKU is in its production list
                    if line and sku_id in getattr(line, 'compatible_skus_max_production', {}):
                        filtered_line_ids.append(line_id)
                compatible_resources[req.resource_type].extend(filtered_line_ids)
            else:
                # For all other resource types (Tanks, Rooms, etc.), we assume the list is correct
                compatible_resources[req.resource_type].extend(req.compatible_ids)
                
        return dict(compatible_resources)

    def generate_all_tasks(self):
        """
        Orchestrates task generation by first grouping orders by line type
        to create separate, parallel production campaigns.
        """
        self.log_entries.append("--- Task Generation: Parallel Campaign Mode ---")
        
        valid_indents = [i for i in self.indents.values() if i.due_date >= self.schedule_start_dt]
        
        # 1. Group all orders by the type of line they need
        orders_by_line_type = self._group_orders_by_line_type(valid_indents)

        for line_type, orders in orders_by_line_type.items():
            if not orders: continue
            
            self.log_entries.append(f"\n-- Generating campaign for Line Type: {line_type} --")
            
            # 2. Create a specific allocation plan for this group
            allocation_plan = self._create_allocation_plan(orders, line_type)
            
            self.log_entries.append(f"--- {line_type} Master Batch Allocation Breakdown ---")
            for batch_plan in allocation_plan:
                self.log_entries.append(
                    f"  -> Batch '{batch_plan['batch_id']}' | Vol: {batch_plan['volume']:.0f}L"
                )
            
            # 3. Create the task chains for this campaign
            for batch_plan in allocation_plan:
                final_bulk_tasks = self._create_bulk_production_chain(batch_plan)
                if not final_bulk_tasks: continue

                remaining_order_demand = batch_plan["allocations"].copy()
                cab_counter = sum(1 for t in self.master_task_list if t.task_type == TaskType.FINISHING)

                for source_task in final_bulk_tasks:
                    source_volume_left = source_task.volume_liters
                    
                    for order_no, demand in list(remaining_order_demand.items()):
                        if source_volume_left <= 0: break
                        if demand <= 0: continue

                        take_volume = min(demand, source_volume_left)
                        
                        indent = self.indents[order_no]
                        sub_order = SubOrder(
                            parent_order_no=order_no,
                            sub_order_id=f"{order_no}-cab{cab_counter}",
                            sku_id=indent.sku_id, volume=take_volume,
                            priority=indent.priority.value, due_date=indent.due_date,
                            master_batch_id=batch_plan["batch_id"]
                        )
                        self._create_finishing_chain(sub_order, source_task)
                        
                        remaining_order_demand[order_no] -= take_volume
                        source_volume_left -= take_volume
                        cab_counter += 1

        self._identify_and_prepare_anchors()
    
    def _get_effective_downstream_capacity(self, current_step_index: int, product: Product) -> float:
        """
        Finds the effective processing capacity for the next step in the chain.
        Handles the 'ZeroStagnation' rule by looking ahead.
        """
        if current_step_index + 1 >= len(product.processing_steps):
            return float('inf') # No next step, no capacity constraint

        next_step = product.processing_steps[current_step_index + 1]
        step_to_check = next_step
        
        # If the next step is a flow task, its capacity is defined by the step after it.
        if getattr(next_step, 'scheduling_rule', SchedulingRule.DEFAULT) == SchedulingRule.ZERO_STAGNATION:
            if current_step_index + 2 < len(product.processing_steps):
                step_to_check = product.processing_steps[current_step_index + 2]
            else:
                return float('inf') # Flow task is the last one, no constraint

        # Find the maximum capacity among all compatible resources for the determined step.
        max_capacity = 0
        for req in step_to_check.requirements:
            if req.resource_type == ResourceType.TANK:
                for tank_id in req.compatible_ids:
                    if tank_id in self.tanks:
                        max_capacity = max(max_capacity, self.tanks[tank_id].capacity_liters)
        
        # Return a very large number if no capacity is defined, effectively preventing a split.
        return max_capacity if max_capacity > 0 else float('inf')

    def _create_bulk_production_chain(self, batch_plan: Dict) -> List[HeuristicTask]:
            """
            Creates a potentially branching chain of bulk production tasks.
            It splits a batch into multiple smaller tasks if a downstream capacity bottleneck is detected.
            """
            product_category = batch_plan["product_category"]
            product = self.products.get(product_category)
            if not product: return []

            # Get only the bulk production steps (before packaging)
            first_pack_idx = next((i for i, s in enumerate(product.processing_steps) if getattr(s, 'process_type', None) == ProcessType.PACKAGING), len(product.processing_steps))
            bulk_steps = product.processing_steps[:first_pack_idx]

            if not bulk_steps: return []

            # 1. Create the single, large root task for the first step.
            root_task = self._create_single_task(
                job_id=batch_plan["batch_id"], sku_id=product_category, batch_idx=0,
                step=bulk_steps[0], step_idx=0, task_type=TaskType.BULK_PRODUCTION,
                volume=batch_plan["volume"], priority=Priority.MEDIUM.value,sub_batch_id=batch_plan["batch_id"]
            )
            leaf_tasks = [root_task]

            # 2. Loop through the rest of the bulk steps to build out the graph.
            for i, step in enumerate(bulk_steps[1:]):
                step_idx = i + 1
                next_leaf_tasks = []
                
                # Determine the capacity constraint for this upcoming step.
                downstream_capacity = self._get_effective_downstream_capacity(i, product)

                # 3. For each current leaf task, decide if it needs to be split.
                for parent_task in leaf_tasks:
                    volume_to_process = parent_task.volume_liters
                    
                    if volume_to_process <= downstream_capacity:
                        # No split needed, create one child task.
                        new_task = self._create_single_task(
                            job_id=parent_task.job_id, sku_id=parent_task.sku_id, batch_idx=len(next_leaf_tasks),
                            step=step, step_idx=step_idx, task_type=TaskType.BULK_PRODUCTION,
                            volume=volume_to_process, priority=parent_task.priority, sub_batch_id= parent_task.job_id
                        )
                        new_task.prerequisites.append(parent_task)
                        parent_task.next_task = new_task # Maintain single link for simple cases
                        new_task.previous_task = parent_task
                        next_leaf_tasks.append(new_task)
                    else:
                        # Split needed, using the "Fill to Capacity" method.
                        child_tasks = []
                        remaining_volume = volume_to_process
                        
                        # Create the first child task filled to capacity.
                        vol_for_first_split = min(remaining_volume, downstream_capacity)
                        first_split_task = self._create_single_task(
                            job_id=parent_task.job_id, sku_id=parent_task.sku_id, batch_idx=len(next_leaf_tasks),
                            step=step, step_idx=step_idx, task_type=TaskType.BULK_PRODUCTION,
                            volume=vol_for_first_split, priority=parent_task.priority, sub_batch_id=parent_task.job_id
                        )
                        child_tasks.append(first_split_task)
                        remaining_volume -= vol_for_first_split

                        # Create the second child task with the remainder.
                        if remaining_volume > 0:
                            second_split_task = self._create_single_task(
                                job_id=parent_task.job_id, sku_id=parent_task.sku_id, batch_idx=len(next_leaf_tasks) + 1,
                                step=step, step_idx=step_idx, task_type=TaskType.BULK_PRODUCTION,
                                volume=remaining_volume, priority=parent_task.priority, sub_batch_id=parent_task.job_id
                            )
                            child_tasks.append(second_split_task)
                        
                        # --- NEW LINKING LOGIC ---
                        # Link the parent to the first child to ensure the chain is connected.
                        parent_task.next_task = child_tasks[0]
                        
                        for child in child_tasks:
                            # This is the key dependency link for the scheduler.
                            child.prerequisites.append(parent_task)
                            # This helps simple traversal logic.
                            child.previous_task = parent_task
                        
                        next_leaf_tasks.extend(child_tasks)

                leaf_tasks = next_leaf_tasks

            # 4. Return the final list of leaf tasks at the end of the bulk chain.
            return leaf_tasks

    def _create_finishing_chain(self, sub_order: SubOrder, final_bulk_task: HeuristicTask):
        """
        Creates the finishing/packing tasks for a single sub-order.
        If a PACKAGING step is followed by a POST_PACKAGING step, it will
        automatically create a chain of 15-minute sub-tasks to enable a
        continuous pipeline flow for all subsequent post-packaging steps.
        """
        product = self.products.get(self.skus[sub_order.sku_id].product_category)
        if not product: return

        all_steps = product.processing_steps
        first_pack_idx = next((i for i, s in enumerate(all_steps) if getattr(s, 'process_type', None) == ProcessType.PACKAGING), -1)
        if first_pack_idx == -1: return

        finishing_steps = all_steps[first_pack_idx:]
        
        # Check if the pipeline logic should be triggered
        is_pipeline_flow = (
            len(finishing_steps) > 1 and
            finishing_steps[0].process_type == ProcessType.PACKAGING and
            finishing_steps[1].process_type == ProcessType.POST_PACKAGING
        )

        if is_pipeline_flow:
            # --- PIPELINE LOGIC: Create 15-minute sub-batches ---
            self.log_entries.append(f"  [PIPELINE] Detected pipeline flow for {sub_order.sub_order_id}. Creating 15-min sub-tasks.")
            
            pack_step = finishing_steps[0]
            
            # Create a temporary task to calculate total duration accurately
            temp_pack_task = HeuristicTask(
                task_id="temp", job_id=sub_order.sub_order_id, sku_id=sub_order.sku_id,
                step=pack_step, task_type=TaskType.FINISHING, volume_liters=sub_order.volume,
                compatible_resources=self._get_compatible_resources_for_step(sub_order.sku_id, pack_step)
            )
            total_duration_minutes = self._calculate_dynamic_duration_tokens(temp_pack_task) * self.TIME_BLOCK_MINUTES
            
            if total_duration_minutes == 0: return
            
            num_sub_batches = math.ceil(total_duration_minutes / self.TIME_BLOCK_MINUTES)
            if num_sub_batches == 0: num_sub_batches = 1
            
            volume_per_sub_batch = sub_order.volume / num_sub_batches
            
            last_pack_sub_task = final_bulk_task
            for i in range(num_sub_batches):
                # 1. Create the 15-minute packaging sub-task
                pack_sub_task_id = f"{sub_order.sub_order_id}-pack-p{i+1}"
                pack_sub_task = self._create_single_task(
                    job_id=sub_order.sub_order_id, sku_id=sub_order.sku_id, batch_idx=i,
                    step=pack_step, step_idx=first_pack_idx, task_type=TaskType.FINISHING,
                    volume=volume_per_sub_batch, priority=sub_order.priority, sub_batch_id=sub_order.master_batch_id,
                    custom_id=pack_sub_task_id
                )
                pack_sub_task.base_duration_tokens = 1 # Force duration to 15 minutes
                
                # Link to the previous task in the chain (either the final bulk task or the previous packing sub-task)
                pack_sub_task.prerequisites.append(last_pack_sub_task)
                pack_sub_task.previous_task = last_pack_sub_task
                
                # The next packing sub-task will depend on this one
                last_pack_sub_task = pack_sub_task 

                # 2. CORRECTED LOGIC: Create the FULL post-packaging chain for THIS sub-task
                last_task_in_sub_chain = pack_sub_task
                post_pack_steps = finishing_steps[1:]
                for j, post_step in enumerate(post_pack_steps):
                    post_pack_sub_task = self._create_single_task(
                        job_id=sub_order.sub_order_id, sku_id=sub_order.sku_id, batch_idx=i,
                        step=post_step, step_idx=first_pack_idx + 1 + j, task_type=TaskType.FINISHING,
                        volume=volume_per_sub_batch, priority=sub_order.priority, sub_batch_id=sub_order.master_batch_id,
                        custom_id=f"{sub_order.sub_order_id}-{post_step.step_id}-p{i+1}"
                    )
                    post_pack_sub_task.prerequisites.append(last_task_in_sub_chain)
                    post_pack_sub_task.previous_task = last_task_in_sub_chain
                    last_task_in_sub_chain = post_pack_sub_task

        else:
            # --- STANDARD LOGIC: Create one task per step (Unchanged) ---
            last_task_in_chain = final_bulk_task
            for i, step in enumerate(finishing_steps):
                task = self._create_single_task(
                    job_id=sub_order.sub_order_id, sku_id=sub_order.sku_id, batch_idx=0,
                    step=step, step_idx=first_pack_idx + i, task_type=TaskType.FINISHING,
                    volume=sub_order.volume, priority=sub_order.priority, sub_batch_id=sub_order.master_batch_id
                )
                task.prerequisites.append(last_task_in_chain)
                task.previous_task = last_task_in_chain
                last_task_in_chain = task

    def _group_orders_by_line_type(self, orders: List[UserIndent]) -> Dict[str, List[UserIndent]]:
        """
        Dynamically groups orders by creating a 'signature' for each line based on the
        product categories it can produce. All lines with the same signature are grouped.
        """
        # 1. Create a signature for each line based on the product categories it serves.
        line_signatures = defaultdict(list)
        for line_id, line in self.lines.items():
            # Find all unique product categories this line can make
            categories = {self.skus[sku_id].product_category for sku_id in line.compatible_skus_max_production}
            # Create a sorted, string-based signature (e.g., "CURD,LASSI")
            signature = ",".join(sorted(list(categories)))
            if signature:
                line_signatures[signature].append(line_id)

        # 2. For each order, find which group signature it belongs to.
        grouped_orders = defaultdict(list)
        for order in orders:
            sku = self.skus.get(order.sku_id)
            if not sku: continue
            
            # Find the signature that contains this order's line requirements
            for signature, line_ids in line_signatures.items():
                product = self.products.get(sku.product_category)
                if not product: continue
                pack_step = next((s for s in product.processing_steps if s.process_type == ProcessType.PACKAGING), None)
                if not pack_step: continue

                # If this SKU can be made on any line in this signature group, assign it.
                if any(line_id in pack_step.requirements[0].compatible_ids for line_id in line_ids):
                    grouped_orders[signature].append(order)
                    break # Move to the next order once assigned

        return grouped_orders

    def _identify_and_prepare_anchors(self):
        """Finds all anchor tasks, marks them, and calculates their lead times."""
        anchor_tasks = [
            t for t in self.master_task_list
            if getattr(t.step, 'process_type', None) == ProcessType.PACKAGING
        ]
        self.log_entries.append(f'[INFO] Found {len(anchor_tasks)} anchor tasks')
        
        for task in anchor_tasks:
            task.is_anchor_task = True
            task.total_prereq_duration_tokens = self._estimate_prereq_duration(task, set())
            self.log_entries.append(f'Anchor task {task.task_id} has prereq duration: {task.total_prereq_duration_tokens}')
        
        self.sorted_anchors = sorted(anchor_tasks, key=lambda t: t.priority, reverse=True)

    def _aggregate_demand_by_category(self) -> Dict[str, Dict]:
        """Groups valid indents by product category."""
        aggregated_demand = defaultdict(lambda: {'total_qty': 0, 'indents': []})
        current_time = datetime.now()
        
        for indent in self.indents.values():
            if indent.due_date < current_time:
                self.log_entries.append(f"Skipping past-due order: {indent.order_no}")
                continue
            
            sku = self.skus.get(indent.sku_id)
            if not sku:
                self.log_entries.append(f"Warning: SKU '{indent.sku_id}' not found for order '{indent.order_no}'.")
                continue
                
            product_category = sku.product_category
            aggregated_demand[product_category]['total_qty'] += indent.qty_required_liters
            aggregated_demand[product_category]['indents'].append(indent)
            
        return aggregated_demand
    
    def _find_bulk_prerequisites(self, product_category: str, first_pack_idx: int) -> List[HeuristicTask]:
        """Finds the bulk production tasks that finishing tasks depend on."""
        if first_pack_idx <= 0:
            return []
        
        bulk_product_def = self.products.get(product_category)
        if not bulk_product_def:
            return []
        
        # Get the final bulk step (the one just before packaging)
        final_bulk_step = bulk_product_def.processing_steps[first_pack_idx - 1]
        
        # Find all tasks for this final bulk step
        source_bulk_tasks = [
            t for t in self.master_task_list
            if (t.job_id == product_category and 
                t.step.step_id == final_bulk_step.step_id and
                t.task_type == TaskType.BULK_PRODUCTION)
        ]
        
        self.log_entries.append(f"Found {len(source_bulk_tasks)} bulk prerequisites for {product_category}")
        return source_bulk_tasks

    def _create_single_task(self, job_id, sku_id, batch_idx, step, step_idx, task_type, volume, priority, sub_batch_id: str, custom_id: Optional[str] = None) -> HeuristicTask:
        """
        Helper to create one HeuristicTask instance and add it to the master lists.
        Now supports a custom_id for pipeline sub-tasks.
        """
        # Use the custom ID if provided, otherwise generate the standard ID
        task_id = custom_id if custom_id else f"{job_id}-{sku_id}-b{batch_idx}-{step.step_id}"
        
        compatible_resources = defaultdict(list)
        if step:
            for req in step.requirements:
                # This logic for filtering compatible lines remains important
                if task_type == TaskType.FINISHING and req.resource_type == ResourceType.LINE:
                    filtered_line_ids = []
                    for line_id in req.compatible_ids:
                        line = self.lines.get(line_id)
                        if line and sku_id in getattr(line, 'compatible_skus_max_production', {}):
                            filtered_line_ids.append(line_id)
                    compatible_resources[req.resource_type].extend(filtered_line_ids)
                else:
                    compatible_resources[req.resource_type].extend(req.compatible_ids)
        
        task = HeuristicTask(
            task_id=task_id,
            job_id=job_id,
            sku_id=sku_id,
            step=step,
            step_idx=step_idx,
            batch_idx=batch_idx,
            task_type=task_type,
            compatible_resources=dict(compatible_resources),
            volume_liters=volume,
            priority=priority,
            sub_batch_id=sub_batch_id
        )

        task.base_duration_tokens = self._calculate_dynamic_duration_tokens(task)

        self.log_entries.append(f"Appending task: {task.task_id} to master task list")
        self.master_task_list.append(task)
        self.task_lookup[task.task_id] = task
        return task

    def _calculate_dynamic_duration_tokens(self, task: HeuristicTask) -> int:
        """
        Calculates the task duration in tokens, dynamically and safely.
        It now correctly uses the task's specific compatible resources.
        """
        # The step can be None for auto-generated CIP tasks
        if not task.step:
            return task.base_duration_tokens

        step = task.step
        base_minutes = getattr(step, 'duration_minutes', 60)
        
        longest_duration_minutes = 0

        # --- START: CORRECTED LOGIC ---
        # Iterate over the task's pre-filtered compatible resources, not the step's raw list.
        for res_type, res_ids in task.compatible_resources.items():
            if res_type not in [ResourceType.LINE, ResourceType.EQUIPMENT]:
                continue
            
            for res_id in res_ids:
                processing_speed = 0
                if res_type == ResourceType.LINE:
                    line = self.lines.get(res_id)
                    if line:
                        speeds = getattr(line, 'compatible_skus_max_production', {})
                        processing_speed = speeds.get(task.sku_id, 0)
                
                elif res_type == ResourceType.EQUIPMENT:
                    equipment = self.equipments.get(res_id)
                    if equipment:
                        processing_speed = getattr(equipment, 'processing_speed', 0)

                if processing_speed > 0:
                    work_time = (task.volume_liters / processing_speed)
                    longest_duration_minutes = max(longest_duration_minutes, work_time)
                # No warning needed here anymore, as we are only checking truly compatible resources.
        # --- END: CORRECTED LOGIC ---

        final_duration_minutes = longest_duration_minutes if longest_duration_minutes > 0 else base_minutes
        final_duration_minutes += getattr(step, 'setup_time', 0) + getattr(step, 'cool_down', 0)
        
        if final_duration_minutes <= 0:
            self.log_entries.append(f"                  [WARNING] Calculated duration for task {task.task_id} is zero or negative. Defaulting to 1 token.")
            return 1

        # Update the task's base duration with the more accurate calculation
        task.base_duration_tokens = self._round_up_duration_to_tokens(final_duration_minutes)
        return task.base_duration_tokens

    def _get_backfill_candidates(self) -> List[HeuristicTask]:
        """Find tasks that are ready to be backfilled by checking the prerequisites of booked tasks."""
        candidates = []
        
        # Get all tasks that are prerequisites of already booked tasks
        booked_tasks = [t for t in self.master_task_list if t.status == ScheduleStatus.BOOKED]
        
        for booked_task in booked_tasks:
            for prereq in booked_task.prerequisites:
                # If the prerequisite is pending, it's a candidate for backfilling
                if prereq.status == ScheduleStatus.PENDING:
                    # To avoid duplicates if a task is a prereq for multiple booked tasks
                    if prereq not in candidates:
                        candidates.append(prereq)
                        self.log_entries.append(f"[BACKFILL-CANDIDATE] Found {prereq.task_id} as prerequisite for {booked_task.task_id}")

        return candidates
    
    def _estimate_prereq_duration(self, task: HeuristicTask, visited: set) -> int:
        """
        Recursively estimates the total token duration for the LONGEST prerequisite chain (critical path).
        """
        if task.task_id in visited:
            return 0 # Avoid infinite loops
        visited.add(task.task_id)

        if not task.prerequisites:
            return 0

        # Instead of summing all prerequisites, find the duration of the longest one.
        max_prereq_path_duration = 0
        for prereq in task.prerequisites:
            # The duration of this path is the prereq's own duration plus the critical path leading to it.
            current_path_duration = prereq.base_duration_tokens + self._estimate_prereq_duration(prereq, visited)
            if current_path_duration > max_prereq_path_duration:
                max_prereq_path_duration = current_path_duration
        
        return max_prereq_path_duration
    
    def _get_ready_tasks(self) -> List[HeuristicTask]:
        """Finds all tasks whose prerequisites are met."""
        ready = []
        for task in self.master_task_list:
            if task.status == ScheduleStatus.PENDING:
                # A task is ready if all its prerequisites are booked.
                if all(p.status == ScheduleStatus.BOOKED for p in task.prerequisites):
                    ready.append(task)
        return ready
    
    def _find_earliest_forward_slot(self, task: HeuristicTask) -> Optional[int]:
        """Searches FORWARDS from the earliest prerequisite finish time to find the first available slot."""
        earliest_start = self._get_prerequisite_finish_token(task)
        search_token = earliest_start
        duration_tokens = self._calculate_dynamic_duration_tokens(task)

        while search_token < self.schedule_end_token:
            placements, _ = self._get_concurrent_placements_at_token(task, search_token, duration_tokens)
            if placements:
                return search_token # Found the earliest possible start token
            search_token += 1
        return None

    def _score_and_select_best_task(self, tasks_to_score: List[HeuristicTask]) -> HeuristicTask:
        """
        Selects the best task to schedule next using SKU-aware prioritization
        to group similar tasks together and minimize changeovers.
        """
        # Sort tasks by priority, then by their true master batch, then by SKU.
        tasks_to_score.sort(key=lambda t: (
            -t.priority, 
            t.sub_batch_id, # Use the reliable sub_batch_id for grouping
            t.sku_id,       # The new key for SKU-aware optimization
            t.step_idx
        ))
        
        best_task = tasks_to_score[0]
        self.log_entries.append(f"[SELECT] Selected task {best_task.task_id} (Batch: {best_task.sub_batch_id}, SKU: {best_task.sku_id})")
        
        return best_task
   
    def _get_last_task_on_resource(self, resource_id: str, before_token: int) -> Tuple[int, Optional[str]]:
        """Finds the end token and ID of the last task booked on a resource before a given token."""
        last_task_end_token = 0
        last_task_id = None
        # We only need the end time and ID, so we use _ for the start time
        for _, booked_end, t_id in self.resource_timelines.get(resource_id, []):
            if booked_end <= before_token:
                if booked_end > last_task_end_token:
                    last_task_end_token = booked_end
                    last_task_id = t_id
        return last_task_end_token, last_task_id

    def _get_task_resource_reqs(self, task: HeuristicTask) -> List[ResourceRequirement]:
        """Helper to safely get the list of resource requirements from a task's step."""
        return getattr(task.step, 'requirements', [])

    def _schedule_drained_tank_cip(self):
        """
        Finds tanks that have been drained and schedules a CIP using the ResourceManager.
        """
        self.log_entries.append("  -> Checking for drained tanks that need CIP...")
        
        for tank_id in self.tanks.keys():
            last_drain_time = None
            
            # Find the end time of the very last task on this tank
            if self.resource_manager.timeline[tank_id]:
                # The timeline is sorted, so the last task is at the end
                last_start, last_end, last_task_id, _ = self.resource_manager.timeline[tank_id][-1]
                last_drain_time = last_end

            if not last_drain_time:
                continue

            # Check the final state of the tank in the ResourceManager
            final_state = self.resource_manager.get_state(tank_id)
            if final_state and final_state.status == ResourceStatus.DIRTY:
                self.log_entries.append(f"  [TANK-CIP] Tank {tank_id} was left DIRTY at {last_drain_time}, scheduling CIP.")
                self._schedule_cip_for_resource(tank_id, last_drain_time)

    def _schedule_final_equipment_cip(self):
        """
        Schedules a final CIP for any non-tank equipment left in a dirty state.
        """
        self.log_entries.append("  -> Checking for other equipment needing final CIP...")
        
        other_equipment_ids = list(self.lines.keys()) + list(self.equipments.keys())

        for resource_id in other_equipment_ids:
            final_state = self.resource_manager.get_state(resource_id)
            if final_state and final_state.status == ResourceStatus.DIRTY:
                last_task_end_time = final_state.dirty_since
                if last_task_end_time:
                    self.log_entries.append(f"  [FINAL-CIP] Equipment {resource_id} needs a final CIP after {last_task_end_time}.")
                    self._schedule_cip_for_resource(resource_id, last_task_end_time)

    def _schedule_cip_for_resource(self, resource_id: str, start_after_dt: datetime):
        """
        Helper function to find a slot and book a CIP task for a given resource using datetimes.
        """
        resource_object = self.resource_manager.resource_objects.get(resource_id)
        if not resource_object or not hasattr(resource_object, 'CIP_duration_minutes'):
            self.log_entries.append(f"  [CIP-HELPER-WARN] No CIP duration for {resource_id}.")
            return

        cip_duration = timedelta(minutes=getattr(resource_object, 'CIP_duration_minutes', 90))

        search_dt = start_after_dt
        # Define the end of the search window as a datetime object
        extended_schedule_end_dt = self.schedule_end_dt + timedelta(days=2)

        while search_dt < extended_schedule_end_dt:
            cip_end_dt = search_dt + cip_duration
            
            if self.resource_manager.is_available(resource_id, search_dt, cip_end_dt):
                # Found a slot, book it using the ResourceManager
                cip_task_id = f"AUTO-CIP-on-{resource_id}"
                # The step attribute can be None for these auto-generated tasks
                cip_task = HeuristicTask(task_id=cip_task_id, job_id="CLEANUP", sku_id="CIP", step=None, task_type=TaskType.CIP)
                
                # Use the commit method to ensure state is updated correctly
                self.resource_manager.commit_task_to_timeline(cip_task, resource_id, search_dt, cip_end_dt)
                
                # Update the main task list for final reporting
                cip_task.status = ScheduleStatus.BOOKED
                cip_task.assigned_resource_id = resource_id
                cip_task.start_time = search_dt
                cip_task.end_time = cip_end_dt
                self.master_task_list.append(cip_task)
                self.task_lookup[cip_task.task_id] = cip_task
                return

            search_dt += timedelta(minutes=self.TIME_BLOCK_MINUTES)

        self.log_entries.append(f"  [CIP-HELPER-FAIL] Could not find slot for CIP on {resource_id}.")

    def _get_prerequisite_finish_time(self, task: HeuristicTask) -> datetime:
        """Calculates the datetime when all prerequisites for a task are complete."""
        # The default earliest time is the start of the entire schedule
        finish_time = self.schedule_start_dt
        if task.prerequisites:
            # Get the end_time (which is already a datetime) of all booked prerequisites
            end_times = [p.end_time for p in task.prerequisites if p.status == ScheduleStatus.BOOKED and p.end_time]
            if end_times:
                # Return the latest of these datetime objects
                finish_time = max(end_times)
        return finish_time
    
    def _is_resource_of_type(self, resource_id: str, resource_type: ResourceType) -> bool:
        """Verifies that a resource ID belongs to the correct resource category."""
        if resource_type == ResourceType.LINE and resource_id in self.lines: return True
        if resource_type == ResourceType.TANK and resource_id in self.tanks: return True
        if resource_type == ResourceType.EQUIPMENT and resource_id in self.equipments: return True
        # Add other resource types as needed (e.g., ROOM)
        return False

    def _create_scheduling_result(self) -> SchedulingResult:
        """
        Translates the internal scheduler tasks into the SchedulingResult object
        expected by the Gantt chart UI.
        """
        scheduled_tasks = []
        cip_schedules = []

        for task in self.master_task_list:
            if task.status != ScheduleStatus.BOOKED:
                continue

            # This part seems to have a bug with base_order_no not being defined.
            # We will default the priority for now.
            priority_val = Priority.MEDIUM

            if task.task_type == TaskType.CIP:
                if not task.assigned_resource_id: continue
                primary_resource = task.assigned_resource_id.split(',')[0]
                
                following_task_id = "N/A"
                if '-for-' in task.task_id:
                    following_task_id = task.task_id.split('-for-')[1]

                # Find the preceding task by comparing datetimes directly
                preceding_task_id = "N/A"
                for booked_start, booked_end, t_id, _ in self.resource_manager.timeline.get(primary_resource, []):
                    if booked_end == task.start_time:
                        preceding_task_id = t_id
                        break
                
                cip_schedules.append(CIPSchedule(
                    CIP_id=task.task_id,
                    resource_id=primary_resource,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    duration_minutes=(task.end_time - task.start_time).total_seconds() / 60,
                    CIP_type="Major",
                    preceding_task_id=preceding_task_id,
                    following_task_id=following_task_id
                ))
            
            elif task.task_type in [TaskType.BULK_PRODUCTION, TaskType.FINISHING]:
                if not task.assigned_resource_id or not task.step: continue
                base_order_no = task.job_id
                if task.task_type == TaskType.FINISHING and '-cab' in task.job_id:
                    base_order_no = task.job_id.split('-cab')[0]

                primary_resource = task.assigned_resource_id.strip("[]'\"").split(',')[0]
                
                scheduled_tasks.append(TaskSchedule(
                    task_id=task.task_id,
                    order_no=base_order_no,
                    sku_id=task.sku_id,
                    step_id=task.step.step_id,
                    batch_index=task.batch_idx,
                    resource_id=primary_resource,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    volume=task.volume_liters,
                    priority=priority_val
                ))

        dummy_metrics = type('Metrics', (), {
            'schedule_efficiency': 0.95,
            'total_production_volume': sum(t.volume for t in scheduled_tasks),
            'otif_rate': 1.0, 'solve_time': 0.1
        })()
            
        return SchedulingResult(
            scheduled_tasks=scheduled_tasks, CIP_schedules=cip_schedules,
            production_summary={}, metrics=dummy_metrics, solve_time=0.1,
            status=1, objective_value=100.0, resource_utilization={}
        )
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """
        Calculates the utilization percentage for each resource based on its timeline.
        """
        utilization_map = {}
        total_schedule_duration_minutes = (self.schedule_end_dt - self.schedule_start_dt).total_seconds() / 60
        
        if total_schedule_duration_minutes == 0:
            return {}

        for resource_id, timeline in self.resource_manager.timeline.items():
            busy_minutes = 0
            for start, end, task_id, _ in timeline:
                # We only count time for actual production/CIP tasks, not placeholder "LOCKED" tasks
                task = self.task_lookup.get(task_id)
                if task and task.task_type != TaskType.LOCKED:
                    busy_minutes += (end - start).total_seconds() / 60
            
            utilization = (busy_minutes / total_schedule_duration_minutes) * 100
            utilization_map[resource_id] = utilization
            
        return utilization_map

    def write_schedule_log_file(self, file_path: str = "heuristic_schedule_v2_log.txt"):
        """
        Writes a comprehensive log of the entire scheduling run to a file,
        now including a bottleneck analysis section.
        """
        self.log_entries.append(f"Writing full schedule log to {file_path}...")
        
        # --- START: New Bottleneck Analysis Logic ---
        utilization = self._calculate_resource_utilization()
        bottleneck_resource = ""
        max_utilization = -1.0

        if utilization:
            # Find the resource with the highest utilization (ignoring rooms for now as they are parallel)
            non_room_utilization = {res: util for res, util in utilization.items() if not isinstance(self.resource_manager.resource_objects.get(res), Room)}
            if non_room_utilization:
                bottleneck_resource = max(non_room_utilization, key=non_room_utilization.get)
                max_utilization = non_room_utilization[bottleneck_resource]
        # --- END: New Bottleneck Analysis Logic ---

        with open(file_path, "w") as f:
            f.write('-'*20 + f' Log made at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ' + '-'*20 + '\n')
            
            # --- START: New Analysis Summary Section ---
            if bottleneck_resource:
                f.write("\n" + "="*80 + "\n")
                f.write("BOTTLENECK ANALYSIS\n")
                f.write("="*80 + "\n")
                f.write(f"Primary Bottleneck Identified: {bottleneck_resource} (at {max_utilization:.2f}% utilization)\n\n")
                f.write("Key Resource Utilization:\n")
                
                sorted_util = sorted(utilization.items(), key=lambda item: item[1], reverse=True)
                for res, util in sorted_util:
                    if util > 1: # Only show resources with meaningful utilization
                        f.write(f"  - {res:<20}: {util:.2f}%\n")
                f.write("="*80 + "\n\n")
            # --- END: New Analysis Summary Section ---

            f.write("="*80 + "\n")
            f.write("TASK RELATIONSHIP DEBUG\n")
            f.write("="*80)
            # ... (rest of the task relationship debug log is unchanged) ...
            jobs = defaultdict(list)
            for task in self.master_task_list:
                jobs[task.job_id].append(task)
            
            for job_id, tasks in jobs.items():
                f.write(f"\n--- JOB: {job_id} ---\n")
                # ... (the rest of this loop is unchanged)
                for task in tasks:
                    prereq_ids = [p.task_id for p in task.prerequisites]
                    f.write(f"  {task.task_id} (Prio: {task.priority}, Status: {task.status.name}):\n")
                    f.write(f"    - Prereqs: {prereq_ids}\n")
                    f.write(f"    - Assigned: {task.assigned_resource_id}\n")
                    if task.start_time:
                        f.write(f"    - Time: {task.start_time.strftime('%H:%M')} -> {task.end_time.strftime('%H:%M')}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("DECISION LOG\n")
            f.write("="*80 + "\n")
            for entry in self.log_entries:
                f.write(entry + "\n")

            f.write("\n\n--- Final Schedule by Resource ---\n")
            header = f"{'Resource':<20} | {'Start Time':<16} | {'End Time':<16} | {'Dur(m)':>6} | {'Volume(L)':>9} | {'Task ID'}\n"
            f.write(header)
            f.write('-' * (len(header) + 5) + '\n')

            tasks_by_id = {t.task_id: t for t in self.master_task_list}

            for resource_id, timeline in sorted(self.resource_manager.timeline.items()):
                f.write(f"--- {resource_id} ---\n")
                last_end_dt = self.schedule_start_dt
                for start_dt, end_dt, task_id, capacity_consumed in timeline:
                    idle_minutes = (start_dt - last_end_dt).total_seconds() / 60
                    if idle_minutes > 1:
                        f.write(f"{'':20} | {'...':<16} | {'...':<16} | {idle_minutes:>6.0f} | {'':>9} | {'(IDLE)'}\n")
                    
                    task = tasks_by_id.get(task_id)
                    duration_minutes = (end_dt - start_dt).total_seconds() / 60
                    volume_str = ""
                    if task and task.volume_liters > 0:
                        volume_str = f"{task.volume_liters:.0f}"

                    f.write(
                        f"{resource_id:<20} | {start_dt.strftime('%m-%d %H:%M'):<16} | {end_dt.strftime('%m-%d %H:%M'):<16} | "
                        f"{duration_minutes:>6.0f} | {volume_str:>9} | {task_id}\n"
                    )
                    last_end_dt = end_dt
                f.write("\n")

    def _report_unscheduled_tasks(self):
        """Adds a summary of any tasks that couldn't be scheduled to the log."""
        print("\n--- Unscheduled Task Report ---")
        found_any = False
        for task in self.master_task_list:
            if task.status != ScheduleStatus.BOOKED:
                found_any = True
                reason = "Prerequisites were not met or no valid slot was found."
                self.log_entries.append(f"Task: {task.task_id:<40} | Status: {task.status.name:<10} | Reason: {reason}")
        if not found_any:
            self.log_entries.append("All tasks were successfully scheduled.")

    def generate_room_capacity_plot(self, output_dir: str = "."):
        """
        Generates an interactive HTML plot showing the EUI capacity usage
        over time for each Room resource.
        """
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import os

        self.log_entries.append("\n--- Generating Room Capacity Plots ---")
        
        # Identify all room resources
        room_ids = [res_id for res_id, res_obj in self.resource_manager.resource_objects.items() if isinstance(res_obj, Room)]
        
        if not room_ids:
            self.log_entries.append("No Room resources found to plot.")
            print("No Room resources found to plot.")
            return

        # Create one plot per room
        for room_id in room_ids:
            timeline = self.resource_manager.timeline.get(room_id, [])
            room_obj = self.resource_manager.resource_objects.get(room_id)
            
            if not timeline:
                self.log_entries.append(f"Room {room_id} has no scheduled tasks. Skipping plot.")
                continue

            # Create a list of events: (time, change_in_capacity)
            events = []
            for start_time, end_time, task_id, capacity_consumed in timeline:
                events.append((start_time, capacity_consumed))  # Capacity increases at the start
                events.append((end_time, -capacity_consumed)) # Capacity decreases at the end

            # Sort events chronologically
            events.sort()

            # Build the time-series data for the plot
            plot_data = []
            current_capacity = 0
            
            # Add a point at the very beginning of the schedule to anchor the chart
            if events:
                plot_data.append({'time': events[0][0] - timedelta(minutes=1), 'capacity': 0})

            for time_event, capacity_change in events:
                # Add a point just before the change to create a step effect
                if plot_data:
                    plot_data.append({'time': time_event - timedelta(seconds=1), 'capacity': current_capacity})
                
                current_capacity += capacity_change
                plot_data.append({'time': time_event, 'capacity': current_capacity})

            if not plot_data:
                continue
                
            df = pd.DataFrame(plot_data)

            # Create the plot
            fig = go.Figure()

            # Add the capacity usage line
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['capacity'],
                mode='lines',
                line_shape='hv', # Creates the step-chart effect
                name='Used Capacity (EUI)',
                fill='tozeroy' # Fills the area under the curve
            ))

            # Add the red line for total capacity
            fig.add_hline(
                y=room_obj.capacity_units,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Max Capacity: {room_obj.capacity_units} EUI",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title=f"Capacity Utilization Timeline for {room_id}",
                xaxis_title="Time",
                yaxis_title="Equivalent Units of Inventory (EUI) Consumed",
                template="plotly_white"
            )
            
            # Save the plot to an HTML file
            output_path = os.path.join(output_dir, f"capacity_timeline_{room_id}.html")
            fig.write_html(output_path)
            self.log_entries.append(f"Successfully generated plot for {room_id} at: {output_path}")
            print(f"Plot for {room_id} saved to {output_path}")

    
    
    def _create_scheduling_result_for_export(self) -> SchedulingResult:
        """
        Translates internal HeuristicTask objects into a list of clean TaskSchedule
        objects for the UI, ensuring all fields are correctly mapped.
        """
        scheduled_tasks: List[TaskSchedule] = []
        cip_schedules: List[CIPSchedule] = []
        
        # Filter for only tasks that were successfully booked
        booked_tasks = [task for task in self.master_task_list if task.status == ScheduleStatus.BOOKED]

        for task in booked_tasks:
            # Skip tasks that don't have the necessary info (e.g., auto-generated LOCK tasks)
            if not task.step:
                continue

            # Map production and finishing tasks to the TaskSchedule object
            if task.task_type in [TaskType.BULK_PRODUCTION, TaskType.FINISHING]:
                
                # --- Field Mapping Logic ---
                
                # For order_no, clean up the job_id if it's a finishing task
                order_no = task.job_id
                if task.task_type == TaskType.FINISHING and '-cab' in order_no:
                    order_no = task.job_id.split('-cab')[0]

                # For resource_id, ensure we get a single, clean ID string
                resource_id = str(task.assigned_resource_id).strip("[]'\"").split(',')[0]
                
                # --- Create the final TaskSchedule object ---
                task_to_add = TaskSchedule(
                    task_id=task.task_id,
                    order_no=order_no,
                    sku_id=task.sku_id,
                    batch_index=task.batch_idx,
                    step_id=task.step.step_id,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    resource_id=resource_id,
                    volume=int(task.volume_liters),          # Convert float to int
                    priority=Priority(task.priority),        # Convert int to Enum
                    process_type=task.step.process_type,
                    setup_time=getattr(task.step, 'setup_time', 0),
                    CIP_required=False # See discussion below
                )
                scheduled_tasks.append(task_to_add)

        # --- Result Creation ---
        dummy_metrics = type('Metrics', (), {'schedule_efficiency': 0.95, 'total_production_volume': sum(t.volume for t in scheduled_tasks), 'otif_rate': 1.0, 'solve_time': 0.1})()
            
        result = SchedulingResult(
            scheduled_tasks=scheduled_tasks,
            CIP_schedules=cip_schedules, # Note: CIP tasks are not yet added here
            production_summary={},
            metrics=dummy_metrics,
            solve_time=0.1,
            status=1,
            objective_value=100.0,
            resource_utilization={}
        )
        
        # Attach the data needed for the interactive to-do list
        result.task_lookup = {t.task_id: t for t in booked_tasks}
        result.task_graph = {t.task_id: [p.task_id for p in t.prerequisites] for t in booked_tasks}

        return result

if __name__ == "__main__":
    print("Running Heuristic Scheduler in standalone mode...")

    # Load data directly from config files
    loader = DataLoader()
    loader.clear_all_data()
    loader.load_sample_data()
    
    scheduler = HeuristicScheduler(
        indents=config.USER_INDENTS,
        skus=config.SKUS,
        products=config.PRODUCTS,
        lines=config.LINES,
        tanks=config.TANKS,
        equipments=config.EQUIPMENTS,
        shifts=config.SHIFTS
    )
    print(config.LINES)
    
    # Run the scheduler
    scheduler.run_heuristic_scheduler()
    
    # Write the detailed text log file
    scheduler.write_schedule_log_file()
    scheduler.generate_room_capacity_plot()