import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, time
from enum import Enum, auto
from collections import defaultdict
import math
import plotly.io as pio
import json
import datetime


# Assume 'config.py' and 'utils/data_models.py' exist in the same directory
# or are accessible in the Python path.
import config
from utils.data_models import *
from utils.data_loader import *
import plotly.express as px
import pandas as pd

# --- Enums for Status and Task Types ---

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

class TimeManager:
    """Manages working shifts and finds valid uninterruptible time slots."""
    def __init__(self, shifts: Dict, time_block_minutes: int):
        self.TIME_BLOCK_MINUTES = time_block_minutes
        self.shifts_by_day = defaultdict(list)
        for s in shifts.values():
            if s.is_active:
                # Assume an active shift runs every day of the week
                for day_of_week in range(7):
                    shift_start = s.start_time
                    shift_end = s.end_time
                    self.shifts_by_day[day_of_week].append((shift_start, shift_end))

    def _is_in_shift(self, dt_to_check: datetime) -> bool:
        """Checks if a single datetime falls within any active shift for that day."""
        day_of_week = dt_to_check.weekday()
        time_of_day = dt_to_check.time()

        for shift_start, shift_end in self.shifts_by_day[day_of_week]:
            if shift_end <= shift_start: # Overnight shift
                if time_of_day >= shift_start or time_of_day < shift_end:
                    return True
            else: # Normal shift
                if shift_start <= time_of_day < shift_end:
                    return True
        return False

    def get_valid_slots_for_duration(self, schedule_start_dt: datetime, schedule_end_dt: datetime, duration_tokens: int) -> list:
        """Finds all possible start tokens for an uninterruptible slot of a given duration."""
        valid_start_tokens = []
        duration_delta = timedelta(minutes=duration_tokens * self.TIME_BLOCK_MINUTES)

        current_dt = schedule_start_dt
        while current_dt < schedule_end_dt:
            # Only check times that align with the start of a token
            if (current_dt - schedule_start_dt).total_seconds() % (self.TIME_BLOCK_MINUTES * 60) == 0:
                slot_end_dt = current_dt + duration_delta

                # Check if both start and end times are within a valid shift period.
                # A simple check is that if start and end are on the same day, they must be in the same shift.
                if self._is_in_shift(current_dt) and self._is_in_shift(slot_end_dt - timedelta(minutes=1)):
                    # A more robust check for overnight shifts would be needed for tasks that can span midnight
                    # For now, we assume packing tasks don't span midnight.
                    if current_dt.date() == (slot_end_dt - timedelta(minutes=1)).date():
                         start_token = int((current_dt - schedule_start_dt).total_seconds() / 60 / self.TIME_BLOCK_MINUTES)
                         valid_start_tokens.append(start_token)

            current_dt += timedelta(minutes=self.TIME_BLOCK_MINUTES)
            
        return valid_start_tokens
    
    def is_slot_in_single_shift(self, start_dt: datetime, duration_tokens: int) -> bool:
        """Checks if a slot of a given duration falls entirely within one shift."""
        end_dt = start_dt + timedelta(minutes=duration_tokens * self.TIME_BLOCK_MINUTES)
        
        day_of_week = start_dt.weekday()
        if day_of_week not in self.shifts_by_day:
            return False

        for shift_start_time, shift_end_time in self.shifts_by_day[day_of_week]:
            # Simple case: shift does not cross midnight
            if shift_start_time <= shift_end_time:
                if start_dt.time() >= shift_start_time and end_dt.time() <= shift_end_time:
                    if start_dt.date() == (end_dt - timedelta(seconds=1)).date():
                        return True
            # Complex case: shift crosses midnight
            else:
                if start_dt.time() >= shift_start_time or (end_dt - timedelta(seconds=1)).time() < shift_end_time:
                     # This logic needs to be more robust for spanning midnight, but is a start
                    return True
        return False

# Place this near the top with the other dataclasses
@dataclass
class SubOrder:
    """Represents a piece of a larger order, sized to fit in a batch."""
    parent_order_no: str
    sub_order_id: str
    sku_id: str
    volume: float
    priority: int
    due_date: datetime

@dataclass
class HeuristicTask:
    """Represents a single, schedulable unit of work for the heuristic model."""
    # Core Identifiers
    task_id: str
    job_id: str  # e.g., Order number or bulk production ID
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

class HeuristicScheduler:
    def __init__(self, indents, skus, products, lines, tanks, equipments, shifts):
        self.indents = indents
        self.skus = skus
        self.products = products
        self.lines = lines
        self.tanks = tanks
        self.equipments = equipments
        self.shifts = shifts
        self.cip_circuits = config.CIP_CIRCUIT
        self.TIME_BLOCK_MINUTES = 15
        self.MAX_MASTER_BATCH_SIZE = 10000

        self.time_manager = TimeManager(self.shifts, self.TIME_BLOCK_MINUTES)
        self.schedule_start_dt = datetime.combine(datetime.now().date() + timedelta(days=1), time(22, 0))
        self.schedule_end_dt = self.schedule_start_dt + timedelta(days=2)
        self.schedule_start_token =0
        self.schedule_end_token = self._to_tokens(self.schedule_end_dt)
        

        self.master_task_list: List[HeuristicTask] = []
        self.task_lookup: Dict[str, HeuristicTask] = {}
        self.resource_timelines: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        self.log_entries: List[str] = []
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
        Main engine that implements a forward-chaining, ASAP (As Soon As Possible)
        scheduling model to guarantee correct process flow.
        """
        self.log_entries.append("--- Heuristic Scheduler Run Started (ASAP Model) ---")

        # 1. Generate all tasks and their dependencies first.
        self.log_entries.append("Step 1: Generating all tasks...")
        self.generate_all_tasks()
        self.log_entries.append(f"-> {len(self.master_task_list)} tasks generated.")

        # 2. Iteratively schedule tasks as their prerequisites are met.
        self.log_entries.append("\nStep 2: Finding ready tasks and scheduling them...")
        
        max_loops = len(self.master_task_list) * 2  # Safety break
        for i in range(max_loops):
            # Check if all tasks are done.
            if not any(t for t in self.master_task_list if t.status == ScheduleStatus.PENDING):
                self.log_entries.append(f"\n-> All tasks scheduled after {i} iterations.")
                break

            # Find all tasks whose prerequisites are now met.
            ready_tasks = self._get_ready_tasks()

            if not ready_tasks:
                self.log_entries.append("[ERROR] Deadlock detected. No ready tasks found but pending tasks still exist.")
                self.log_entries.append("\n[ERROR] Deadlock detected. Cannot continue scheduling.")
                break
            
            # Prioritize which ready task to schedule next.
            task_to_schedule = self._score_and_select_best_task(ready_tasks)
            
            self.log_entries.append(f"  [{i+1:>3}] Scheduling {task_to_schedule.task_id:<75}...")

            # Find the earliest possible start time based on when its prerequisites finish.
            earliest_start_token = self._get_prerequisite_finish_token(task_to_schedule)
            
            # Find the best placement using the robust forward-search function.
            placement = self._find_placement_for_block_task(task_to_schedule, earliest_start_token)
            
            if placement:
                self._book_task(task_to_schedule, placement)
                self.log_entries.append(" OK")
            else:
                self.log_entries.append(" FAILED")
                task_to_schedule.status = ScheduleStatus.FAILED
                self.log_entries.append(f"[FAIL] Could not find a valid slot for task {task_to_schedule.task_id}")

        else:
            self.log_entries.append(f"\n[WARNING] Scheduling loop reached max iterations. Check for unresolved dependencies.")
            print("[WARNING] Scheduling loop reached max iterations.")
        
        self.log_entries.append("\n--- Heuristic Scheduling Complete ---")
        self.log_entries.append("\nStep 3: Scheduling mandatory cleanup tasks...")
        # First, handle tanks that have been used and drained.
        self._schedule_drained_tank_cip()
        # Second, handle all other equipment for a final end-of-day clean.
        self._schedule_final_equipment_cip()
        self.log_entries.append("-> Cleanup CIP scheduling complete.")
        # --- END: MODIFICATION ---
        self._report_unscheduled_tasks()
        return self._create_scheduling_result()

    def _find_anchor_for_task(self, task: HeuristicTask) -> Optional[HeuristicTask]:
        """Traces a task chain forward to find its ultimate anchor task."""
        current_task = task
        while current_task.next_task:
            current_task = current_task.next_task
        
        if getattr(current_task, 'is_anchor_task', False):
            return current_task
        return None

    def _unbook_task_chain(self, anchor_task: HeuristicTask):
        """Un-books an anchor and all its prerequisites that have already been scheduled."""
        tasks_to_unbook = [anchor_task]
        
        # Use a queue to find all prerequisites recursively
        queue = list(anchor_task.prerequisites)
        while queue:
            task = queue.pop(0)
            # Check if we haven't already added this task to avoid infinite loops
            if task not in tasks_to_unbook:
                tasks_to_unbook.append(task)
                queue.extend(task.prerequisites)
        
        self.log_entries.append(f"[UNBOOK] Unbooking chain for anchor {anchor_task.task_id}. Tasks to reset: {[t.task_id for t in tasks_to_unbook]}")
        for task in tasks_to_unbook:
            if task.status == ScheduleStatus.BOOKED:
                # Remove from resource timelines
                for res_id, timeline in self.resource_timelines.items():
                    self.resource_timelines[res_id] = [booking for booking in timeline if booking[2] != task.task_id]
                
                # Reset task status
                task.status = ScheduleStatus.PENDING
                task.start_time = None
                task.end_time = None
                task.assigned_resource_id = None

    def _create_allocation_plan(self) -> List[Dict]:
            """
            Creates a production plan by aggregating demand into large "master batches"
            using proportional allocation based on finishing line speeds to increase parallelism.
            """
            self.log_entries.append("[INFO] Creating master batch plan using Proportional Draw-Down.")
            master_plan = []
            master_batch_counter = 0

            # 1. Group all valid indents by the product category.
            orders_by_product = defaultdict(list)
            for indent in self.indents.values():
                if indent.due_date < self.schedule_start_dt: continue
                sku = self.skus.get(indent.sku_id)
                if sku:
                    orders_by_product[sku.product_category].append(indent)

            for product_cat, orders in orders_by_product.items():
                # Sort orders by priority then due date to handle tie-breaks consistently
                orders.sort(key=lambda o: (o.priority.value, o.due_date))
                remaining_demand = {order.order_no: order.qty_required_liters for order in orders}

                # Pre-calculate the max packing speed for each SKU in this category
                line_speeds_by_sku = defaultdict(float)
                for order in orders:
                    sku_id = order.sku_id
                    if sku_id not in line_speeds_by_sku:
                        max_speed = 0
                        for line in self.lines.values():
                            if sku_id in getattr(line, 'compatible_skus_max_production', {}):
                                max_speed = max(max_speed, line.compatible_skus_max_production[sku_id])
                        line_speeds_by_sku[sku_id] = max_speed

                # 2. Continue creating master batches until all demand for the category is met.
                while sum(remaining_demand.values()) > 0:
                    bus_id = f"{product_cat}-master{master_batch_counter}"
                    bus_capacity = self.MAX_MASTER_BATCH_SIZE
                    bus_allocations = defaultdict(float)

                    # 3. First Pass: Allocate volume proportionally based on packing speed.
                    skus_with_demand = {self.indents[ono].sku_id for ono, dem in remaining_demand.items() if dem > 0}
                    total_packing_speed = sum(line_speeds_by_sku[sku_id] for sku_id in skus_with_demand)

                    if total_packing_speed > 0:
                        for order in orders:
                            if remaining_demand[order.order_no] > 0:
                                order_line_speed = line_speeds_by_sku[order.sku_id]
                                # Calculate this order's proportional share of the batch
                                proportional_share = (bus_capacity * (order_line_speed / total_packing_speed))
                                allocated_vol = min(proportional_share, remaining_demand[order.order_no])
                                
                                # To avoid tiny allocations, we can round to a reasonable number
                                allocated_vol = round(allocated_vol / 50) * 50
                                
                                bus_allocations[order.order_no] += allocated_vol
                                remaining_demand[order.order_no] -= allocated_vol
                    
                    # 4. Second Pass: Fill any remaining capacity in the batch FIFO style.
                    allocated_so_far = sum(bus_allocations.values())
                    if allocated_so_far < bus_capacity:
                        for order in orders:
                            if remaining_demand[order.order_no] > 0 and allocated_so_far < bus_capacity:
                                fill_amount = min(bus_capacity - allocated_so_far, remaining_demand[order.order_no])
                                bus_allocations[order.order_no] += fill_amount
                                allocated_so_far += fill_amount
                                remaining_demand[order.order_no] -= fill_amount

                    # 5. Finalize and add the new master batch to the plan.
                    final_bus_volume = sum(bus_allocations.values())
                    if final_bus_volume > 0:
                        master_plan.append({
                            "batch_id": bus_id,
                            "product_category": product_cat,
                            "volume": final_bus_volume,
                            "allocations": {k: v for k, v in bus_allocations.items() if v > 0} # Clean up zero allocations
                        })
                    master_batch_counter += 1
                    
            return master_plan

    def generate_all_tasks(self):
        """
        Orchestrates the entire task generation process.
        1. Creates a master plan of aggregated batches.
        2. Generates a branching bulk production graph for each master batch.
        3. For each final bulk task, it proportionally allocates its volume to all
            required finishing tasks ("cabs") to maximize parallelism.
        4. Includes a second "fill-up" pass to distribute any leftover volume.
        """
        self.log_entries.append("--- Task Generation ---")
        
        allocation_plan = self._create_allocation_plan()
        self.log_entries.append("--- Master Batch Allocation Breakdown ---")
        for batch_plan in allocation_plan:
            self.log_entries.append(
                f"  -> Batch '{batch_plan['batch_id']}' | Volume: {batch_plan['volume']}L | Supplies: {list(batch_plan['allocations'].items())}"
            )
        self.log_entries.append("------------------------------------")

        for batch_plan in allocation_plan:
            final_bulk_tasks = self._create_bulk_production_chain(batch_plan)
            if not final_bulk_tasks: continue

            # This dict tracks the remaining volume needed for each customer order.
            remaining_order_demand = batch_plan["allocations"].copy()
            cab_counter = 0

            # Pre-calculate line speeds for all SKUs in this batch.
            line_speeds_by_sku = defaultdict(float)
            for order_no in remaining_order_demand.keys():
                sku_id = self.indents[order_no].sku_id
                if sku_id not in line_speeds_by_sku:
                    max_speed = 0
                    for line in self.lines.values():
                        if sku_id in getattr(line, 'compatible_skus_max_production', {}):
                            max_speed = max(max_speed, line.compatible_skus_max_production[sku_id])
                    line_speeds_by_sku[sku_id] = max_speed

            # NEW LOGIC: Iterate through each bulk source and assign its volume proportionally.
            for source_task in final_bulk_tasks:
                source_volume = source_task.volume_liters
                allocations_for_this_source = defaultdict(float)

                # Calculate total packing speed of SKUs with remaining demand.
                skus_with_demand = {self.indents[ono].sku_id for ono, dem in remaining_order_demand.items() if dem > 0.1}
                total_packing_speed = sum(line_speeds_by_sku[sku_id] for sku_id in skus_with_demand)
                if total_packing_speed <= 0: continue

                # --- START: MODIFIED LOGIC FOR OVERFLOW ALLOTMENT ---

                # 1. First Pass: Proportional Allocation
                # Determine the proportional allocation, capping it by each order's remaining demand.
                for order_no, demand in remaining_order_demand.items():
                    if demand > 0.1:
                        order_sku_id = self.indents[order_no].sku_id
                        proportional_share = round((source_volume * (line_speeds_by_sku[order_sku_id] / total_packing_speed))/5)*5
                        allocations_for_this_source[order_no] = min(proportional_share, demand)
                
                # 2. Calculate Overflow
                # Find the unallocated volume after the proportional pass.
                allocated_so_far = sum(allocations_for_this_source.values())
                overflow_volume = source_volume - allocated_so_far

                # 3. Second Pass: Distribute Overflow
                # If there's overflow, distribute it to orders that can still take more volume.
                # We sort orders by priority to decide who gets the overflow first.
                if overflow_volume > 0.1:
                    self.log_entries.append(f"[OVERFLOW] Found {overflow_volume:.2f}L of overflow to re-allocate from {source_task.task_id}.")
                    
                    # Get a list of orders for sorting
                    orders_in_batch = [self.indents[ono] for ono in remaining_order_demand.keys()]
                    orders_in_batch.sort(key=lambda o: (o.priority.value, o.due_date))

                    for order in orders_in_batch:
                        if overflow_volume <= 0.1: break
                        
                        order_no = order.order_no
                        current_allocation = allocations_for_this_source.get(order_no, 0)
                        demand_for_order = remaining_order_demand.get(order_no, 0)
                        
                        # Check if this order can take more volume
                        if demand_for_order > current_allocation:
                            capacity_to_take = demand_for_order - current_allocation
                            fill_amount = min(overflow_volume, capacity_to_take)
                            
                            allocations_for_this_source[order_no] += fill_amount
                            overflow_volume -= fill_amount
                            self.log_entries.append(f"  -> Giving {fill_amount:.2f}L of overflow to {order_no}.")

                # --- END: MODIFIED LOGIC FOR OVERFLOW ALLOTMENT ---
                
                # Create a finishing task (cab) for each proportional allocation from this source.
                for order_no, volume_for_cab in allocations_for_this_source.items():
                    if volume_for_cab > 0.1:
                        indent = self.indents[order_no]
                        sub_order = SubOrder(
                            parent_order_no=order_no,
                            sub_order_id=f"{order_no}-cab{cab_counter}-from-{source_task.task_id}",
                            sku_id=indent.sku_id,
                            volume=volume_for_cab,
                            priority=indent.priority.value,
                            due_date=indent.due_date
                        )
                        # Link this cab to the current bulk source task.
                        self._create_finishing_chain(sub_order, source_task)
                        self.log_entries.append(f'[CAB ALLOTMENT] Cab{cab_counter}, Order_no {order_no}, Volume {volume_for_cab}')
                        
                        # Update the master demand tracking and counters.
                        remaining_order_demand[order_no] -= volume_for_cab
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
                volume=batch_plan["volume"], priority=Priority.MEDIUM.value
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
                            volume=volume_to_process, priority=parent_task.priority
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
                            volume=vol_for_first_split, priority=parent_task.priority
                        )
                        child_tasks.append(first_split_task)
                        remaining_volume -= vol_for_first_split

                        # Create the second child task with the remainder.
                        if remaining_volume > 0:
                            second_split_task = self._create_single_task(
                                job_id=parent_task.job_id, sku_id=parent_task.sku_id, batch_idx=len(next_leaf_tasks) + 1,
                                step=step, step_idx=step_idx, task_type=TaskType.BULK_PRODUCTION,
                                volume=remaining_volume, priority=parent_task.priority
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
        """Creates the finishing/packing tasks for a single sub-order and links it."""
        product = self.products.get(self.skus[sub_order.sku_id].product_category)
        if not product: return

        all_steps = product.processing_steps
        first_pack_idx = next((i for i, s in enumerate(all_steps) if getattr(s, 'process_type', None) == ProcessType.PACKAGING), -1)
        if first_pack_idx == -1: return

        finishing_steps = all_steps[first_pack_idx:]
        last_task_in_chain = None
        first_finishing_task = None
        
        for i, step in enumerate(finishing_steps):
            task = self._create_single_task(
                job_id=sub_order.sub_order_id, sku_id=sub_order.sku_id, batch_idx=0,
                step=step, step_idx=first_pack_idx + i, task_type=TaskType.FINISHING,
                volume=sub_order.volume, priority=sub_order.priority
            )
            if i == 0:
                first_finishing_task = task
            if last_task_in_chain:
                task.prerequisites.append(last_task_in_chain)
                task.previous_task = last_task_in_chain
                last_task_in_chain.next_task = task
            last_task_in_chain = task
                
        if first_finishing_task:
            first_finishing_task.prerequisites.append(final_bulk_task)
            final_bulk_task.next_task = first_finishing_task
            
            ### START: FIX FOR SOURCE LOCKING ###
            # This line was missing. It ensures the first packaging task knows its direct predecessor.
            first_finishing_task.previous_task = final_bulk_task
            ### END: FIX FOR SOURCE LOCKING ###

            self.log_entries.append(f"[FINAL-LINK] Linked bus task {final_bulk_task.task_id} to cab task {first_finishing_task.task_id}")

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

    def _create_single_task(self, job_id, sku_id, batch_idx, step, step_idx, task_type, volume, priority) -> HeuristicTask:
        """Helper to create one HeuristicTask instance and add it to the master lists."""
        task_id = f"{job_id}-{sku_id}-b{batch_idx}-{step.step_id}"
        
        compatible_resources = defaultdict(list)
        for req in step.requirements:
            # --- NEW: SKU-to-Line Compatibility Logic ---
            if task_type == TaskType.FINISHING and req.resource_type == ResourceType.LINE:
                filtered_line_ids = []
                for line_id in req.compatible_ids:
                    line = self.lines.get(line_id)
                    # Only add the line if it is explicitly compatible with the SKU.
                    if line and sku_id in getattr(line, 'compatible_skus_max_production', {}):
                        filtered_line_ids.append(line_id)
                compatible_resources[req.resource_type].extend(filtered_line_ids)
            # --- END: New Logic ---
            else:
                # For all other cases, use the requirements as is.
                compatible_resources[req.resource_type].extend(req.compatible_ids)
        
        task = HeuristicTask(
            task_id=task_id,
            job_id=job_id,
            sku_id=sku_id,
            step=step,
            step_idx=step_idx,
            task_type=task_type,
            compatible_resources=dict(compatible_resources),
            volume_liters=volume,
            priority=priority
        )

        task.base_duration_tokens = self._calculate_dynamic_duration_tokens(task)

        self.log_entries.append(f"Appending task: {task.task_id} to master task list")
        self.master_task_list.append(task)
        self.task_lookup[task.task_id] = task
        return task

    def _calculate_dynamic_duration_tokens(self, task: HeuristicTask) -> int:
        """
        Calculates the task duration in tokens, dynamically and safely.
        It finds the most constraining resource (slowest speed) if multiple are compatible.
        """
        step = task.step
        base_minutes = getattr(step, 'duration_minutes', 60)
        
        longest_duration_minutes = 0

        for req in step.requirements:
            res_type = req.resource_type
            
            # For Lines and Equipment, duration is volume-based
            if res_type == ResourceType.LINE or res_type == ResourceType.EQUIPMENT:
                for res_id in req.compatible_ids:
                    processing_speed = 0
                    if res_type == ResourceType.LINE:
                        line = self.lines.get(res_id)
                        # Safely get the speed for the specific SKU
                        if line:
                            speeds = getattr(line, 'compatible_skus_max_production', {})
                            processing_speed = speeds.get(task.sku_id, 0)
                    
                    elif res_type == ResourceType.EQUIPMENT:
                        equipment = self.equipments.get(res_id)
                        if equipment:
                            processing_speed = getattr(equipment, 'processing_speed', 0)

                    if processing_speed > 0:
                        work_time = (task.volume_liters / processing_speed)
                        # Find the max possible work time among all compatible resources
                        longest_duration_minutes = max(longest_duration_minutes, work_time)
                    else:
                        self.log_entries.append(f"[WARNING] Zero processing speed for task {task.task_id} on resource {res_id}. Check config.")

        # If we calculated a valid process-driven duration, use it. Otherwise, use the fixed base time.
        final_duration_minutes = longest_duration_minutes if longest_duration_minutes > 0 else base_minutes

        # Add setup and cooldown times, which are always fixed for the step
        final_duration_minutes += getattr(step, 'setup_time', 0) + getattr(step, 'cool_down', 0)
        
        if final_duration_minutes <= 0:
            self.log_entries.append(f"                  [WARNING] Calculated duration for task {task.task_id} is zero or negative. Defaulting to 1 token.")
            return 1

        return self._round_up_duration_to_tokens(final_duration_minutes)

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

    def _find_anchor_slot(self, task: HeuristicTask, search_after_token: int = 0) -> Optional[Dict]:
        """Finds the earliest valid slot for an anchor task, now handling ZERO_STAGNATION."""
        
        # --- NEW: Dispatch based on scheduling rule ---
        scheduling_rule = getattr(task.step, 'scheduling_rule', SchedulingRule.DEFAULT)
        if scheduling_rule == SchedulingRule.ZERO_STAGNATION:
            start_search = max(task.total_prereq_duration_tokens, search_after_token)
            return self._find_placement_for_block_task(task, start_search) # Reuse the logic we just built
        # --- END: New Dispatch Logic ---

        task_duration_tokens = self._calculate_dynamic_duration_tokens(task)
        search_token = max(task.total_prereq_duration_tokens, search_after_token)

        while search_token < self.schedule_end_token:
            end_token = search_token + task_duration_tokens
            if end_token > self.schedule_end_token: break

            if not self.time_manager.is_slot_in_single_shift(self._to_datetime(search_token), task_duration_tokens):
                search_token += 1
                continue

            placements, true_start_token = self._get_concurrent_placements_at_token(task, search_token, end_token)

            if placements:
                if true_start_token > search_token:
                    self.log_entries.append(f"[ANCHOR-INFO] Task {task.task_id} must wait... New start token: {true_start_token}")
                    search_token = true_start_token
                    continue
                return {"resources": list(placements.values()), "start_token": search_token, "end_token": end_token}
            else:
                self.log_entries.append(f"[ANCHOR-DEBUG] Resource busy for {task.task_id} at {search_token}. Next available: {true_start_token}.")
                search_token = max(true_start_token, search_token + 1)

        self.log_entries.append(f"[FAIL-ANCHOR] Exhausted all search options for {task.task_id}.")
        return None
    
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
        Selects the best task to schedule next.
        This version ensures that all steps for a single sub-batch are prioritized
        to maintain a logical processing sequence (e.g., Fill -> Process -> Drain).
        """
        
        def get_batch_identifier(task_id: str) -> str:
            """
            Helper to extract a consistent identifier for a sub-batch from a task ID.
            e.g., 'PROD-master0-PROD-b0-STEP' -> 'PROD-master0-PROD-b0'
            """
            parts = task_id.split('-')
            for i, part in enumerate(parts):
                if part.startswith('b') and part[1:].isdigit():
                    return "-".join(parts[:i+1])
            # Fallback for tasks without a clear batch index (like finishing tasks)
            return task_id.rsplit('-', 1)[0]

        # Sort tasks by priority (high to low), then by their batch identifier (to keep batches intact),
        # and finally by their step index (to ensure correct order within a batch).
        tasks_to_score.sort(key=lambda t: (
            -t.priority, 
            get_batch_identifier(t.task_id),
            t.step_idx
        ))
        
        best_task = tasks_to_score[0]
        self.log_entries.append(f"[SELECT] Selected task {best_task.task_id} (Job: {best_task.job_id}, Prio: {best_task.priority}, Step: {best_task.step_idx})")
        
        return best_task

    def _is_resource_free(self, resource_id: str, start_token: int, end_token: int, requesting_task: HeuristicTask) -> Tuple[bool, int]:
        """
        Checks if a resource is free for a given time window.
        This final version removes the complex "dirty tank" logic that caused the deadlock,
        relying on the robust setup time calculation for all CIP decisions.
        """
        # 1. Standard check for time collision
        for booked_start, booked_end, t_id in self.resource_timelines.get(resource_id, []):
            if booked_start < end_token and booked_end > start_token:
                return False, booked_end

        # 2. Calculate setup time based on product changeover. This is now the ONLY check for CIP.
        last_task_end_token, last_task_id = self._get_last_task_on_resource(resource_id, start_token)
        # Use the product category of the requesting task for the check
        current_product_cat = self.skus[requesting_task.sku_id].product_category if requesting_task.sku_id in self.skus else requesting_task.sku_id
        setup_minutes = self._calculate_setup_time(last_task_id, current_product_cat, resource_id)

        if setup_minutes == 0:
            return True, -1  # Slot is free and no setup CIP is needed.

        # If we are here, a CIP is required due to product changeover.
        cip_duration_tokens = self._round_up_duration_to_tokens(setup_minutes)
        compatible_circuits = [c.circuit_id for c in self.cip_circuits.values() if resource_id in c.connected_resource_ids]
        
        # If no CIP circuit is available for this resource, we can't clean it.
        # For now, we allow the task, assuming manual cleaning or a data error.
        if not compatible_circuits: 
            self.log_entries.append(f"[WARNING-CIP] No CIP circuit for {resource_id} to perform required changeover clean.")
            return True, -1

        # Find the earliest time the CIP can start (after the last task on the resource).
        earliest_cip_start_token = last_task_end_token
        
        # Now, find the earliest time that BOTH the resource AND a circuit are free.
        while True:
            cip_slot_end = earliest_cip_start_token + cip_duration_tokens
            if cip_slot_end > self.schedule_end_token:
                return False, self.schedule_end_token # CIP doesn't fit in schedule horizon

            free_circuit_found = False
            for circuit_id in compatible_circuits:
                is_circuit_busy = any(bs < cip_slot_end and be > earliest_cip_start_token for bs, be, _ in self.resource_timelines.get(circuit_id, []))
                if not is_circuit_busy:
                    free_circuit_found = True
                    break
            
            if free_circuit_found:
                break # A free circuit was found for this time slot
            
            # If all circuits are busy, inch forward the search time.
            earliest_cip_start_token += 1

        # This is the first moment the resource will be free after the mandatory CIP.
        effective_busy_until = earliest_cip_start_token + cip_duration_tokens
        
        # If the requested start time is before the resource is truly free, deny the placement.
        if start_token < effective_busy_until:
            return False, effective_busy_until

        return True, -1
    
    def _find_best_placement_for_task(self, task: HeuristicTask) -> Optional[Dict]:
        """
        Finds the best available time slot, applying sophisticated look-ahead
        resource blocking for ZERO_STAGNATION tasks.
        """
        earliest_prereq_finish_token = self._get_prerequisite_finish_token(task)
        
        # --- START OF FIX ---
        # We must check the SCHEDULING_RULE, not the PROCESS_TYPE.
        scheduling_rule = getattr(task.step, 'scheduling_rule', SchedulingRule.DEFAULT)
        # --- END OF FIX ---

        if scheduling_rule == SchedulingRule.ZERO_STAGNATION:
            self.log_entries.append(f"[INFO] Task {task.task_id} is ZERO_STAGNATION. Applying Look-Ahead Blocking.")
            return self._find_placement_for_flow_task(task, earliest_prereq_finish_token)
        else:
            # Standard block tasks use the simpler gap-finding logic
            return self._find_placement_for_block_task(task, earliest_prereq_finish_token)
            
    def _find_placement_for_block_task(self, task: HeuristicTask, start_after_token: int) -> Optional[Dict]:
        """
        Finds the earliest available gap for any task. This corrected version uses a single,
        robust placement-finding logic for all task types.
        """
        search_token = start_after_token
        duration_tokens = self._calculate_dynamic_duration_tokens(task)

        while search_token < self.schedule_end_token:
            # ALL tasks now use the same, correct function to find concurrent placements.
            placements, true_start_token = self._get_concurrent_placements_at_token(task, search_token, duration_tokens)
            
            if placements:
                final_end_token = true_start_token + duration_tokens
                if final_end_token > self.schedule_end_token:
                    search_token = true_start_token + 1
                    continue
                # If a valid placement is found, return it immediately.
                return {"resources": list(placements.values()), "start_token": true_start_token, "end_token": final_end_token}
            else:
                # If no placement is found, intelligently jump the search forward.
                search_token = max(true_start_token, search_token + 1)
        
        # If the loop finishes without finding a slot, return None.
        return None

    def _find_placement_for_flow_task(self, task: HeuristicTask, start_at_token: int) -> Tuple[Optional[Dict], int]:
        """
        Finds a placement for a FLOW task, now correctly using the task-specific
        compatible resources for the primary task being scheduled.
        """
        duration_tokens = self._calculate_dynamic_duration_tokens(task)
        end_token = start_at_token + duration_tokens
        if end_token > self.schedule_end_token:
            return None, self.schedule_end_token

        all_reqs = self._get_task_resource_reqs(task).copy()
        if task.previous_task:
            all_reqs.extend(self._get_task_resource_reqs(task.previous_task))
        if task.next_task:
            all_reqs.extend(self._get_task_resource_reqs(task.next_task))
        
        chosen_resource_ids = set()
        max_busy_until = 0

        for req in all_reqs:
            found_resource_for_this_req = False

            # --- START: THE MISSING FIX ---
            # Determine which list of compatible IDs to use.
            if req in task.step.requirements:
                # If the requirement belongs to the main task, use its SKU-specific filtered list.
                ids_to_check = task.compatible_resources.get(req.resource_type, [])
            else:
                # If the requirement is from a source/destination task, use its generic list.
                ids_to_check = req.compatible_ids
            # --- END: THE MISSING FIX ---

            for resource_id in ids_to_check:
                if resource_id in chosen_resource_ids: continue
                is_free, busy_until = self._is_resource_free(resource_id, start_at_token, end_token, task)
                if self._is_resource_of_type(resource_id, req.resource_type) and is_free:
                    chosen_resource_ids.add(resource_id)
                    found_resource_for_this_req = True
                    break
                else:
                    max_busy_until = max(max_busy_until, busy_until)
            
            if not found_resource_for_this_req:
                return None, max_busy_until
        
        return {"resources": list(chosen_resource_ids), "start_token": start_at_token, "end_token": end_token}, start_at_token
    
    def _find_backfill_slot(self, task: HeuristicTask) -> Optional[Dict]:
        """
        Works backwards from a task's successor(s) to find the LATEST possible
        just-in-time slot, searching backwards if the ideal slot is taken.
        This version correctly handles 1-to-N splits.
        """
        # Find all successors by checking which tasks have 'self' as a prerequisite.
        successors = [t for t in self.master_task_list if task in t.prerequisites]
        booked_successors = [s for s in successors if s.status == ScheduleStatus.BOOKED]

        if not booked_successors:
            self.log_entries.append(f"[BACKFILL-FAIL] Task {task.task_id} has no booked successor.")
            return None

        # The constraint is the start time of the EARLIEST starting successor.
        successor_start_token = min(self._to_tokens(s.start_time) for s in booked_successors)
        task_duration_tokens = self._calculate_dynamic_duration_tokens(task)
        desired_start_token = successor_start_token - task_duration_tokens
        
        # We search backwards until we hit the end of the latest prerequisite
        earliest_possible_start = self._get_prerequisite_finish_token(task)

        search_token = desired_start_token
        while search_token >= earliest_possible_start:
            
            placements, true_start_token = self._get_concurrent_placements_at_token(task, search_token, task_duration_tokens)
            # A valid backfill slot is only found if the placement can start
            # at the exact token we are currently checking.
            if placements and true_start_token == search_token:
                self.log_entries.append(
                    f"[BACKFILL-FOUND] Found JIT slot for {task.task_id} at exact token {search_token}."
                )
                return {
                    "resources": list(placements.values()),
                    "start_token": search_token,
                    "end_token": search_token + task_duration_tokens,
                }

            search_token -= 1

        self.log_entries.append(
            f"[BACKFILL-FAIL] No available JIT slot found for {task.task_id} in window [{earliest_possible_start}, {desired_start_token}]."
        )
        return None

    def _book_task(self, task: HeuristicTask, placement: Dict):
        """
        This is the final, corrected booking function. It includes:
        1. Multi-resource CIP checks.
        2. A definitive "Ghost Process" check to ensure a tank's contents match the product being drained.
        """
        task_start_token = placement["start_token"]
        task_end_token = placement["end_token"]
        all_resources_in_placement = placement["resources"]

        # --- Iterative CIP check for all involved resources ---
        for resource_id in all_resources_in_placement:
            last_task_on_res_end_token, last_task_on_res_id = self._get_last_task_on_resource(resource_id, task_start_token)
            current_product_cat = self.skus[task.sku_id].product_category if task.sku_id in self.skus else task.sku_id
            setup_minutes = self._calculate_setup_time(last_task_on_res_id, current_product_cat, resource_id)

            if setup_minutes > 0:
                cip_duration_tokens = self._round_up_duration_to_tokens(setup_minutes)
                cip_start_token = task_start_token - cip_duration_tokens
                circuit_to_book = next((c.circuit_id for c in self.cip_circuits.values() if resource_id in c.connected_resource_ids), None)
                if circuit_to_book:
                    cip_task_id = f"CIP-{resource_id}-for-{task.task_id}"
                    if cip_task_id not in self.task_lookup:
                        cip_task = HeuristicTask(task_id=cip_task_id, job_id=task.job_id, sku_id="CIP", step=task.step, task_type=TaskType.CIP, base_duration_tokens=cip_duration_tokens, status=ScheduleStatus.BOOKED, assigned_resource_id=f"{resource_id},{circuit_to_book}", start_time=self._to_datetime(cip_start_token), end_time=self._to_datetime(task_start_token))
                        self.master_task_list.append(cip_task)
                        self.task_lookup[cip_task.task_id] = cip_task
                        self.resource_timelines[resource_id].append((cip_start_token, task_start_token, cip_task.task_id))
                        self.resource_timelines[circuit_to_book].append((cip_start_token, task_start_token, cip_task.task_id))

        task.status = ScheduleStatus.BOOKED
        task.start_time = self._to_datetime(task_start_token)
        task.end_time = self._to_datetime(task_end_token)
        task.assigned_resource_id = str(all_resources_in_placement)

        scheduling_rule = getattr(task.step, 'scheduling_rule', SchedulingRule.DEFAULT)
        if scheduling_rule == SchedulingRule.ZERO_STAGNATION:
            if task.previous_task and task.previous_task.assigned_resource_id:
                source_resource_id = task.previous_task.assigned_resource_id.strip("[]'\"").split(',')[0]
                
                # --- START: ROBUST GHOST PROCESS PREVENTION ---
                if task.task_type == TaskType.FINISHING:
                    product_needed = self.skus[task.sku_id].product_category
                    product_in_tank = None
                    
                    for _, _, t_id in reversed(self.resource_timelines.get(source_resource_id, [])):
                        if self.task_lookup[t_id].end_time > task.start_time: continue
                        task_to_check = self.task_lookup.get(t_id)
                        if not task_to_check: continue

                        if task_to_check.task_type == TaskType.CIP:
                            product_in_tank = "EMPTY"; break
                        if task_to_check.task_type == TaskType.BULK_PRODUCTION:
                            product_in_tank = self.skus[task_to_check.sku_id].product_category if task_to_check.sku_id in self.skus else task_to_check.sku_id
                            break
                        if task_to_check.task_type == TaskType.LOCKED and 'SOURCE' not in task_to_check.task_id:
                            try:
                                original_task_id = task_to_check.task_id.split('-for-')[1].split('-on-')[0]
                                original_task = self.task_lookup.get(original_task_id)
                                if original_task:
                                    product_in_tank = self.skus[original_task.sku_id].product_category if original_task.sku_id in self.skus else original_task.sku_id
                                    break
                            except (IndexError, KeyError): continue

                    if product_in_tank and product_in_tank != "EMPTY" and product_in_tank != product_needed:
                        self.log_entries.append(f"[GHOST-PROCESS-FAIL] Task {task.task_id} needs '{product_needed}' but tank {source_resource_id} contains '{product_in_tank}'. Booking failed.")
                        task.status = ScheduleStatus.FAILED; task.start_time = None; task.end_time = None; task.assigned_resource_id = None
                        return
                # --- END: ROBUST GHOST PROCESS PREVENTION ---

                locked_source_task_id = f"LOCKED-SOURCE-for-{task.task_id}-on-{source_resource_id}"
                if locked_source_task_id not in self.task_lookup:
                    locked_task = HeuristicTask(task_id=locked_source_task_id, job_id=task.job_id, sku_id="LOCKED", step=task.step, task_type=TaskType.LOCKED, base_duration_tokens=(task_end_token - task_start_token), status=ScheduleStatus.BOOKED, assigned_resource_id=source_resource_id, start_time=task.start_time, end_time=task.end_time)
                    self.master_task_list.append(locked_task); self.task_lookup[locked_task.task_id] = locked_task
                    self.resource_timelines[source_resource_id].append((task_start_token, task_end_token, locked_task.task_id))

            main_resource_id, locked_resources = None, []
            if task.step.requirements:
                primary_req_type = task.step.requirements[0].resource_type
                for res_id in all_resources_in_placement:
                    if self._is_resource_of_type(res_id, primary_req_type):
                        main_resource_id = res_id; break
            if main_resource_id: locked_resources = [res for res in all_resources_in_placement if res != main_resource_id]
            else: main_resource_id = all_resources_in_placement[0]; locked_resources = all_resources_in_placement[1:]
            self.resource_timelines[main_resource_id].append((task_start_token, task_end_token, task.task_id))

            for locked_res_id in locked_resources:
                if locked_res_id == locals().get('source_resource_id'): continue
                locked_task_id = f"LOCKED-for-{task.task_id}-on-{locked_res_id}"
                if locked_task_id not in self.task_lookup:
                    locked_task = HeuristicTask(task_id=locked_task_id, job_id=task.job_id, sku_id="LOCKED", step=task.step, task_type=TaskType.LOCKED, base_duration_tokens=(task_end_token - task_start_token), status=ScheduleStatus.BOOKED, assigned_resource_id=locked_res_id, start_time=task.start_time, end_time=task.end_time)
                    self.master_task_list.append(locked_task); self.task_lookup[locked_task.task_id] = locked_task
                    self.resource_timelines[locked_res_id].append((task_start_token, task_end_token, locked_task.task_id))
        else:
            for resource_id in all_resources_in_placement:
                self.resource_timelines[resource_id].append((task_start_token, task_end_token, task.task_id))
        
        all_affected_resources = set(all_resources_in_placement)
        if 'source_resource_id' in locals() and locals()['source_resource_id']: all_affected_resources.add(locals()['source_resource_id'])
        for res_id in all_affected_resources:
            if res_id in self.resource_timelines: self.resource_timelines[res_id].sort()

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
        Specifically finds tanks that have been drained by packing/transfer tasks
        and schedules a CIP immediately after they are empty.
        """
        self.log_entries.append("  -> Checking for drained tanks that need CIP...")
        
        for tank_id in self.tanks.keys():
            if tank_id not in self.resource_timelines:
                continue

            # Find the end time of the very last task that locked this tank as a source.
            # This signifies when the tank was fully drained.
            last_drain_token = 0
            for start, end, task_id in self.resource_timelines[tank_id]:
                task = self.task_lookup.get(task_id)
                # A 'LOCKED' or 'LOCKED-SOURCE' task indicates the tank is being drained.
                if task and task.task_type == TaskType.LOCKED:
                    if end > last_drain_token:
                        last_drain_token = end
            
            if last_drain_token == 0:
                # This tank was never used as a source for draining.
                continue

            # Check if a CIP already exists after the drain time
            has_future_cip = any(
                self.task_lookup.get(t_id).task_type == TaskType.CIP and s >= last_drain_token
                for s, e, t_id in self.resource_timelines[tank_id]
            )
            if has_future_cip:
                self.log_entries.append(f"  [TANK-CIP] Skipping {tank_id}, a cleanup CIP is already scheduled.")
                continue

            self.log_entries.append(f"  [TANK-CIP] Tank {tank_id} was drained at token {last_drain_token}, scheduling CIP.")
            # This tank was drained and not cleaned. Schedule a CIP now.
            # (This reuses the booking logic from the general cleanup for consistency)
            self._schedule_cip_for_resource(tank_id, last_drain_token)


    def _schedule_final_equipment_cip(self):
        """
        Schedules a final CIP for any non-tank equipment (Lines, etc.) that was
        used and not left in a clean state at the end of the schedule.
        """
        self.log_entries.append("  -> Checking for other equipment needing final CIP...")
        
        # Combine all non-tank, non-CIP-system equipment
        other_equipment_ids = list(self.lines.keys()) + list(self.equipments.keys())

        for resource_id in other_equipment_ids:
            if resource_id not in self.resource_timelines or not self.resource_timelines[resource_id]:
                continue
            
            # Get the very last task on this resource's timeline.
            last_booking = self.resource_timelines[resource_id][-1]
            last_task_end_token = last_booking[1]
            last_task_id = last_booking[2]
            last_task_obj = self.task_lookup.get(last_task_id)

            if last_task_obj and last_task_obj.task_type == TaskType.CIP:
                continue

            self.log_entries.append(f"  [FINAL-CIP] Equipment {resource_id} needs a final CIP.")
            self._schedule_cip_for_resource(resource_id, last_task_end_token)
            

    def _schedule_cip_for_resource(self, resource_id: str, start_after_token: int):
        """Helper function to find a slot and book a CIP task for a given resource."""
        
        resource_object = self.tanks.get(resource_id) or self.lines.get(resource_id) or self.equipments.get(resource_id)
        if not resource_object or not hasattr(resource_object, 'CIP_duration_minutes'):
            self.log_entries.append(f"  [CIP-HELPER-WARN] No CIP duration for {resource_id}.")
            return

        cip_duration_minutes = getattr(resource_object, 'CIP_duration_minutes', 90)
        cip_duration_tokens = self._round_up_duration_to_tokens(cip_duration_minutes)

        circuit_to_use = next((c.circuit_id for c in self.cip_circuits.values() if resource_id in c.connected_resource_ids), None)
        if not circuit_to_use:
            self.log_entries.append(f"  [CIP-HELPER-WARN] No compatible CIP circuit for {resource_id}.")
            return

        # Find the earliest available slot for the CIP task
        search_token = start_after_token
        extended_schedule_end_token = self.schedule_end_token + 192 # Allow 2 extra days

        while search_token < extended_schedule_end_token:
            cip_end_token = search_token + cip_duration_tokens
            
            is_resource_busy = any(start < cip_end_token and end > search_token for start, end, _ in self.resource_timelines.get(resource_id, []))
            is_circuit_busy = any(start < cip_end_token and end > search_token for start, end, _ in self.resource_timelines.get(circuit_to_use, []))

            if not is_resource_busy and not is_circuit_busy:
                cip_task_id = f"AUTO-CIP-on-{resource_id}"
                cip_task = HeuristicTask(
                    task_id=cip_task_id, job_id="CLEANUP", sku_id="CIP", step=None,
                    task_type=TaskType.CIP, base_duration_tokens=cip_duration_tokens,
                    status=ScheduleStatus.BOOKED, assigned_resource_id=f"{resource_id},{circuit_to_use}",
                    start_time=self._to_datetime(search_token), end_time=self._to_datetime(cip_end_token)
                )
                self.master_task_list.append(cip_task)
                self.task_lookup[cip_task.task_id] = cip_task
                self.resource_timelines[resource_id].append((search_token, cip_end_token, cip_task_id))
                self.resource_timelines[circuit_to_use].append((search_token, cip_end_token, cip_task_id))
                self.resource_timelines[resource_id].sort()
                self.resource_timelines[circuit_to_use].sort()
                self.log_entries.append(f"  [CIP-HELPER-BOOKED] Task {cip_task_id} on {resource_id} & {circuit_to_use} @ token {search_token}")
                return

            search_token += 1

        self.log_entries.append(f"  [CIP-HELPER-FAIL] Could not find slot for CIP on {resource_id}.")

    def _get_concurrent_placements_at_token(self, task: HeuristicTask, start_token: int, duration_tokens: int) -> Tuple[Optional[Dict], int]:
        """
        Checks for concurrent resource availability, using the robust intersection-based
        resource affinity logic and now also handling ZERO_STAGNATION look-ahead for destinations.
        """
        placements = {}
        latest_required_start_time = start_token
        end_token = start_token + duration_tokens

        ### START: FINAL AFFINITY FIX (INTERSECTION LOGIC) ###
        constrained_resource_id = None
        if task.previous_task and task.previous_task.assigned_resource_id:
            # 1. Get the list of all resources that were locked by the previous task.
            prev_resources_str = task.previous_task.assigned_resource_id
            cleaned_str = prev_resources_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace(' ', '')
            prev_locked_resources = cleaned_str.split(',')

            # 2. Get the list of all compatible tanks for the CURRENT step.
            current_compatible_tanks = []
            for req in task.step.requirements:
                if req.resource_type == ResourceType.TANK:
                    current_compatible_tanks.extend(task.compatible_resources.get(req.resource_type, []))

            # 3. Find the intersection: the resource that is in BOTH lists.
            for res_id in prev_locked_resources:
                if res_id in current_compatible_tanks:
                    constrained_resource_id = res_id
                    self.log_entries.append(f"[AFFINITY] Task {task.task_id} constrained to {res_id} via intersection logic.")
                    break # Found our resource.
        ### END: FINAL AFFINITY FIX (INTERSECTION LOGIC) ###
        
        # --- START: MODIFICATION FOR DESTINATION LOCKING ---
        # Get the base requirements for the current task
        all_requirements = self._get_task_resource_reqs(task)[:]

        # If the task is a flow task and has a successor, add the successor's requirements
        # to the list of resources we need to find concurrently.
        scheduling_rule = getattr(task.step, 'scheduling_rule', SchedulingRule.DEFAULT)
        if scheduling_rule == SchedulingRule.ZERO_STAGNATION and task.next_task:
            # The 'next_task' attribute is a list, so we get the first successor.
            # This assumes a simple linear flow for destination locking.
            next_task_in_chain = task.next_task[0] if isinstance(task.next_task, list) and task.next_task else task.next_task
            if next_task_in_chain:
                self.log_entries.append(f"[FLOW-LOOKAHEAD] Task {task.task_id} is looking ahead for resources for {next_task_in_chain.task_id}")
                next_task_reqs = self._get_task_resource_reqs(next_task_in_chain)
                all_requirements.extend(next_task_reqs)
        # --- END: MODIFICATION FOR DESTINATION LOCKING ---


        for req in all_requirements:
            best_resource_for_req = None
            earliest_available_token = float('inf')

            compatible_ids_for_task = task.compatible_resources.get(req.resource_type, [])
            
            # If the requirement is from the NEXT task, we need its compatible resources, not the current task's.
            # This is a fallback in case the task's own compatible_resources dict doesn't cover the successor.
            if not compatible_ids_for_task and 'next_task_reqs' in locals() and req in next_task_reqs:
                 # We need to find the compatible resources for the next_task_in_chain
                 # This part of logic is complex, for now we will assume the primary task's compatible_resources are sufficient.
                 # A more robust solution would be to pass the next_task_in_chain object and get its specific compatibilities.
                 ids_to_check = req.compatible_ids
            else:
                 ids_to_check = compatible_ids_for_task


            if constrained_resource_id and req.resource_type == ResourceType.TANK:
                if constrained_resource_id in ids_to_check:
                    ids_to_check = [constrained_resource_id]
                else:
                    self.log_entries.append(f"[ERROR-AFFINITY] Constrained resource {constrained_resource_id} not in compatible list for {task.task_id}")
                    return None, self.schedule_end_token

            for resource_id in ids_to_check:
                is_free, busy_until = self._is_resource_free(resource_id, start_token, end_token, task)
                
                if is_free:
                    current_best_token = start_token
                else:
                    current_best_token = busy_until

                if current_best_token < earliest_available_token:
                    earliest_available_token = current_best_token
                    best_resource_for_req = resource_id

            if best_resource_for_req:
                placements[req.resource_type] = best_resource_for_req
                latest_required_start_time = max(latest_required_start_time, earliest_available_token)
            else:
                if constrained_resource_id and req.resource_type == ResourceType.TANK:
                    self.log_entries.append(f"[FAIL-AFFINITY] Could not place {task.task_id} on required resource {constrained_resource_id}")
                return None, self.schedule_end_token

        if latest_required_start_time + duration_tokens > self.schedule_end_token:
            return None, self.schedule_end_token

        for resource_id in placements.values():
            is_free, busy_until = self._is_resource_free(resource_id, latest_required_start_time, latest_required_start_time + duration_tokens, task)
            if not is_free:
                return None, busy_until

        return placements, latest_required_start_time
    
    def _get_prerequisite_finish_token(self, task: HeuristicTask) -> int:
        """Calculates the token when all prerequisites for a task are complete."""
        finish_token = self.schedule_start_token
        if task.prerequisites:
            # In a forward model, all prerequisites are guaranteed to be booked.
            # We find the latest finish time among them.
            end_times = [self._to_tokens(p.end_time) for p in task.prerequisites if p.status == ScheduleStatus.BOOKED and p.end_time]
            if end_times:
                finish_token = max(end_times)
        return finish_token
    
    def _is_resource_of_type(self, resource_id: str, resource_type: ResourceType) -> bool:
        """Verifies that a resource ID belongs to the correct resource category."""
        if resource_type == ResourceType.LINE and resource_id in self.lines: return True
        if resource_type == ResourceType.TANK and resource_id in self.tanks: return True
        if resource_type == ResourceType.EQUIPMENT and resource_id in self.equipments: return True
        # Add other resource types as needed (e.g., ROOM)
        return False

    def _calculate_setup_time(self, last_task_id: Optional[str], current_sku_id: str, resource_id: str) -> int:
        """
        Calculates the required setup/CIP time. This final version correctly
        handles LOCKED tasks and the different uses of `sku_id` for bulk vs finishing tasks.
        """
        if not last_task_id:
            return 0 # No preceding task, no setup needed.

        last_task = self.task_lookup.get(last_task_id)
        if not last_task:
            return 90  # Fallback CIP time if task lookup fails.

        if last_task.task_type == TaskType.CIP:
            return 0

        if last_task.task_type == TaskType.LOCKED:
            ### START: FIX FOR CIP REGRESSION ###
            # Make the parsing more robust to handle different LOCKED task name formats.
            parts = last_task.task_id.split('-for-')
            if len(parts) > 1:
                original_task_id = parts[1].split('-on-')[0]
                original_task = self.task_lookup.get(original_task_id)
                if original_task:
                    last_task = original_task # Overwrite last_task for the check below.
            else:
                # If the name is not in the expected format, we cannot determine the product.
                # We must assume a major clean is required for safety.
                self.log_entries.append(f"[WARNING-CIP] Could not parse original task from LOCKED task: {last_task.task_id}. Assuming CIP is required.")
                return 90
            ### END: FIX FOR CIP REGRESSION ###

        last_sku_obj = self.skus.get(last_task.sku_id)
        last_category = last_sku_obj.product_category if last_sku_obj else last_task.sku_id

        current_sku_obj = self.skus.get(current_sku_id)
        current_category = current_sku_obj.product_category if current_sku_obj else current_sku_id

        if not last_category or not current_category:
            return 0

        if last_category != current_category:
            resource_object = self.tanks.get(resource_id) or self.lines.get(resource_id) or self.equipments.get(resource_id)
            if resource_object and hasattr(resource_object, 'CIP_duration_minutes'):
                return getattr(resource_object, 'CIP_duration_minutes', 90)
            return 90
        
        return 0

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
            
            try:
                priority_val = self.indents.get(base_order_no).priority
                if isinstance(priority_val, int):
                    priority_val = Priority(priority_val)
            except:
                priority_val = Priority.MEDIUM # Default


            if task.task_type == TaskType.CIP:
                primary_resource = task.assigned_resource_id.split(',')[0]
                cip_start_token = self._to_tokens(task.start_time)

                # 1. Get the CIP_id directly from the task object
                cip_id = task.task_id

                # 2. Parse the CIP task name to find the task that comes AFTER it.
                try:
                    following_task_id = task.task_id.split('-for-')[1]
                except IndexError:
                    following_task_id = "N/A"

                # 3. Find the task that comes BEFORE the CIP by checking the resource timeline.
                preceding_task_id = "N/A"
                for booked_start, booked_end, t_id in self.resource_timelines.get(primary_resource, []):
                    if booked_end == cip_start_token:
                        preceding_task_id = t_id
                        break
                
                cip_schedules.append(CIPSchedule(
                    CIP_id=cip_id,
                    resource_id=primary_resource,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    duration_minutes=(task.end_time - task.start_time).total_seconds() / 60,
                    CIP_type="Major",
                    preceding_task_id=preceding_task_id,
                    following_task_id=following_task_id
                ))
            # Handle Production and Finishing tasks
            elif task.task_type in [TaskType.BULK_PRODUCTION, TaskType.FINISHING]:
                # This logic extracts the base order number (e.g., 'ORD_101') from the job_id
                # to allow the Gantt chart to correctly distinguish bulk vs order jobs.
                base_order_no = task.job_id
                if task.task_type == TaskType.FINISHING:
                    base_order_no = task.job_id.split('-cab')[0]

                # The assigned_resource_id can be a list, so we take the first one as primary
                primary_resource = task.assigned_resource_id.strip("[]'\"").split(',')[0]
                
                # Find the priority from the original indent if possible
                try:
                    priority_val = self.indents.get(base_order_no).priority
                except:
                    priority_val = Priority.MEDIUM # Default

                scheduled_tasks.append(TaskSchedule(
                    task_id=task.task_id,
                    order_no=base_order_no,
                    sku_id=task.sku_id,
                    step_id=task.step.step_id,
                    batch_index=int(task.task_id.split('-b')[-1].split('-')[0]), # Heuristic way to get batch index
                    resource_id=primary_resource,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    volume=task.volume_liters,
                    priority=priority_val
                ))

        # Create dummy metrics for compatibility
        dummy_metrics = type('Metrics', (), {
            'schedule_efficiency': 0.95,
            'total_production_volume': sum(t.volume for t in scheduled_tasks),
            'otif_rate': 1.0,
            'solve_time': 0.1 
        })()
            
        return SchedulingResult(
            #is_feasible=True,
            scheduled_tasks=scheduled_tasks,
            CIP_schedules=cip_schedules,
            production_summary={}, # Gantt doesn't use this part
            metrics=dummy_metrics,
            solve_time=0.1,
            status=1,
            objective_value=100.0,
            resource_utilization={}
        )
    
    def write_schedule_log_file(self, file_path: str = "heuristic_schedule_log.txt"):
        """
        Writes a comprehensive log of the entire scheduling run to a file.
        """
        self.log_entries.append(f"Writing full schedule log to {file_path}...")
        with open(file_path, "w") as f:
            f.write('-'*20 + f' Log made at {datetime.now().time()} ' + '-'*20 + '\n')
            f.write("\n" + "="*80)
            f.write("TASK RELATIONSHIP DEBUG")
            f.write("="*80)
            
            jobs = defaultdict(list)
            for task in self.master_task_list:
                jobs[task.job_id].append(task)
            
            for job_id, tasks in jobs.items():
                f.write(f"\n--- JOB: {job_id} ---\n")
                f.write(f"Total tasks: {len(tasks)} \n")
                
                anchor_tasks = [t for t in tasks if t.is_anchor_task]
                f.write(f"Anchor tasks: {len(anchor_tasks)}\n")
                for anchor in anchor_tasks:
                    f.write(f"  -> {anchor.task_id} (priority: {anchor.priority})\n")
                
                # Show task chains
                for task in tasks:
                    prereq_count = len(task.prerequisites)
                    next_task_id = task.next_task.task_id if task.next_task else "None"
                    prev_task_id = task.previous_task.task_id if task.previous_task else "None"
                    
                    f.write(f"  {task.task_id}:\n")
                    f.write(f"    Prerequisites: {prereq_count}\n")
                    f.write(f"    Previous: {prev_task_id}\n")
                    f.write(f"    Next: {next_task_id}\n")
                    f.write(f"    Status: {task.status.name}\n")
                    f.write(f"    Is Anchor: {task.is_anchor_task}\n")
            f.write("="*80 + "\n")
            f.write(" " * 25 + "HEURISTIC SCHEDULER RUN LOG\n")
            f.write("="*80 + "\n")
            
            # Write all logged decisions
            f.write("\n--- Decision Log ---\n")
            for entry in self.log_entries:
                f.write(entry + "\n")

            # --- Final Schedule by Resource section ---
            f.write("\n\n--- Final Schedule by Resource ---\n")
            header = f"{'Resource':<15} | {'Start Time':<16} | {'End Time':<16} | {'S_Token':>7} | {'E_Token':>7} | {'Dur(T)':>6} | {'Rule':<15} | {'Task ID'}\n"
            f.write(header)
            f.write('-' * (len(header) + 5) + '\n')

            tasks_by_id = {t.task_id: t for t in self.master_task_list}

            for resource_id, timeline in sorted(self.resource_timelines.items()):
                f.write(f"--- {resource_id} ---\n")
                last_end_token = 0
                for start_token, end_token, task_id in timeline:
                    idle_tokens = start_token - last_end_token
                    if idle_tokens > 0:
                        f.write(f"{' ':15} | {'...':<16} | {'...':<16} | {'...':>7} | {'...':>7} | {idle_tokens:>6} | {'(IDLE)':<15} |\n")
                    
                    task = tasks_by_id.get(task_id)
                    
                    # --- START: MODIFICATION TO HANDLE NEW TASK TYPES ---
                    rule_name = "UNKNOWN"
                    if task:
                        if task.task_type == TaskType.LOCKED:
                            rule_name = "LOCKED"
                        elif task.task_type == TaskType.CIP:
                            rule_name = "CIP"
                        else:
                            rule_name = getattr(task.step, 'scheduling_rule', SchedulingRule.DEFAULT).name
                    # --- END: MODIFICATION ---

                    start_dt = self._to_datetime(start_token)
                    end_dt = self._to_datetime(end_token)
                    duration_tokens = end_token - start_token
                    
                    f.write(
                        f"{resource_id:<15} | {start_dt.strftime('%m-%d %H:%M'):<16} | {end_dt.strftime('%m-%d %H:%M'):<16} | "
                        f"{start_token:>7} | {end_token:>7} | {duration_tokens:>6} | {rule_name:<15} | {task_id}\n"
                    )
                    last_end_token = end_token
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


if __name__ == "__main__":
    print("Running Heuristic Scheduler in standalone mode...")

    # Load data directly from config files
    loader = DataLoader()
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
    
    # Run the scheduler
    scheduler.run_heuristic_scheduler()
    
    # Write the detailed text log file
    scheduler.write_schedule_log_file()


