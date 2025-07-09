from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import logging
from datetime import datetime, timedelta, time
from enum import Enum
import pandas as pd
from collections import defaultdict
import heapq
import numpy as np
import config
from utils.data_models import *
import math
from collections import defaultdict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeManager:
    """Advanced time management with multiple calendars and constraints"""

    
    def __init__(self, schedule_start: datetime = datetime.combine(datetime.now().date() + timedelta(1), time(14, 0)),
                 working_days: List[int] = None,
                 holidays: List[datetime] = None, schedule_horizon: int = 4): 

        self.schedule_start = schedule_start
        self.shifts = config.SHIFTS
        self.working_days = working_days or [0, 1, 2, 3, 4, 5, 6]  # Default: factory open all days
        self.holidays = {h.date() for h in (holidays or [])} # Convert to set of dates for efficient lookup
        self.schedule_horizon = schedule_horizon
        self.logger = logger
        
        # These methods depend on self.schedule_horizon, so they are called after it's assigned.
        self.shift_starts = self._calculate_shift_start_points()
        self.working_windows = self._calculate_working_windows()
    
    def _calculate_shift_start_points(self) -> List[int]:
        """
        Calculates all valid shift start points in minutes from schedule_start.
        Tasks can only begin at these specific points.
        """
        start_points = set() 
        horizon_day = self.schedule_horizon

        for day_offset in range(horizon_day):
            day = self.schedule_start.date() + timedelta(days=day_offset)

            # Skip non-working days
            if day.weekday() not in self.working_days:
                continue

            # Skip holidays (check if the date itself is a holiday)
            if day in self.holidays: 
                continue

            for shift_id, shift in self.shifts.items():
                if shift.is_active: 
                    # Combine the current 'day' with the shift's 'start_time'
                    shift_start_dt = datetime.combine(day, shift.start_time)

                    # Only consider shift starts that are on or after the overall schedule_start
                    if shift_start_dt >= self.schedule_start:
                        minutes_offset = int((shift_start_dt - self.schedule_start).total_seconds() / 60)
                        start_points.add(minutes_offset)

        return sorted(list(start_points))
    
    def _calculate_working_windows(self) -> List[Tuple[int, int]]:
        """Calculate available working time windows"""
        windows = []
        # Iterate over the same horizon as shift start points to maintain consistency
        horizon_day = self.schedule_horizon
        schedule_end = self.schedule_start + timedelta(days=horizon_day)
        
        for day_offset in range(horizon_day):
            day = self.schedule_start.date() + timedelta(days=day_offset)
            if day.weekday() not in self.working_days:
                continue
            if day in self.holidays:
                continue
            
            # Iterate through each active shift definition
            for shift_id, shift in self.shifts.items():
                if shift.is_active:
                    # Determine the actual start and end datetime for this specific shift on this specific 'day'
                    shift_start_dt = datetime.combine(day, shift.start_time)
                    shift_end_dt = datetime.combine(day, shift.end_time)
                    
                    # Handle shifts that cross midnight (e.g., 22:00-06:00)
                    if shift.end_time <= shift.start_time:  # This indicates the shift wraps around to the next day
                        shift_end_dt += timedelta(days=1)
                    
                    # Ensure the shift window intersects with our schedule period
                    if shift_end_dt <= self.schedule_start or shift_start_dt >= schedule_end:
                        continue
                    
                    # Clip the shift to our schedule boundaries
                    actual_window_start = max(shift_start_dt, self.schedule_start)
                    actual_window_end = min(shift_end_dt, schedule_end)
                    
                    # Calculate minutes offset from schedule_start
                    start_minutes = int((actual_window_start - self.schedule_start).total_seconds() / 60)
                    end_minutes = int((actual_window_end - self.schedule_start).total_seconds() / 60)
                    
                    # Only add the window if it has a positive duration
                    if end_minutes > start_minutes:
                        windows.append((start_minutes, end_minutes))
        
        return sorted(windows)
    # Add this method to your TimeManager class

    def _create_non_working_intervals(self) -> List[Tuple[int, int]]:
        """
        Calculates the inverse of working_windows to find all non-working periods.
        Returns a list of (start_minute, duration_minutes) tuples for non-working times.
        """
        if not self.working_windows:
            # If there are no working windows, the entire horizon is a non-working interval.
            return [(0, self.schedule_horizon * 24 * 60)]

        non_working_intervals = []
        current_time = 0
        
        # self.working_windows is already sorted
        for start_win, end_win in self.working_windows:
            # If there's a gap between the last window's end and this window's start
            if current_time < start_win:
                duration = start_win - current_time
                non_working_intervals.append((current_time, duration))
            
            # Move the current time to the end of the current working window
            current_time = max(current_time, end_win)
        
        # Check for a final gap after the last working window to the end of the horizon
        horizon_minutes = self.schedule_horizon * 24 * 60
        if current_time < horizon_minutes:
            duration = horizon_minutes - current_time
            non_working_intervals.append((current_time, duration))
            
        return non_working_intervals
    
    def is_working_time(self, time_minutes: int) -> bool:
        """Check if given time is within working hours (in minutes from schedule_start)"""
        for start, end in self.working_windows:
            if start <= time_minutes < end:
                return True
        return False

class AdvancedProductionScheduler:
    def __init__(self, schedule_start: datetime = datetime.combine(datetime.today() + timedelta(1), time(14, 0)),
                 schedule_horizon: int = 4,
                 working_days: List[int] = None,
                 holidays: List[datetime] = None):
        
        self.indent = config.USER_INDENTS
        self.skus = config.SKUS
        self.products = config.PRODUCTS
        self.lines = config.LINES
        self.tanks = config.TANKS
        self.rooms = config.ROOMS
        self.shifts = config.SHIFTS
        self.CIP_circuits = config.CIP_CIRCUIT
        self.equipments = config.EQUIPMENTS
        self.schedule_horizon = self._create_schedule_horzion()
        self.time_manager = TimeManager(schedule_start, working_days, holidays, schedule_horizon=schedule_horizon)

        self.all_resource_dicts = [self.lines, self.tanks, self.equipments, self.rooms]

        self.weights = {
            "production_volume": 10.0,
            "overproduction_penalty": 50.0,
            "underproduction_penalty_tiers": [ 1, 50, 100, 500 ],
            "setup_time_penalty": 5.0,
            "CIP_time_penalty": 20.0,
            "room_utilisation_bonus": 10.0,
            "room_underutilisation_penalty": 10.0,
            "tank_utilisation_bonus": 10.0,
            "tank_underutilisation_penalty_tiers": [10, 50, 200],
            "idle_time_penalty": 1000.0, # [NEW] Penalty for stagnation between steps
            "priority_bonus": 25.0,
            "otif_bonus": 30.0,
            "late_minutes_penalty": 100.0,
            "makespan_penalty": 10
        }


        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        self.resource_usage_map = defaultdict(list)
        # Decision variables

        self.batch_qty= {}
        self.produced_quantity = {}
        self.underproduction = {}
        self.overproduction = {}
        self.task_vars ={}
        self.tardiness_vars = {} 
        self.stagnation_vars = [] # [NEW] To hold idle time variables between steps
        self.CIP_vars = {}
        self.cost_vars = {}
        self.CIP_circuit_usage = defaultdict(list)
        self.valid_indents: List[UserIndent] = field(default_factory= List)
        
        # Tracking
        self.solve_start_time = None
        self.warnings = []

        self._is_scheduled = {}
        self.produced_quantity = {}
        
        # Logging
        self.logger = logger

    def _create_schedule_horzion(self) -> int:
        horizon = datetime.now()
        for  indent in self.indent.values():
            if indent.due_date > horizon:
                horizon = indent.due_date
        
        horizon = (horizon - datetime.now()).days
        horizon *= 1440
        return horizon

    def schedule_production(self, time_limit: int = 600, max_iterations: int = 5) -> SchedulingResult:
        """Schedule production using multi-stage optimization approach."""
        self.solver_start_time = datetime.now()
        try:
            # Initialize valid_indents BEFORE validation
            current_time = datetime.now()
            self.valid_indents = self._sort_indent(current_time)

            self.DEFAULT_BATCH_SIZE = 5000  # Fallback size in liters if no constraints are found
            self.LINE_PROD_TIME_MINUTES = 480 
            
            # Now validate setup
            if not self.validate_schedule_setup():
                self.logger.error("Schedule validation failed - cannot proceed")
                return SchedulingResult(
                    status=cp_model.INFEASIBLE,
                    objective_value=0,
                    scheduled_tasks=[],
                    resource_utilization={},
                    production_summary={},
                    solve_time=0,
                    warnings=self.warnings.copy()
                )
            
            # Rest of the method remains the same...
            best_result = None
            best_score = float('-inf')
            
            # Configure solver parameters for scalability
            self.solver.parameters.num_search_workers = 16
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.random_seed = 64
            self.solver.parameters.linearization_level = 2
            
            for iter_num in range(max_iterations):
                self.logger.info("="*50 + f"--- Starting Iteration {iter_num + 1}/{max_iterations} ---" + "="*50)
                print('='*200)
                print(self.time_manager.working_windows)
                self._adjust_parameters_for_iterations(iter=iter_num)
                result = self._solve_iteration(time_limit=(time_limit // max_iterations))
                self.run_task_lifecycle_diagnostics(result)
                if result and result.is_feasible:
                    score = abs(self._calculate_score(result))
                    self.logger.info(f"Iteration {iter_num + 1} finished with score: {score:.2f}")
                    if score > best_score:
                        best_score = score
                        best_result = result
                        self.logger.info(f"*** New best score found: {best_score:.2f} ***")
                else:
                    self.logger.warning(f"Iteration {iter_num + 1} did not yield a feasible solution.")

                self._reset_model()

            if best_result and best_result.is_feasible:
                log_file_name = f"schedule_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                self.generate_schedule_log_file(best_result, file_path=log_file_name)
                
            return best_result or None
        
            

        except Exception as e:
            self.logger.error(f"Error in scheduling: {e}", exc_info=True)
            self.warnings.append(f"Scheduling error: {str(e)}")
            return None
    
    def _sort_indent(self, current_datetime: datetime = None) -> List[UserIndent]:
        """Fixed version with proper parameter handling and explicit filtering"""
        if current_datetime is None:
            current_datetime = datetime.now()
        
        due_date_weight = 10
        priority_weight = 5

        valid_indents = []
        excluded_indents = []
        
        for indent in config.USER_INDENTS.values():
            if indent.due_date >= current_datetime:
                valid_indents.append(indent)
            else:
                excluded_indents.append(indent)
                self.logger.critical(f'Order No.: {indent.order_no} is not scheduled. Due date {indent.due_date} is in the past.')

        if excluded_indents:
            self.logger.info(f'Excluded {len(excluded_indents)} orders with past due dates from scheduling.')
        
        def score(indent: UserIndent) -> int:
            days_until_due = (indent.due_date - current_datetime).days
            priority_score = indent.priority.value
            return due_date_weight * days_until_due - priority_weight * priority_score
        
        return sorted(valid_indents, key=score)

    def validate_schedule_setup(self) -> bool:
        """Validate that the scheduling setup is viable"""
        validation_passed = True
        if not self.time_manager.working_windows:
            self.logger.error("No working time windows available - scheduling impossible")
            self.warnings.append("Critical: No working time windows available")
            validation_passed = False
        if not self.valid_indents:
            self.logger.error("No valid orders to schedule")
            self.warnings.append("Critical: No valid orders available")
            validation_passed = False
        
        return validation_passed

    def _adjust_parameters_for_iterations(self, iter: int):
        """
        Adjusts the weights used in the objective function based on the current optimization iteration.
        """

        if iter == 0:
            # Focus on meeting the target (fulfillment and delivery)
            
            self.weights["production_volume"] = 5000.0
            self.weights["overproduction_penalty"] = 500.0
            self.weights["underproduction_pct_penalty"] = 1000.0
            self.weights["setup_time_penalty"] = 2.0
            self.weights["CIP_time_penalty"] = 5.0
            self.weights["room_utilisation_bonus"] = 5.0
            self.weights["room_underutilisation_penalty"] = 5.0
            self.weights["tank_utilisation_bonus"] = 5.0
            self.weights["tank_underutilisation_penalty_tiers"] = [2, 100, 5000]
            self.weights["idle_time_penalty"] = 5000.0
            self.weights["priority_bonus"] = 100.0
            self.weights["otif_bonus"] = 500.0
            self.weights["late_minutes_penalty"] = 500.0
            self.weights["makes_span_penalty"] = 10

        else:
            # Balanced optimization
            self.weights["production_volume"] = 5000.0
            self.weights["overproduction_penalty"] = 100.0
            self.weights["underproduction_pct_penalty"] = 2000.0
            self.weights["setup_time_penalty"] = 10.0
            self.weights["CIP_time_penalty"] = 60.0
            self.weights["room_utilisation_bonus"] = 1.0
            self.weights["room_underutilisation_penalty"] = 2.0
            self.weights["tank_utilisation_bonus"] = 2.0
            self.weights["tank_underutilisation_penalty_tiers"] = [2, 200, 1000]
            self.weights["idle_time_penalty"] = 10000.0
            self.weights["priority_bonus"] = 100.0
            self.weights["otif_bonus"] = 50.0
            self.weights["late_minutes_penalty"] = 50.0

    def _solve_iteration(self, time_limit: int) -> Optional[SchedulingResult]:
        try:
            if not self.valid_indents:
                self.logger.warning("No valid indents to schedule")
                self.warnings.append("No valid orders found for scheduling")
                return None
            
            # Restore the aggregation logic
            self.aggregated_demand = self._aggregate_demand_by_category()
            if not self.aggregated_demand:
                self.logger.warning("No valid demand to schedule after aggregation.")
                return None
            
            # Loop through aggregated categories to create all tasks
            for product_category, demand_info in self.aggregated_demand.items():
                # Create the shared, upstream bulk production tasks
                self._create_bulk_production_vars(
                    product_category, 
                    demand_info['total_qty']
                )

                # Create the SKU-specific finishing/packaging tasks
                self._create_finishing_vars(
                    demand_info['indents']
                )
            
            # Add all constraints and the objective function
            self._add_constraints()
            self._create_objective()

            self.solver.parameters.max_time_in_seconds = time_limit
            self.solver.parameters.num_search_workers = 8
            self.solver.parameters.log_search_progress = True

            status = self.solver.Solve(self.model)

            self._run_diagnostics()

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return self._extract_enhanced_solution(status)
            else:
                self.logger.warning(f"Solver returned status: {self.solver.StatusName(status)}")
                self.warnings.append(f"Solver failed with status: {self.solver.StatusName(status)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in iteration: {e}", exc_info=True)
            self.warnings.append(f"Iteration error: {str(e)}")
            return None
        
    def _aggregate_demand_by_category(self) -> Dict[str, Dict]:
        """
        [ROBUST] Groups valid indents by product_category, calculates total
        required quantity, and handles potential data errors gracefully.
        """
        aggregated_demand = defaultdict(lambda: {'total_qty': 0, 'indents': []})
        
        for indent in self.valid_indents:
            sku = self.skus.get(indent.sku_id)
    
            if not sku:
                self.logger.warning(
                    f"SKU ID '{indent.sku_id}' from Order No '{indent.order_no}' "
                    f"not found in configuration. Skipping this item."
                )
                self.warnings.append(f"SKU '{indent.sku_id}' not found for order '{indent.order_no}'.")
                continue
            
            product_category = sku.product_category

            if product_category:
                aggregated_demand[product_category]['total_qty'] += indent.qty_required_liters
                aggregated_demand[product_category]['indents'].append(indent)
                
        return aggregated_demand
    
    def _create_bulk_production_vars(self, product_category: str, total_qty: float):
        """
        [ROBUST] Creates all batch and task variables for a single, shared,
        upstream bulk production run for a given product category.
        Handles potential data errors gracefully.
        """
        product = self.products.get(product_category)
        if not product:
            self.logger.error(
                f"Could not create bulk production tasks for category '{product_category}' "
                f"as it was not found in the product configuration. Skipping."
            )
            self.warnings.append(f"Product category '{product_category}' not found.")
            return 
        
        all_steps = product.processing_steps
        
        # This helper creates the batch quantity variables (e.g., two 5000L batches).
        self._create_batching_and_fulfillment_vars(product_category, total_qty, product_category, product_category)
        
        # Identify which steps in the recipe are part of the bulk process.
        first_packaging_idx = next((i for i, s in enumerate(all_steps) if s.process_type == ProcessType.PACKAGING), len(all_steps))
        bulk_steps = all_steps[:first_packaging_idx]

        # --- Start of Improvement ---
        if not bulk_steps:
            self.logger.info(f"Product category '{product_category}' has no defined bulk processing steps (all steps are packaging or later).")
            # This is valid, so we just return without creating tasks.
            return
        else:
            self.logger.info(f"Identified {len(bulk_steps)} bulk steps for '{product_category}'.")
        # --- End of Improvement ---

        # For each potential bulk batch, create a task for each bulk step.
        for batch_index, batch_var in enumerate(self.batch_qty.get(product_category, [])):
            for step in bulk_steps:
                # This helper creates the final task with all its time and resource variables.
                self._create_single_task_var(
                    job_id=product_category,
                    sku_id=product_category, # For bulk, sku_id is the category itself
                    batch_idx=batch_index,
                    step=step,
                    volume_var=batch_var,
                    priority=Priority.MEDIUM # Bulk tasks can have a neutral priority
                )

    def _create_finishing_vars(self, indents_in_group: List[UserIndent]):
        """
        [ROBUST & SCALED] Creates finishing task variables based on the "Quantity-Flow" model.
        Uses scaled integers for all quantity variables to ensure precision.
        """
        QTY_SCALE = 100 

        for indent in indents_in_group:
            order_no = indent.order_no
            
            sku = self.skus.get(indent.sku_id)
            if not sku:
                self.logger.warning(f"SKU '{indent.sku_id}' for order '{order_no}' not found. Skipping.")
                continue
            
            product = self.products.get(sku.product_category)
            if not product:
                self.logger.warning(f"Product Category '{sku.product_category}' for order '{order_no}' not found. Skipping.")
                continue

            product_category = product.product_category
            source_bulk_batches = self.batch_qty.get(product_category, [])
            if not source_bulk_batches:
                self.warnings.append(f"No source bulk batches found for {product_category} to supply order {order_no}.")
                continue
            
            source_batch_size = self._get_batch_size(product_category, product_category)
            if source_batch_size <= 0:
                self.warnings.append(f"Cannot create allocations for {order_no} due to invalid source batch size.")
                continue
            
            max_vol_scaled = int(source_batch_size * QTY_SCALE)

            all_steps = product.processing_steps
            first_packaging_idx = next((i for i, s in enumerate(all_steps) if s.process_type == ProcessType.PACKAGING), -1)
            if first_packaging_idx == -1: continue
            finishing_steps = all_steps[first_packaging_idx:]

            allocation_volumes = []
            for source_batch_idx in range(len(source_bulk_batches)):
                allocation_vol_var = self.model.NewIntVar(0, max_vol_scaled, f"vol_{order_no}_from_bulk_{source_batch_idx}")
                allocation_volumes.append(allocation_vol_var)

                for step in finishing_steps:
                    self._create_single_task_var(
                        job_id=order_no,
                        sku_id=sku.sku_id,
                        batch_idx=source_batch_idx, 
                        step=step,
                        volume_var=allocation_vol_var, # Pass the scaled variable
                        priority=indent.priority
                    )

            if allocation_volumes:
                self._create_finishing_fulfillment_vars(order_no, indent.qty_required_liters, allocation_volumes)

    def _create_single_task_var(self, job_id: str, sku_id: str, batch_idx: int, step: ProcessingStep, volume_var: cp_model.IntVar, priority: Priority):
        """
        [NEW HELPER] Creates and stores a single, complete task variable entry in self.task_vars.
        This is the definitive method for creating any task in the model.
        """
        task_key = (job_id, sku_id, batch_idx, step.step_id)
        if task_key in self.task_vars:
            return # Avoid duplicate task creation

        start_var, end_var = self._create_time_vars(job_id, batch_idx, step.step_id)
        resource_vars, room_vars = self._create_resource_vars(task_key, step, volume_var)

        self._update_resource_usage_map(resource_vars, task_key, start_var, end_var, sku_id, step.step_id, batch_idx)
        self._update_resource_usage_map(room_vars, task_key, start_var, end_var, sku_id, step.step_id, batch_idx)
        
        self.task_vars[task_key] = {
            'start': start_var,
            'end': end_var,
            'volume': volume_var,
            'resources': resource_vars,
            'rooms': room_vars,
            'sku_id': sku_id,
            'step_id': step.step_id,
            'batch_index': batch_idx,
            'base_duration': getattr(step, 'duration_minutes', 60),
            'setup_time': getattr(step, 'setup_time', 0),
            'priority': priority
        }

    def _create_finishing_fulfillment_vars(self, order_no: str, required_qty: float, allocation_volumes: List[cp_model.IntVar]):
        """
        [ROBUST & SCALED] Creates total production and over/under variables for a finishing
        order, using scaled integers to maintain precision.
        """
        # --- Start of Scaling Improvement ---
        QTY_SCALE = 100 # Match the scale used in other functions

        # Immediately scale the required quantity to match the allocation variables
        required_qty_scaled = int(required_qty * QTY_SCALE)
        
        # Calculate a safe upper bound using the scaled value
        total_possible_scaled = required_qty_scaled * 2 
        
        # The total produced quantity for this order is the sum of its scaled allocations
        self.produced_quantity[order_no] = self.model.NewIntVar(0, total_possible_scaled, f"produced_qty_{order_no}")
        self.model.Add(self.produced_quantity[order_no] == sum(allocation_volumes))

        # Standard over/under production variables (now also scaled)
        self.underproduction[order_no] = self.model.NewIntVar(0, required_qty_scaled, f"underproduction_{order_no}")
        self.overproduction[order_no] = self.model.NewIntVar(0, total_possible_scaled, f"overproduction_{order_no}")
        
        # The delta calculation must use the scaled required quantity
        delta = self.produced_quantity[order_no] - required_qty_scaled
        self.model.AddMaxEquality(self.underproduction[order_no], [0, -delta])
        self.model.AddMaxEquality(self.overproduction[order_no], [0, delta])
        
        # This logic remains correct
        self._is_scheduled[order_no] = self.model.NewBoolVar(f'is_scheduled_{order_no}')
        self.model.Add(self.produced_quantity[order_no] > 0).OnlyEnforceIf(self._is_scheduled[order_no])
        self.model.Add(self.produced_quantity[order_no] == 0).OnlyEnforceIf(self._is_scheduled[order_no].Not())
      
    def _create_batching_and_fulfillment_vars(self, job_id: str, required_qty: float, sku_id: str, product_id: str):
        """
        [ROBUST] A generic helper to create batching and fulfillment variables for a job.
        Uses scaled integers for quantity calculations to ensure precision.
        """
        # --- Start of Improvement ---
        QTY_SCALE = 100 # Use a scale to work with integers (e.g., 100.5 L -> 10050)
        
        # Convert all quantity inputs to scaled integers at the beginning
        required_qty_scaled = int(required_qty * QTY_SCALE)

        batch_size = self._get_batch_size(sku_id, product_id)
        if batch_size <= 0:
            self.warnings.append(f"Invalid batch size of {batch_size} for job {job_id}. Using default.")
            batch_size = self.DEFAULT_BATCH_SIZE
        
        batch_size_scaled = int(batch_size * QTY_SCALE)

        if batch_size_scaled <= 0:
            # If batch size is still zero, we can't create batches.
            # This prevents division by zero and logs a clear issue.
            self.logger.error(f"Cannot create batches for job {job_id} due to zero batch size.")
            self.warnings.append(f"Zero batch size for job {job_id}.")
            return

        max_batches = (required_qty_scaled + batch_size_scaled - 1) // batch_size_scaled
        # --- End of Improvement ---

        # Batch-level quantity variables (now in scaled integers)
        batch_vars = [self.model.NewIntVar(0, batch_size_scaled, f"batch_{job_id}_{b}") for b in range(max_batches)]
        self.batch_qty[job_id] = batch_vars

        # Total produced quantity for this job (scaled)
        max_possible_scaled = batch_size_scaled * max_batches
        self.produced_quantity[job_id] = self.model.NewIntVar(0, max_possible_scaled, f"produced_qty_{job_id}")
        self.model.Add(self.produced_quantity[job_id] == sum(batch_vars))

        # Under/overproduction variables (scaled)
        self.underproduction[job_id] = self.model.NewIntVar(0, required_qty_scaled, f"underproduction_{job_id}")
        self.overproduction[job_id] = self.model.NewIntVar(0, max_possible_scaled, f"overproduction_{job_id}")

        delta = self.produced_quantity[job_id] - required_qty_scaled
        self.model.AddMaxEquality(self.underproduction[job_id], [0, -delta])
        self.model.AddMaxEquality(self.overproduction[job_id], [0, delta])
        
        # This logic remains the same
        self._is_scheduled[job_id] = self.model.NewBoolVar(f'is_scheduled_{job_id}')
        self.model.Add(self.produced_quantity[job_id] > 0).OnlyEnforceIf(self._is_scheduled[job_id])
        self.model.Add(self.produced_quantity[job_id] == 0).OnlyEnforceIf(self._is_scheduled[job_id].Not())

#######################################################################################################################################
    def _get_batch_size(self, sku_id: str, product_id: str) -> int:
        """
        Calculates the maximum batch size for a product by finding the most restrictive
        resource constraint across all processing steps.
        
        Returns the minimum capacity that limits production across all steps.
        """
        product = self.products.get(product_id)
        if not product:
            self.logger.error(f"Product ID '{product_id}' not found in configuration.")
        if not product:
            return self.DEFAULT_BATCH_SIZE
        
        # Find the bottleneck capacity across all processing steps
        bottleneck_capacity, bottleneck_info = self._find_bottleneck_capacity(product, sku_id)

# Apply product-specific batch size limit if configured
        if hasattr(product, "batch_size") and product.batch_size > 0 and product.batch_size < bottleneck_capacity:
            bottleneck_capacity = product.batch_size
            bottleneck_info = f"Product batch size limit: {product.batch_size}"
        
        # Return default if no valid constraints found
        if bottleneck_capacity == float('inf'):
            self.logger.info(f"No capacity constraints found for product '{product_id}'. Using default batch size.")
            return self.DEFAULT_BATCH_SIZE
        
        self.logger.info(f"Batch size for product '{product_id}' limited to {int(bottleneck_capacity)} by: {bottleneck_info}")
        return int(bottleneck_capacity)

    def _find_bottleneck_capacity(self, product, sku_id: str) -> tuple[float, str]:
        """Find the most restrictive capacity constraint across all processing steps."""
        min_capacity = float('inf')
        bottleneck_info = "No constraints found"
        
        for step_idx, step in enumerate(product.processing_steps):
            step_capacity, step_info = self._calculate_step_capacity(step, sku_id, step_idx)
            if step_capacity > 0 and step_capacity < min_capacity:
                min_capacity = step_capacity
                bottleneck_info = f"Step {step_idx + 1}: {step_info}"
        
        return min_capacity, bottleneck_info

    def _calculate_step_capacity(self, step: ProcessingStep, sku_id: str, step_idx: int) -> float:
        """Calculate the maximum capacity for a single processing step."""
        step_capacity = float('inf')
        
        for requirement in step.requirements:
            requirement_capacity, bottleneck_info = self._calculate_requirement_capacity(requirement, sku_id)
            if requirement_capacity > 0:
                step_capacity = min(step_capacity, requirement_capacity)
        
        # Convert inf to 0 if no valid requirements found
        step_capacity = float('inf')
        bottleneck_info = "No valid requirements"
        
        for requirement in step.requirements:
            requirement_capacity, req_info = self._calculate_requirement_capacity(requirement, sku_id)
            if requirement_capacity > 0 and requirement_capacity < step_capacity:
                step_capacity = requirement_capacity
                bottleneck_info = req_info
        
        return (0, "No capacity found") if step_capacity == float('inf') else (step_capacity, bottleneck_info)
    
    def _calculate_requirement_capacity(self, requirement: ResourceRequirement, sku_id: str) -> tuple[float, str]:
        if requirement.resource_type == ResourceType.TANK:
            capacity, tank_id = self._get_max_tank_capacity(requirement.compatible_ids)
            return capacity, f"Tank {tank_id} (capacity: {capacity}L)"
        elif requirement.resource_type == ResourceType.LINE:
            capacity, line_id = self._get_max_line_capacity(requirement.compatible_ids, sku_id)
            return capacity, f"Line {line_id} (capacity: {capacity} units)"
        else:
            self.logger.warning(f"Unknown resource type: {requirement.resource_type}")
            return 0, f"Unknown resource type: {requirement.resource_type}"

    def _get_max_tank_capacity(self, tank_ids: list) -> float:
        """Find the maximum capacity among compatible tanks."""
        max_capacity = 0
        best_tank_id = "None"
        
        for tank_id in tank_ids:
            tank = self.tanks.get(tank_id)
            if not tank:
                self.logger.warning(f"Tank ID '{tank_id}' not found")
                continue
                
            if not hasattr(tank, 'capacity_liters') or tank.capacity_liters <= 0:
                self.logger.warning(f"Tank '{tank_id}' has invalid capacity")
                continue
                
            if tank.capacity_liters > max_capacity:
                max_capacity = tank.capacity_liters
                best_tank_id = tank_id

        return max_capacity, best_tank_id

    def _get_max_line_capacity(self, line_ids: list, sku_id: str) -> float:
        """Find the maximum production capacity among compatible lines."""
        max_capacity = 0
        best_line_id = "None"
        
        for line_id in line_ids:
            line_capacity = self._calculate_single_line_capacity(line_id, sku_id)
            if line_capacity > max_capacity:
                max_capacity = line_capacity
                best_line_id = line_id
        
        return max_capacity, best_line_id

    def _calculate_single_line_capacity(self, line_id: str, sku_id: str) -> float:
        """Calculate production capacity for a specific line and SKU."""
        line = self.lines.get(line_id)
        if not line:
            self.logger.warning(f"Line ID '{line_id}' not found")
            return 0
        
        # Validate line has production speed data
        if not hasattr(line, 'compatible_skus_max_production'):
            self.logger.warning(f"Line '{line_id}' missing production speed data")
            return 0
        
        # Check if SKU is compatible with this line
        if sku_id not in line.compatible_skus_max_production:
            return 0  # SKU not compatible, capacity is 0
        
        speed = line.compatible_skus_max_production.get(sku_id)
        if not speed or speed <= 0:
            self.logger.warning(f"Line '{line_id}' has invalid production speed for SKU '{sku_id}'")
            return 0
        
        # Calculate effective production time
        setup_time = getattr(line, 'setup_time_minutes', 0)
        production_time = self.LINE_PROD_TIME_MINUTES - setup_time
        
        if production_time <= 0:
            self.logger.warning(f"Line '{line_id}' has insufficient production time after setup")
            return 0
        
        # Calculate capacity: speed × time × efficiency
        oee = getattr(self, 'TARGET_OEE', 0.85)  # Default OEE if not set
        capacity = speed * production_time * oee
        
        return capacity
#######################################################################################################################################

    def _create_time_vars(self, order_no: str, batch_index: int, step_id: str):
        """Create start and end time variables for a task."""
        start_var = self.model.NewIntVar(
            0, self.schedule_horizon, f'start_{order_no}_{batch_index}_{step_id}'
        )
        end_var = self.model.NewIntVar(
            0, self.schedule_horizon, f'end_{order_no}_{batch_index}_{step_id}'
        )
        return start_var, end_var
   
    def _create_resource_vars(self, task_key: tuple, step: ProcessingStep, volume_var: cp_model.IntVar):
        """
        [REFACTORED & ROBUST] Creates all resource selection variables for a step.
        Receives the volume variable directly to ensure a reliable link to task activity.
        """
        job_id, sku_id, batch_idx, step_id = task_key
        all_resource_vars = {}
        all_room_vars = {}
        
        if not hasattr(step, 'requirements'):
            self.warnings.append(f"Step {step.step_id} is missing 'requirements' attribute.")
            return {}, {}

        # Create a single boolean variable to represent if this task is active.
        # The task is active if its volume is greater than zero.
        is_active = self.model.NewBoolVar(f'is_active_{job_id}_{batch_idx}_{step_id}')
        self.model.Add(volume_var > 0).OnlyEnforceIf(is_active)
        self.model.Add(volume_var == 0).OnlyEnforceIf(is_active.Not())

        for req_idx, requirement in enumerate(step.requirements):
            # Create choice variables for this specific requirement
            choice_vars = {
                res_id: self.model.NewBoolVar(f'use_{requirement.resource_type.name}_{job_id}_{batch_idx}_{step_id}_{req_idx}_{res_id}')
                for res_id in requirement.compatible_ids
            }
            
            # A resource can only be chosen if the parent task is active.
            for var in choice_vars.values():
                self.model.AddImplication(var, is_active)

            # If the task is active, exactly one resource for THIS REQUIREMENT must be chosen.
            if choice_vars:
                self.model.Add(sum(choice_vars.values()) == 1).OnlyEnforceIf(is_active)
                self.model.Add(sum(choice_vars.values()) == 0).OnlyEnforceIf(is_active.Not())

            if requirement.resource_type == ResourceType.ROOM:
                all_room_vars.update(choice_vars)
            else:
                all_resource_vars.update(choice_vars)
                
        return all_resource_vars, all_room_vars

    def _update_resource_usage_map(self, resource_vars: dict, task_key: tuple, 
                                start_var, end_var, sku_id: str, step_id: str, batch_index: int):
        """Update the resource usage mapping for capacity constraints."""
        for res_id, res_var in resource_vars.items():
            self.resource_usage_map[res_id].append({
                'task_key': task_key,
                'start': start_var,
                'end': end_var,
                'assign_var': res_var,
                'sku_id': sku_id,
                'step_id': step_id,
                'batch_index': batch_index,
            })

    def _add_constraints(self):
        """
        [FINAL & POLISHED] Adds the complete and final set of constraints in the
        correct order, with no duplicates, for the "Quantity-Flow" model.
        """
        self.logger.info("Adding FINAL and POLISHED constraint set...")

        # --------------------------------------------------------------------
        # Group 1: Core Task & Environmental Constraints
        # Define the fundamental properties of each task (duration) and how it
        # interacts with the environment (working hours).
        # --------------------------------------------------------------------
        self._add_dynamic_duration_constraints()
        self._add_time_window_constraints() # Depends on durations

        # --------------------------------------------------------------------
        # Group 2: Process Flow & Sequencing Constraints
        # Enforce the rules of the recipe: step order and material flow from
        # bulk to finishing. These are the core "unbreakable" rules.
        # --------------------------------------------------------------------
        self._add_intra_batch_sequencing()
        self._add_quantity_flow_constraints()
        self._add_material_flow_timing_links()

        # --------------------------------------------------------------------
        # Group 3: Resource Contention Constraints
        # Manage competition for shared resources, including production
        # equipment and the CIP circuits themselves.
        # --------------------------------------------------------------------
        self._add_resource_capacity_constraints()
        self._add_CIP_constraints()
        self._add_CIP_circuit_capacity_constraints() # Must be after _add_CIP_constraints

        # --------------------------------------------------------------------
        # Group 4: Fulfillment Metrics & Soft Constraints
        # Define optimization goals and penalties, such as meeting due dates,
        # minimizing idle time, and encouraging a consistent batch order.
        # --------------------------------------------------------------------
        self._add_due_date_constraints()
        self._add_production_quantity_constraints()
        self._add_stagnation_constraints()
        self._add_batch_ordering_constraints()

    def _add_dynamic_duration_constraints(self):
        """
        [FINAL-REVISED] Correctly links task duration to volume, fixing a bug
        that allowed zero-duration tasks when a resource had no defined speed.
        """
        self.logger.info("Adding FINAL dynamic duration constraints...")
        OEE_FACTOR = 0.85
        RATE_SCALE = 100 

        for task_key, task_data in self.task_vars.items():
            _, sku_id, _, _ = task_key
            duration_var = self.model.NewIntVar(0, self.schedule_horizon, f"duration_{task_key}")
            self.model.Add(task_data['end'] == task_data['start'] + duration_var)
            task_data['duration'] = duration_var
            
            volume_var = task_data['volume']
            is_active = self.model.NewBoolVar(f'is_active_check_{task_key}')
            self.model.Add(volume_var > 0).OnlyEnforceIf(is_active)
            self.model.Add(volume_var == 0).OnlyEnforceIf(is_active.Not())
            
            self.model.Add(duration_var == 0).OnlyEnforceIf(is_active.Not())
            
            # --- START OF LOGIC FIX ---
            # We now loop through each possible resource assignment and apply a duration
            # rule specific to that resource.
            
            # Get the task's default duration to use as a fallback.
            base_duration = task_data.get('base_duration', 60) # Default to 60 if not specified
            
            # Track if ANY rule has been applied.
            # A task must have at least one way to determine its duration.
            a_rule_is_possible = False

            for resource_id, assign_var in task_data.get('resources', {}).items():
                a_rule_is_possible = True
                line = self.lines.get(resource_id)
                
                # Check if this specific resource has a dynamic speed rule for this SKU
                speed = None
                if line and hasattr(line, 'compatible_skus_max_production'):
                    speed = line.compatible_skus_max_production.get(sku_id)

                if speed and speed > 0:
                    # Path 1: Apply the dynamic formula if a speed exists for this resource.
                    effective_rate = int(speed * OEE_FACTOR * RATE_SCALE)
                    setup_time = getattr(line, 'setup_time_minutes', 0)
                    self.model.Add(
                        duration_var * effective_rate == volume_var + (setup_time * effective_rate)
                    ).OnlyEnforceIf(assign_var)
                else:
                    # Path 2: Apply the fallback base duration if no speed rule exists for this resource.
                    self.model.Add(duration_var == base_duration).OnlyEnforceIf(assign_var)
            
            # This handles tasks that have no resource assignments (e.g., abstract tasks)
            # or steps that happen in tanks with fixed times (not on 'lines').
            if not a_rule_is_possible:
                self.model.Add(duration_var == base_duration).OnlyEnforceIf(is_active)
            # --- END OF LOGIC FIX ---
            
    def _add_time_window_constraints(self):
        """
        [ROBUST] Ensures tasks are scheduled only within working windows.
        This function now creates the core optional interval variable for each task,
        making it independent of the call order of other constraint functions.
        """
        self.logger.info("Adding robust time window (working hours) constraints...")
        
        non_working_periods = self.time_manager._create_non_working_intervals()
        if not non_working_periods:
            self.logger.info("No non-working periods found; factory is open 24/7.")
            return

        # Create fixed, non-optional interval variables for each non-working period
        non_working_intervals = []
        for i, (start, duration) in enumerate(non_working_periods):
            if duration > 0:
                non_working_intervals.append(
                    self.model.NewIntervalVar(start, duration, start + duration, f'non_working_{i}')
                )

        for task_key, task_data in self.task_vars.items():
            # Check if the core interval variable for this task already exists.
            if 'interval' not in task_data:
                # A task only exists as an interval if it's active (volume > 0).
                is_active = self.model.NewBoolVar(f'is_active_for_interval_{task_key}')
                volume_var = task_data['volume']
                self.model.Add(volume_var > 0).OnlyEnforceIf(is_active)
                self.model.Add(volume_var == 0).OnlyEnforceIf(is_active.Not())
                
                # The duration variable must exist from the previous constraint function.
                duration_var = task_data['duration']

                # Create the main OptionalIntervalVar for this task.
                task_data['interval'] = self.model.NewOptionalIntervalVar(
                    task_data['start'], duration_var, task_data['end'],
                    is_active, f"interval_{task_key}"
                )
            
            # Now we are guaranteed the interval variable exists.
            task_interval = task_data['interval']
            self.model.AddNoOverlap(non_working_intervals + [task_interval])

    def _add_resource_capacity_constraints(self):
        """
        [FINAL] Adds capacity constraints by delegating to the appropriate
        helper based on the resource's configured capacity_type.
        """
        self.logger.info("Adding definitive, data-driven resource capacity constraints...")

        for resource_id, tasks_on_resource in self.resource_usage_map.items():
            if len(tasks_on_resource) <= 1:
                continue

            resource_object = self._get_resource_object(resource_id)
            capacity_type = getattr(resource_object, 'capacity_type', CapacityType.BATCH)

            if capacity_type == CapacityType.SHARED_BY_CATEGORY:
                self._add_shared_by_category_constraints(resource_id, tasks_on_resource)
                continue

            # --- Generic Logic for all other resources ---
            intervals = [self._get_optional_interval_for_task(t, resource_id) for t in tasks_on_resource]
            if capacity_type == CapacityType.CUMULATIVE:
                demands = [self.task_vars[t['task_key']]['demand'] for t in tasks_on_resource]
                capacity = getattr(resource_object, 'capacity', 1)
                self.model.AddCumulative(intervals, demands, capacity)
            else: # BATCH
                self.model.AddNoOverlap(intervals)

    def _add_shared_by_category_constraints(self, resource_id: str, tasks_on_resource: list):
        """
        [DEFINITIVE-FINAL] Models a resource that can only be used by one product
        category at a time using a direct, pairwise No-Overlap constraint between
        tasks of different categories.
        """
        self.logger.info(f"Applying final pairwise No-Overlap logic for {resource_id}.")

        # This direct approach ensures that for any two tasks on this resource,
        # if their categories are different, they cannot overlap.
        for i in range(len(tasks_on_resource)):
            for j in range(i + 1, len(tasks_on_resource)):
                task_i_info = tasks_on_resource[i]
                task_j_info = tasks_on_resource[j]

                category_i = self._get_product_category_for_sku(task_i_info['sku_id'])
                category_j = self._get_product_category_for_sku(task_j_info['sku_id'])

                # Add a No-Overlap constraint ONLY if the categories are different.
                if category_i and category_j and category_i != category_j:
                    
                    interval_i = self._get_optional_interval_for_task(task_i_info, resource_id)
                    interval_j = self._get_optional_interval_for_task(task_j_info, resource_id)
                    
                    # This constraint is direct and unambiguous.
                    self.model.AddNoOverlap([interval_i, interval_j])


    def _get_optional_interval_for_task(self, task_info: dict, resource_id: str) -> cp_model.IntervalVar:
        """Creates a standard optional interval for a task on a specific resource."""
        task_key = task_info['task_key']
        task_data = self.task_vars[task_key]
        return self.model.NewOptionalIntervalVar(
            task_data['start'],
            task_data['duration'],
            task_data['end'],
            task_info['assign_var'], # The boolean for using this specific resource
            f"optional_interval_{resource_id}_{task_key}"
        )

    def _get_resource_object(self, resource_id: str):
        """
        [SCALABLE] A helper to get the resource object from any category by
        iterating through a centralized list of resource dictionaries.
        """
        for resource_dict in self.all_resource_dicts:
            resource_obj = resource_dict.get(resource_id)
            if resource_obj:
                return resource_obj
        return None # Return None if not found in any dictionary
        
    def _add_CIP_circuit_capacity_constraints(self):
        """
        Ensures that each CIP circuit is used by only one resource at a time.
        """
        self.logger.info("Adding CIP circuit capacity constraints...")
        for circuit_id, intervals in self.CIP_circuit_usage.items():
            if len(intervals) > 1:
                self.model.AddNoOverlap(intervals)
                self.logger.info(f"Added No-Overlap constraint for CIP Circuit: {circuit_id} with {len(intervals)} potential tasks.")

    def _get_product_category_for_sku(self, sku_id: str) -> Optional[str]:
        """
        A helper function to find the product category for a given SKU ID.
        This is used to make the correct decision for CIP requirements.
        """
        sku_data = self.skus.get(sku_id)
        if sku_data:
            return sku_data.product_category
        
        # As a fallback for bulk tasks where the sku_id might be the product_category
        if sku_id in self.products:
            return sku_id
            
        self.warnings.append(f"Could not determine product category for SKU/ID: {sku_id}")
        return None

    def _needs_CIP_between(self, prev_task: dict, next_task: dict, resource_id: str) -> bool:
        """
        [CORRECTED] Determines if a CIP is required between two tasks.
        A CIP is needed if the product_category changes, not just the SKU.
        """
        prev_category = self._get_product_category_for_sku(prev_task['sku_id'])
        next_category = self._get_product_category_for_sku(next_task['sku_id'])

        if prev_category in config.SKUS:
            prev_category = self.skus.get(prev_category).product_category
        elif prev_category in config.PRODUCTS:
            prev_category = self.products.get(prev_category).product_category
        else:
            prev_category = prev_category
        
        if next_category in config.SKUS:
            next_category = self.skus.get(next_category).product_category
        elif next_category in config.PRODUCTS:
            next_category = self.products.get(next_category).product_category
        else:
            next_category = next_category

        
        

        if prev_category and not next_category:
            return True
        
        if next_category and not prev_category:
            return False

        if prev_category != next_category:
            return True
        else:
            return False
    
    def _add_CIP_constraints(self):
        """
        [ADVANCED] Creates explicit, optional CIP tasks between tasks of different
        product categories on the same resource. This enables modeling of CIP circuit capacity.
        """
        self.logger.info("Adding advanced 'explicit CIP task' constraints...")

        for resource_id, tasks in self.resource_usage_map.items():
            if len(tasks) < 2:
                continue

            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    t1_info, t2_info = tasks[i], tasks[j]
                    b1, b2 = t1_info['assign_var'], t2_info['assign_var']

                    # Create literals to represent the two possible orderings of the tasks
                    lit_t1_before_t2 = self.model.NewBoolVar(f"lit_{resource_id}_{i}_before_{j}")
                    
                    # Enforce that one must come before the other if both are on this resource
                    self.model.Add(self.task_vars[t1_info['task_key']]['end'] <= self.task_vars[t2_info['task_key']]['start']).OnlyEnforceIf(lit_t1_before_t2)
                    self.model.Add(self.task_vars[t2_info['task_key']]['end'] <= self.task_vars[t1_info['task_key']]['start']).OnlyEnforceIf(lit_t1_before_t2.Not())
                    self.model.AddBoolOr([lit_t1_before_t2, lit_t1_before_t2.Not()]).OnlyEnforceIf([b1, b2])

                    # If CIP is needed, create the optional CIP tasks for both possible orderings
                    if self._needs_CIP_between(t1_info, t2_info, resource_id):
                        # Create the optional CIP task for the t1 -> CIP -> t2 sequence
                        self._create_CIP_task(
                            resource_id=resource_id,
                            prev_task=t1_info,
                            next_task=t2_info,
                            CIP_index=f"{i}_then_{j}",
                            enforcement_literal=lit_t1_before_t2
                        )
                        
                        # Create the optional CIP task for the t2 -> CIP -> t1 sequence
                        self._create_CIP_task(
                            resource_id=resource_id,
                            prev_task=t2_info,
                            next_task=t1_info,
                            CIP_index=f"{j}_then_{i}",
                            enforcement_literal=lit_t1_before_t2.Not()
                        )
                    
    def _create_CIP_task(self, resource_id: str, prev_task: dict, next_task: dict, CIP_index: Union[int, str], enforcement_literal: cp_model.BoolVarT) -> cp_model.IntervalVar:
        """
        Creates a single, conditional CIP task and returns its interval variable.
        """
        CIP_circuit_id = self._get_resource_circuit_id(resource_id)
        if not CIP_circuit_id:
            self.warnings.append(f"Resource {resource_id} requires CIP but has no circuit assigned.")
            return None

        CIP_time = self._get_CIP_time(resource_id)
        CIP_id = f"CIP_{CIP_index}_{resource_id}"

        CIP_start = self.model.NewIntVar(0, self.schedule_horizon, f"{CIP_id}_start")
        CIP_end = self.model.NewIntVar(0, self.schedule_horizon, f"{CIP_id}_end")

        CIP_interval = self.model.NewOptionalIntervalVar(
            CIP_start, CIP_time, CIP_end, enforcement_literal, f"interval_{CIP_id}"
        )

        # This is the crucial step that enables the circuit capacity constraint
        self.CIP_circuit_usage[CIP_circuit_id].append(CIP_interval)
        
        # Add sequencing constraints for the CIP task
        self.model.Add(CIP_end == CIP_start + CIP_time).OnlyEnforceIf(enforcement_literal)
        self.model.Add(CIP_start >= self.task_vars[prev_task['task_key']]['end']).OnlyEnforceIf(enforcement_literal)
        self.model.Add(self.task_vars[next_task['task_key']]['start'] >= CIP_end).OnlyEnforceIf(enforcement_literal)

        self.CIP_vars[CIP_id] = {
            'start': CIP_start, 'end': CIP_end, 'resource': resource_id,
            'interval': CIP_interval, 'enforced_by': enforcement_literal,
            'preceding_task_key': prev_task['task_key'],
            'following_task_key': next_task['task_key']
        }
        return CIP_interval                    
    
    def _get_CIP_time(self, resource_id: str) -> int:
        """Get CIP duration for a resource with standardized naming"""
        resource = config.TANKS.get(resource_id) or config.LINES.get(resource_id) or config.EQUIPMENTS.get(resource_id)
        if resource:
            # First try to get CIP time from resource itself
            cip_time = getattr(resource, 'CIP_duration_minutes', getattr(resource, 'CIP_duration', None))
            if cip_time:
                return cip_time
        
        # OPTIONAL: If resource doesn't have CIP time, get it from the circuit
        circuit_id = self._get_resource_circuit_id(resource_id)
        if circuit_id and circuit_id in self.CIP_circuits:
            circuit = self.CIP_circuits[circuit_id]
            return getattr(circuit, 'CIP_duration_minutes', getattr(circuit, 'CIP_duration', 60))
        
        return 60  # fallback default
    
    def _get_resource_circuit_id(self, resource_id: str) -> Optional[str]:
        """
        Find the CIP circuit ID for a resource by looking through all CIP circuits
        and checking their connected_resource_ids list.
        """
        for circuit_id, circuit in self.CIP_circuits.items():
            if hasattr(circuit, 'connected_resource_ids') and resource_id in circuit.connected_resource_ids:
                return circuit_id
        return None
   
    def _add_production_quantity_constraints(self):
        """
        [REFINED] Add constraints to link individual batch quantities to the
        overall scheduling status of a job.
        """
        # This function's only unique job is to zero-out batch quantities for unscheduled jobs.
        for job_id, batches in self.batch_qty.items():
            if job_id in self._is_scheduled:
                is_scheduled_var = self._is_scheduled[job_id]

                # If a job is not scheduled, the volume of all its batches must be zero.
                for batch_var in batches:
                    self.model.Add(batch_var == 0).OnlyEnforceIf(is_scheduled_var.Not())

    def _add_due_date_constraints(self):
        """
        Add soft constraints for due dates, storing the tardiness variable in its own dictionary.
        """
        for indent in self.valid_indents:
            order_no = indent.order_no
            due_date_minutes = int((indent.due_date - self.time_manager.schedule_start).total_seconds() / 60)
            
            if due_date_minutes > 0:
                completion_time = self._get_order_completion_time(order_no)
                
                if completion_time is not None:
                    is_scheduled_var = self._is_scheduled[order_no]
                    
                    tardiness = self.model.NewIntVar(0, self.schedule_horizon, f"tardiness_{order_no}")
                    
                    self.model.Add(tardiness >= completion_time - due_date_minutes).OnlyEnforceIf(is_scheduled_var)
                    self.model.Add(tardiness >= 0)
                    self.model.Add(tardiness == 0).OnlyEnforceIf(is_scheduled_var.Not())
                    
                    self.tardiness_vars[order_no] = tardiness
    
    def _add_stagnation_constraints(self):
        """
        [CORRECTED & SIMPLIFIED] Adds a soft penalty for idle time (stagnation)
        between consecutive steps of ANY job (both bulk and finishing).
        """
        self.logger.info("Adding inter-step stagnation penalty constraints...")
        
        # Iterate over all tasks to find consecutive steps, same as intra-batch sequencing.
        for task_key, task_data in self.task_vars.items():
            job_id, sku_id, batch_idx, step_id = task_key
            
            # Determine the product recipe for this task
            product_category = self._get_product_category_for_sku(sku_id) if sku_id in self.skus else job_id
            product = self.products.get(product_category)
            if not product or not product.processing_steps:
                continue
            
            steps = product.processing_steps
            
            current_step_index = -1
            for i, step in enumerate(steps):
                if step.step_id == step_id:
                    current_step_index = i
                    break
            
            # If it's not the first step, create a gap variable between it and the previous step.
            if current_step_index > 0:
                prev_step_id = steps[current_step_index - 1].step_id
                prev_task_key = (job_id, sku_id, batch_idx, prev_step_id)
                
                if prev_task_key in self.task_vars:
                    prev_task_end = self.task_vars[prev_task_key]['end']
                    current_task_start = task_data['start']
                    
                    # This function's sole purpose is to define the gap variable for the penalty.
                    # The hard constraint (start >= end) is handled by _add_intra_batch_sequencing.
                    gap_var = self.model.NewIntVar(0, self.schedule_horizon, f"gap_{job_id}_{batch_idx}_{step_id}")
                    self.model.Add(gap_var == current_task_start - prev_task_end)
                    self.stagnation_vars.append(gap_var)

    def _add_batch_ordering_constraints(self):
        """
        [REFINED] Adds a soft penalty for scheduling bulk production batches out of
        numerical order (e.g., Batch 1 before Batch 0).
        """
        self.logger.info("Adding soft batch ordering constraints for bulk production...")

        if "batch_out_of_sequence_penalty" not in self.weights:
            self.weights["batch_out_of_sequence_penalty"] = 10

        out_of_sequence_penalties = []

        # This constraint only applies to bulk jobs, which are defined in self.batch_qty
        for job_id, batches in self.batch_qty.items():
            if len(batches) <= 1:
                continue

            # For bulk jobs, the sku_id is the job_id (i.e., the product category)
            # and the first step is always the first step in the recipe.
            sku_id = job_id
            product = self.products.get(job_id)
            if not product or not product.processing_steps:
                self.warnings.append(f"Could not determine first step for job {job_id} in batch ordering.")
                continue
            first_step_id = product.processing_steps[0].step_id

            # Loop through consecutive batch pairs (e.g., Batch 0 & 1)
            for i in range(len(batches) - 1):
                prev_key = (job_id, sku_id, i, first_step_id)
                next_key = (job_id, sku_id, i + 1, first_step_id)

                if prev_key in self.task_vars and next_key in self.task_vars:
                    start_prev = self.task_vars[prev_key]['start']
                    start_next = self.task_vars[next_key]['start']

                    # Create a boolean variable that is true if batches are out of order
                    is_out_of_sequence = self.model.NewBoolVar(f'is_ooo_{job_id}_b{i+1}')
                    self.model.Add(start_next < start_prev).OnlyEnforceIf(is_out_of_sequence)
                    out_of_sequence_penalties.append(is_out_of_sequence)

        if out_of_sequence_penalties:
            # Add the total penalty to the objective function
            penalty = sum(out_of_sequence_penalties) * int(self.weights["batch_out_of_sequence_penalty"])
            self.cost_vars['out_of_sequence_penalty'] = penalty

    def _add_intra_batch_sequencing(self):
        """
        [REFINED] Enforces that for any single batch, its processing steps
        occur in the correct, linear sequence.
        """
        self.logger.info("Adding intra-batch step sequencing constraints...")

        for task_key, task_data in self.task_vars.items():
            job_id, sku_id, batch_idx, step_id = task_key
            
            product_category = self.skus[sku_id].product_category if sku_id in self.skus else job_id
            
            product = self.products.get(product_category)
            if not product:
                # This can happen if the product category from an order is not in the config.
                # A warning would have already been logged in _create_bulk_production_vars.
                continue
            
            steps = product.processing_steps
            
            current_step_index = -1
            for i, step in enumerate(steps):
                if step.step_id == step_id:
                    current_step_index = i
                    break
            
            if current_step_index > 0:
                prev_step_id = steps[current_step_index - 1].step_id
                prev_task_key = (job_id, sku_id, batch_idx, prev_step_id)
                
                if prev_task_key in self.task_vars:
                    self.model.Add(task_data['start'] >= self.task_vars[prev_task_key]['end'])

    def _add_quantity_flow_constraints(self):
        """
        [CORRECTED] Adds the mass balance constraints for the "Quantity-Flow" model.
        Ensures that the sum of all "allocations" from a bulk batch does not
        exceed the batch's total volume, correcting a double-counting bug.
        """
        self.logger.info("Adding corrected quantity-flow mass balance constraints...")

        self.bulk_batch_consumers = defaultdict(list)
        # Use a set to track which unique allocation volumes have already been added.
        # An allocation is defined by its order, SKU, and the bulk batch it's drawn from.
        processed_allocations = set()

        for task_key, task_data in self.task_vars.items():
            job_id, sku_id, source_batch_idx, step_id = task_key
            
            # Check if it's a finishing task for a valid order
            if job_id in {i.order_no for i in self.valid_indents}:
                
                # Define a unique key for this specific allocation flow
                allocation_key = (job_id, sku_id, source_batch_idx)

                # Only add the volume to our calculation if we haven't already done so
                # for this specific allocation.
                if allocation_key not in processed_allocations:
                    product_category = self.skus[sku_id].product_category
                    self.bulk_batch_consumers[(product_category, source_batch_idx)].append(task_data['volume'])
                    
                    # Mark this allocation as processed to prevent double-counting
                    processed_allocations.add(allocation_key)

        # This part of the function remains the same, but now operates on the correct data
        for (product_category, batch_idx), consumer_vols in self.bulk_batch_consumers.items():
            if product_category in self.batch_qty and batch_idx < len(self.batch_qty[product_category]):
                source_bulk_volume = self.batch_qty[product_category][batch_idx]
                # The core constraint: Sum of volumes drawn <= source bulk volume
                if consumer_vols:
                    self.model.Add(sum(consumer_vols) <= source_bulk_volume)

    def _add_material_flow_timing_links(self):
        """
        [NEW] Creates the time-based link between a finishing task and the COMPLETION
        of its specific source bulk batch.
        """
        self.logger.info("Adding material flow timing links...")

        # For every finishing task...
        for task_key, task_data in self.task_vars.items():
            job_id, sku_id, source_batch_idx, step_id = task_key
            
            # Find finishing tasks by checking if their job_id is a valid order number
            if job_id in {i.order_no for i in self.valid_indents}:
                product_category = self.skus[sku_id].product_category
                
                # Find the final step of the corresponding bulk process
                product = self.products[product_category]
                first_pack_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
                
                if first_pack_idx > 0:
                    final_bulk_step_id = product.processing_steps[first_pack_idx - 1].step_id
                    
                    # Construct the key for the source bulk task
                    source_bulk_task_key = (product_category, product_category, source_batch_idx, final_bulk_step_id)
                    
                    if source_bulk_task_key in self.task_vars:
                        # Constraint: Finishing task must start after the source bulk task is complete
                        self.model.Add(task_data['start'] >= self.task_vars[source_bulk_task_key]['end'])

    def _get_order_completion_time(self, order_no: str) -> Optional[cp_model.IntVar]:
        """
        [CORRECTED & ROBUST] Gets the final completion time for an order by finding the
        maximum end time across all of its constituent finishing tasks.
        """
        try:
            indent = next(i for i in self.valid_indents if i.order_no == order_no)
            sku_id = indent.sku_id
            product = self.products[self.skus[sku_id].product_category]
            all_steps = product.processing_steps

            # Find the very last step in the finishing process for this SKU
            first_packaging_idx = next((i for i, s in enumerate(all_steps) if s.process_type == ProcessType.PACKAGING), -1)
            if first_packaging_idx == -1:
                self.warnings.append(f"Order {order_no} has no packaging steps; cannot determine its completion time.")
                return None
            last_step_id = all_steps[-1].step_id

            # Collect the end times of the final step for ALL possible allocations for this order
            final_step_end_times = [
                task_data['end']
                for task_key, task_data in self.task_vars.items()
                if task_key[0] == order_no and task_key[3] == last_step_id
            ]

            if not final_step_end_times:
                # This can happen if no tasks were created for this order
                return None

            # The order's completion time is the MAXIMUM of all its final step end times.
            order_completion_var = self.model.NewIntVar(0, self.schedule_horizon, f"completion_time_{order_no}")
            self.model.AddMaxEquality(order_completion_var, final_step_end_times)
            
            return order_completion_var

        except StopIteration:
            self.logger.warning(f"Could not find indent for order_no {order_no} in _get_order_completion_time.")
            return None

    def _create_objective(self):
        """
        [CORRECTED & SCALED] Creates a comprehensive objective function that correctly
        handles scaled quantity variables.
        """
        self.logger.info("Creating fulfillment-focused objective function...")
        
        objective_terms = []
        QTY_SCALE = 100 # The quantity scaling factor used throughout the model

        # --- 1. Fulfillment-Driven Value (Bonuses and Penalties for Final Orders) ---
        for indent in self.valid_indents:
            order_no = indent.order_no
            if order_no not in self.produced_quantity:
                continue

            # a. Production Bonus: Proportional to the quantity produced.
            # We apply the weight to the scaled variable to keep the math in the integer domain.
            objective_terms.append(
                self.produced_quantity[order_no] * (int(self.weights["production_volume"]) // QTY_SCALE)
            )
            
            # b. Under/Overproduction Penalties, now correctly scaled.
            self._add_piecewise_underproduction_penalty(order_no, objective_terms)
            # Fixed typo (remove duplicate weight multiplication) and added scaling
            objective_terms.append(
                -self.overproduction[order_no] * (int(self.weights["overproduction_penalty"]) // QTY_SCALE)
            )

            # c. Priority Bonus (remains correct as it's not a quantity).
            priority_multiplier = indent.priority.value
            objective_terms.append(
                self._is_scheduled[order_no] * priority_multiplier * priority_multiplier * int(self.weights["priority_bonus"])
            )
        
        # --- 2. Global Costs and Penalties (time-based, so no scaling needed) ---

        # a. CIP time penalties
        for CIP_data in self.CIP_vars.values():
            CIP_duration = CIP_data['interval'].SizeExpr()
            objective_terms.append(-CIP_duration * int(self.weights["CIP_time_penalty"]))

        # b. Resource utilization (we will review this helper next)
        self._add_resource_utilization_objectives(objective_terms)
        
        # c. Stagnation (idle time between steps) penalty
        for gap_var in self.stagnation_vars:
            objective_terms.append(-gap_var * int(self.weights["idle_time_penalty"]))
        
        # d. Late delivery penalty
        for tardiness_var in self.tardiness_vars.values():
            objective_terms.append(-tardiness_var * int(self.weights["late_minutes_penalty"]))

        # e. Makespan penalty
        makespan_var = self.model.NewIntVar(0, self.schedule_horizon, 'makespan')
        makespan_var = self.model.NewIntVar(0, self.schedule_horizon, 'makespan')
        # FIX: Remove the 'if' condition. The objective will handle it.
        all_end_times = [task['end'] for task in self.task_vars.values()]
        
        if all_end_times:
            self.model.AddMaxEquality(makespan_var, all_end_times)
            objective_terms.append(-makespan_var * int(self.weights["makespan_penalty"]))

        # --- 3. Finalize Objective ---
        if objective_terms:
            self.model.Maximize(sum(objective_terms))
        else:
            self.logger.warning("No objective terms created!")
            
    def _add_piecewise_underproduction_penalty(self, order_no: str, objective_terms: list):
        """
        [CORRECTED & SCALED] Adds a quadratic penalty for the percentage of underproduction.
        Handles scaled quantity variables correctly.
        """
        # The scale for the FRACTION (e.g., 0-1000 for 0-100.0%)
        FRAC_SCALE = 1000
        # The scale for QUANTITIES used throughout the model
        QTY_SCALE = 100
        
        # This underproduction variable is already scaled by QTY_SCALE
        under_var_scaled = self.underproduction[order_no]
        
        indent = next((i for i in self.valid_indents if i.order_no == order_no), None)
        if not indent: return
        
        # Get the base required quantity and scale it consistently
        required_qty = indent.qty_required_liters
        if required_qty <= 0: return
        required_qty_scaled = int(required_qty * QTY_SCALE)

        # under_frac_scaled represents the underproduction percentage (0-1000)
        under_frac_scaled = self.model.NewIntVar(0, FRAC_SCALE, f"{order_no}_under_frac_scaled")
        
        # Define the fraction: under_frac / FRAC_SCALE = under_qty / required_qty
        # This is rearranged for integer math using our consistent scales:
        self.model.Add(under_frac_scaled * required_qty_scaled == under_var_scaled * FRAC_SCALE)
        
        # Create the squared term for a non-linear penalty
        under_frac_squared = self.model.NewIntVar(0, FRAC_SCALE * FRAC_SCALE, f"{order_no}_under_frac_squared")
        self.model.AddMultiplicationEquality(under_frac_squared, [under_frac_scaled, under_frac_scaled])

        penalty_per_pct = int(self.weights.get("underproduction_pct_penalty", 100))
        
        # Apply the penalty to the fraction and its square
        objective_terms.append(-under_frac_scaled * penalty_per_pct)
        objective_terms.append(-under_frac_squared * (penalty_per_pct // FRAC_SCALE))

    def _calculate_resource_utilization(self, resource_id: str) -> cp_model.IntVar:
        """
        [CORRECTED] Calculate total utilization time for a resource by summing the
        durations of tasks conditionally assigned to it.
        """
        if resource_id not in self.resource_usage_map:
            return self.model.NewConstant(0)

        total_time_var = self.model.NewIntVar(0, self.schedule_horizon, f"total_time_{resource_id}")

        assigned_durations = []
        for task_info in self.resource_usage_map[resource_id]:
            task_key = task_info['task_key']
            
            # Use the definitive 'duration' variable for the task.
            task_duration_var = self.task_vars[task_key]['duration']

            # Create a new variable to hold this task's duration IF it's assigned to this resource.
            assigned_duration_var = self.model.NewIntVar(0, self.schedule_horizon, f"assigned_duration_{resource_id}_{task_key}")

            # Add constraints to link the variables, conditional on the assignment boolean.
            self.model.Add(assigned_duration_var == task_duration_var).OnlyEnforceIf(task_info['assign_var'])
            self.model.Add(assigned_duration_var == 0).OnlyEnforceIf(task_info['assign_var'].Not())
            assigned_durations.append(assigned_duration_var)

        # The total used time for the resource is the sum of all the conditionally assigned durations.
        if assigned_durations:
            self.model.Add(total_time_var == sum(assigned_durations))
        else:
            self.model.Add(total_time_var == 0)

        return total_time_var

    def _add_resource_utilization_objectives(self, objective_terms: list):
        """
        [REFINED] Adds resource under-utilisation penalties using a more robust,
        data-driven method for retrieving penalty weights.
        """
        all_resources = list(config.ROOMS.keys()) + list(config.EQUIPMENTS.keys()) + list(config.LINES.keys()) + list(config.TANKS.keys())

        for resource_id in all_resources:
            total_used_time = self._calculate_resource_utilization(resource_id)
            
            available_time = self._get_available_time_for_resource(resource_id)
            if available_time > 0:
                underutil_var = self.model.NewIntVar(0, available_time, f"underutil_{resource_id}")
                # Define under-utilization as the gap between available and used time.
                self.model.Add(underutil_var >= available_time - total_used_time)
                
                # Get the appropriate penalty weight using our new helper function
                penalty_weight = self._get_underutilisation_penalty_weight(resource_id)
                
                if penalty_weight > 0:
                    objective_terms.append(-underutil_var * penalty_weight)
    
    def _get_underutilisation_penalty_weight(self, resource_id: str) -> int:
        """
        [NEW HELPER] Gets the appropriate under-utilisation penalty weight for a
        given resource, looking for specific, type-based, or default weights.
        """
        resource_type_str = self._get_resource_type(resource_id).name.lower()
        
        # 1. Look for a penalty for the specific resource ID (e.g., "line-1_penalty")
        specific_weight = self.weights.get(f"{resource_id}_underutilisation_penalty")
        if specific_weight is not None:
            return int(specific_weight)
            
        # 2. Look for a penalty for the resource type (e.g., "line_underutilisation_penalty")
        type_weight = self.weights.get(f"{resource_type_str}_underutilisation_penalty")
        if type_weight is not None:
            return int(type_weight)
            
        # 3. Fall back to a default penalty
        return int(self.weights.get("default_underutilisation_penalty", 10))

    def _get_resource_type(self, resource_id: str) -> Optional[ResourceType]:
        """
        Determines the ResourceType of a resource by checking which category
        dictionary it belongs to.
        """
        if resource_id in self.lines:
            return ResourceType.LINE
        if resource_id in self.tanks:
            return ResourceType.TANK
        if resource_id in self.equipments:
            return ResourceType.EQUIPMENT
        if resource_id in self.rooms:
            return ResourceType.ROOM
        
        # Fallback if the resource ID is not found in any category
        self.logger.warning(f"Could not determine the type for resource ID '{resource_id}'.")
        return None

    def _get_available_time_for_resource(self, resource_id: str) -> int:
        """Get total available time for a resource within the schedule horizon"""
        total_available = 0
        for start_time, end_time in self.time_manager.working_windows:
            total_available += (end_time - start_time)
        return total_available
    
    def _extract_enhanced_solution(self, status) -> SchedulingResult:
        """
        [ENHANCED] Extract a comprehensive and informative solution from the solved model,
        un-scaling all quantities back to real-world units.
        """
        self.logger.info("Extracting solution...")
        QTY_SCALE = 100 # The quantity scaling factor used throughout the model

        scheduled_tasks = []
        resource_utilization = defaultdict(list)
        production_summary = {}
        CIP_schedules = []

        # Extract task schedules
        for task_key, task_data in self.task_vars.items():
            volume_scaled = self.solver.Value(task_data['volume'])
            if volume_scaled == 0:
                continue

            start_time = self.solver.Value(task_data['start'])
            end_time = self.solver.Value(task_data['end'])
            
            assigned_resource = None
            all_possible_resources = {**task_data['resources'], **task_data['rooms']}
            for resource_id, resource_var in all_possible_resources.items():
                if self.solver.BooleanValue(resource_var):
                    assigned_resource = resource_id
                    break
            
            if assigned_resource:
                task_schedule = TaskSchedule(
                    task_id=f"{task_key[0]}_{task_key[2]}_{task_key[3]}",
                    order_no=task_key[0], sku_id=task_key[1], batch_index=task_key[2], step_id=task_key[3],
                    start_time=self.time_manager.schedule_start + timedelta(minutes=start_time),
                    end_time=self.time_manager.schedule_start + timedelta(minutes=end_time),
                    resource_id=assigned_resource,
                    # MODIFIED: Un-scale the volume for the final report
                    volume=volume_scaled / QTY_SCALE,
                    priority=task_data['priority']
                )
                scheduled_tasks.append(task_schedule)
                resource_utilization[assigned_resource].append({'start': start_time, 'end': end_time, 'task_id': task_schedule.task_id})
        
        # Extract CIP schedules
        for CIP_id, CIP_data in self.CIP_vars.items():
            if self.solver.BooleanValue(CIP_data['enforced_by']):
                start = self.solver.Value(CIP_data['start'])
                end = self.solver.Value(CIP_data['end'])

                # MODIFIED: Get predecessor and successor task keys
                pred_key = CIP_data.get('preceding_task_key')
                succ_key = CIP_data.get('following_task_key')

                CIP_schedules.append(CIPSchedule(
                    CIP_id=CIP_id,
                    resource_id=CIP_data['resource'],
                    start_time=self.time_manager.schedule_start + timedelta(minutes=start),
                    end_time=self.time_manager.schedule_start + timedelta(minutes=end),
                    duration_minutes=int((end - start)),
                    # MODIFIED: Generate readable IDs for the linked tasks
                    preceding_task_id=f"{pred_key[0]}_{pred_key[2]}_{pred_key[3]}" if pred_key else 'unknown',
                    following_task_id=f"{succ_key[0]}_{succ_key[2]}_{succ_key[3]}" if succ_key else 'unknown'
                ))

        # Extract production summary
        for order_no, produced_var in self.produced_quantity.items():
            produced_qty_scaled = self.solver.Value(produced_var)
            if produced_qty_scaled > 0:
                # MODIFIED: Find original indent to add more context
                indent = next((i for i in self.valid_indents if i.order_no == order_no), None)
                required_qty = indent.qty_required_liters if indent else 0
                
                produced_qty = produced_qty_scaled / QTY_SCALE
                under_qty = self.solver.Value(self.underproduction[order_no]) / QTY_SCALE
                over_qty = self.solver.Value(self.overproduction[order_no]) / QTY_SCALE

                production_summary[order_no] = {
                    'required_quantity': required_qty, # NEW: Added for context
                    'produced_quantity': produced_qty, # Now un-scaled
                    'underproduction': under_qty,      # Now un-scaled
                    'overproduction': over_qty,        # Now un-scaled
                    'fulfillment_percentage': (produced_qty / required_qty * 100) if required_qty > 0 else 100.0, # NEW
                    'scheduled': True
                }
        
        total_solve_time = (datetime.now() - self.solver_start_time).total_seconds()
        objective_value = self.solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0.0
        
        return SchedulingResult(
            status=status,
            objective_value=objective_value,
            scheduled_tasks=scheduled_tasks,
            resource_utilization=dict(resource_utilization),
            production_summary=production_summary,
            solve_time=total_solve_time,
            warnings=self.warnings.copy(),
            CIP_schedules=CIP_schedules,
        )
    
    def _calculate_score(self, result: SchedulingResult) -> float:
        """
        [CORRECTED & ENHANCED] Calculate a comprehensive score for the solution that
        reflects business value like total fulfillment and overproduction costs.
        """
        if not result or not result.is_feasible or not result.scheduled_tasks:
            return float('-inf')

        # --- Define weights for different components of the score ---
        # These can be tuned to reflect business priorities.
        objective_weight = 1.0
        total_volume_bonus = 10.0  # Bonus per liter produced for a valid order
        overproduction_penalty = 50.0 # Penalty per liter of overproduction
        warnings_penalty = 500.0   # Penalty per warning generated
        utilization_bonus = 10000.0 # Bonus for the overall utilization rate

        # --- Calculate Score Components ---
        score = 0.0

        # 1. Start with the solver's own objective value
        score += result.objective_value * objective_weight

        # 2. Add bonuses and penalties from the production summary
        total_produced_volume = 0
        total_overproduction = 0
        if result.production_summary:
            for order_no, summary in result.production_summary.items():
                total_produced_volume += summary.get('produced_quantity', 0)
                total_overproduction += summary.get('overproduction', 0)
            
            score += total_produced_volume * total_volume_bonus
            score -= total_overproduction * overproduction_penalty

        # 3. Penalize warnings
        score -= len(result.warnings) * warnings_penalty

        # 4. Add bonus for resource utilization
        total_utilization_rate = 0
        if result.resource_utilization:
            # Use the master list of all resources for a stable denominator
            num_resources = (len(self.lines) + len(self.tanks) +
                            len(self.equipments) + len(self.rooms))

            for resource_id, tasks in result.resource_utilization.items():
                if tasks:
                    total_time = sum(task['end'] - task['start'] for task in tasks)
                    available_time = self._get_available_time_for_resource(resource_id)
                    if available_time > 0:
                        total_utilization_rate += total_time / available_time
            
            avg_utilization = total_utilization_rate / num_resources if num_resources > 0 else 0
            score += avg_utilization * utilization_bonus
        
        # --- Return the final, comprehensive score ---
        return score

    def _reset_model(self):
            """
            [MODIFIED] Reset the model for the next iteration.
            Adds the new stagnation_vars list to the reset list.
            """
            self.model = cp_model.CpModel()
            self.resource_usage_map = defaultdict(list)
            self.task_vars = {}
            self.tardiness_vars = {} 
            self.stagnation_vars = [] # [FIX] Reset stagnation vars
            self.batch_qty = {}
            self.produced_quantity = {}
            self.underproduction = {}
            self.overproduction = {}
            self._is_scheduled = {}
            self.CIP_vars = {}
            self.cost_vars = {}
            self.CIP_circuit_usage = defaultdict(list)
            self.warnings = []
    
    def generate_schedule_log_file(self, result: SchedulingResult, file_path: str = "schedule_log.txt"):
        """
        [ENHANCED] Generates a more detailed and readable text file log of the
        final production schedule, including warnings, a clearer timeline, and
        idle time calculations.
        """
        self.logger.info(f"Generating detailed schedule log file at: {file_path}")

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except Exception as e:
            self.logger.error(f"Could not create directory for log file: {e}")

        with open(file_path, "w") as f:
            f.write("="*80 + "\n")
            f.write(" " * 25 + "FINAL PRODUCTION SCHEDULE LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Schedule Start Time: {self.time_manager.schedule_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Solver Status: {self.solver.StatusName(result.status)}\n")
            f.write(f"Objective Value: {result.objective_value:.2f}\n")
            # NEW: Display the calculated business score
            f.write(f"Calculated Score: {self._calculate_score(result):.2f}\n")
            f.write("\n")

            # NEW: Display any warnings generated during the run
            if result.warnings:
                f.write("--- WARNINGS ---\n")
                for warning in result.warnings:
                    f.write(f"- {warning}\n")
                f.write("\n")

            if not result.is_feasible or not result.scheduled_tasks:
                f.write("No feasible schedule was generated.\n")
                return

            # --- 1. Chronological Full-Factory Schedule ---
            f.write("="*80 + "\n")
            f.write(" CHRONOLOGICAL FULL-FACTORY SCHEDULE\n".upper())
            f.write("="*80 + "\n\n")

            all_events = result.scheduled_tasks + result.CIP_schedules
            all_events.sort(key=lambda x: x.start_time)

            for event in all_events:
                start_str = event.start_time.strftime('%m-%d %H:%M')
                end_str = event.end_time.strftime('%m-%d %H:%M')
                duration = int((event.end_time - event.start_time).total_seconds() / 60)

                if isinstance(event, TaskSchedule):
                    event_type = "TASK"
                    details = f"{event.order_no:<12} | SKU: {event.sku_id:<8} | Batch: {event.batch_index:<2} | Step: {event.step_id}"
                else: # CIPSchedule
                    event_type = "CLEANING (CIP)"
                    # NEW: Show what the CIP is for
                    details = f"For tasks '{event.preceding_task_id}' -> '{event.following_task_id}'"

                f.write(f"[{start_str} -> {end_str}] ({duration: >3} min) | {event_type:<15} | Resource: {event.resource_id:<15} | {details}\n")

            # --- 2. Schedule by Resource ---
            f.write("\n\n" + "="*80 + "\n")
            f.write(" SCHEDULE BY RESOURCE\n".upper())
            f.write("="*80 + "\n\n")

            resource_schedule = defaultdict(list)
            for event in all_events:
                resource_schedule[event.resource_id].append(event)
            
            sorted_resources = sorted(resource_schedule.keys())

            for resource_id in sorted_resources:
                f.write(f"--- Resource: {resource_id} ---\n")
                sorted_events = sorted(resource_schedule[resource_id], key=lambda x: x.start_time)
                
                last_event_end = None
                total_busy_minutes = 0
                for event in sorted_events:
                    # NEW: Calculate and display idle time between tasks
                    if last_event_end and event.start_time > last_event_end:
                        idle_minutes = int((event.start_time - last_event_end).total_seconds() / 60)
                        f.write(f"  ... (IDLE for {idle_minutes} min) ...\n")

                    start_str = event.start_time.strftime('%m-%d %H:%M')
                    end_str = event.end_time.strftime('%m-%d %H:%M')
                    duration = int((event.end_time - event.start_time).total_seconds() / 60)
                    total_busy_minutes += duration
                    
                      # FIX: Use an explicit check instead of the flawed getattr
                    if isinstance(event, TaskSchedule):
                        details = f"Task: {event.task_id}"
                    else:
                        details = f"Task: {event.CIP_id}"

                    f.write(f"  [{start_str} -> {end_str}] ({duration: >3} min) | {details}\n")
                    last_event_end = event.end_time

                available_time = self._get_available_time_for_resource(resource_id)
                utilization_pct = (total_busy_minutes / available_time * 100) if available_time > 0 else 0
                f.write(f"  Resource Utilization: {utilization_pct:.1f}%\n\n")

            # --- 3. Production Fulfillment Summary ---
            f.write("\n" + "="*80 + "\n")
            f.write(" PRODUCTION FULFILLMENT SUMMARY\n".upper())
            f.write("="*80 + "\n\n")

            for indent in self.valid_indents:
                summary = result.production_summary.get(indent.order_no)
                if summary:
                    produced = summary.get('produced_quantity', 0)
                    required = summary.get('required_quantity', indent.qty_required_liters)
                    fulfillment_pct = summary.get('fulfillment_percentage', 0)
                    status = f"{fulfillment_pct:.1f}% Fulfilled"
                    if summary.get('underproduction', 0) > 0.1: status += f" (UNDER by {summary['underproduction']:.1f}L)"
                    if summary.get('overproduction', 0) > 0.1: status += f" (OVER by {summary['overproduction']:.1f}L)"

                    f.write(f"Order: {indent.order_no:<10} | Required: {required:<7.1f}L | Produced: {produced:<7.1f}L | Status: {status}\n")
                else:
                    f.write(f"Order: {indent.order_no:<10} | Required: {indent.qty_required_liters:<7.1f}L | Produced: 0.0L | Status: NOT PRODUCED\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")


    def _run_diagnostics(self, categories_to_debug: Optional[List[str]] = None):
        """
        [ENHANCED-V3] A comprehensive diagnostic tool to print the logical structure
        of the model, with more intelligent parallelism checks.
        """
        print("\n" + "#"*80)
        print("### RUNNING ADVANCED SCHEDULER DIAGNOSTICS ###")
        print("#"*80 + "\n")

        # --- 1. Diagnose SHARED_BY_CATEGORY Resources (The Heater Logic) ---
        print("="*60)
        print("--- 1. Diagnosing SHARED_BY_CATEGORY Resources (e.g., Curd Heater) ---")
        print("="*60)
        all_resources = self._get_all_resources()
        shared_resources = [
            res_id for res_id, res_obj in all_resources.items()
            if getattr(res_obj, 'capacity_type', CapacityType.BATCH) == CapacityType.SHARED_BY_CATEGORY
        ]

        if not shared_resources:
            print("  [INFO] No resources with 'SHARED_BY_CATEGORY' capacity type found.")
        else:
            for resource_id in shared_resources:
                print(f"\n  * Analyzing '{resource_id}':")
                tasks_by_category = defaultdict(list)
                tasks_on_resource = self.resource_usage_map.get(resource_id, [])
                for task_info in tasks_on_resource:
                    category = self._get_product_category_for_sku(task_info['sku_id'])
                    if category:
                        tasks_by_category[category].append(task_info)

                if not tasks_by_category:
                    print("    - No tasks are assigned to this resource.")
                    continue

                # a. Inter-Category Blocking Check
                print("\n    a. Inter-Category Blocking Check (between different categories):")
                category_keys = list(tasks_by_category.keys())
                if len(category_keys) > 1:
                    for i in range(len(category_keys)):
                        for j in range(i + 1, len(category_keys)):
                            cat1 = category_keys[i]
                            cat2 = category_keys[j]
                            print(f"      - EXPECTED: Tasks from '{cat1}' must NOT overlap with tasks from '{cat2}' on this resource.")
                else:
                    print("      - Only one category uses this resource, so no inter-category blocking is needed.")

                # b. Intra-Category Parallelism Check (IMPROVED)
                print("\n    b. Intra-Category Parallelism Check (within the same category):")
                for category, tasks in tasks_by_category.items():
                    print(f"      --- Analyzing Category: {category} ---")
                    if len(tasks) < 2:
                        print("        - Not enough tasks to check for parallelism.")
                        continue
                    
                    # Check every pair of tasks within the same category
                    for i in range(len(tasks)):
                        for j in range(i + 1, len(tasks)):
                            task_i_info = tasks[i]
                            task_j_info = tasks[j]
                            
                            task_i_resources = set(self.task_vars[task_i_info['task_key']]['resources'].keys())
                            task_j_resources = set(self.task_vars[task_j_info['task_key']]['resources'].keys())
                            
                            # Find any exclusive (BATCH) resources they have in common
                            common_batch_resources = {
                                res for res in task_i_resources.intersection(task_j_resources)
                                if getattr(all_resources.get(res), 'capacity_type', CapacityType.BATCH) == CapacityType.BATCH
                            }
                            
                            task_i_name = task_i_info['task_key']
                            task_j_name = task_j_info['task_key']
                            
                            if common_batch_resources:
                                print(f"        - Tasks {task_i_name} and {task_j_name}:")
                                print(f"          - BLOCKED from running in parallel because they share exclusive resource(s): {', '.join(common_batch_resources)}")
                            else:
                                print(f"        - Tasks {task_i_name} and {task_j_name}:")
                                print(f"          - ALLOWED to run in parallel (from this resource's perspective).")

        # --- 2. Diagnose Material Flow for each Product Category ---
        print("\n" + "="*60)
        print("--- 2. Diagnosing Material & Process Flow per Product Category ---")
        print("="*60)
        
        product_keys = categories_to_debug or list(self.aggregated_demand.keys())

        for category_name in product_keys:
            print(f"\n------------------ CATEGORY: {category_name} ------------------")
            product = self.products.get(category_name)
            if not product: continue

            # a. Check Bulk Production
            print("  a. Bulk Production Check:")
            bulk_batches = self.batch_qty.get(category_name)
            if not bulk_batches:
                print(f"    [!!] WARNING: No bulk batches were created for '{category_name}'.")
                continue
            print(f"    - Found {len(bulk_batches)} bulk batches for this category.")

            # b. Check Mass Balance
            print("\n  b. Mass Balance Check (Finishing Tasks drawing from Bulk):")
            found_consumers = False
            for i, bulk_batch_var in enumerate(bulk_batches):
                # Find all finishing tasks that draw from this specific bulk batch
                consuming_allocations = [
                    task_data['volume'] for task_key, task_data in self.task_vars.items()
                    if task_key[0] in {indent.order_no for indent in self.valid_indents}
                    and self._get_product_category_for_sku(task_key[1]) == category_name
                    and task_key[2] == i
                ]
                unique_consuming_allocations = list(set(consuming_allocations))

                if unique_consuming_allocations:
                    found_consumers = True
                    consumer_names = [v.Name() for v in unique_consuming_allocations]
                    print(f"    - For Bulk Batch {i} ({bulk_batch_var.Name()}):")
                    print(f"      - EXPECTED CONSTRAINT: sum({consumer_names}) <= {bulk_batch_var.Name()}")

            if not found_consumers:
                print("    - [!!] WARNING: No finishing tasks were found drawing from this category.")

            # c. Check Intra-Batch Sequencing
            print("\n  c. Intra-Batch Sequence Check:")
            if product.processing_steps:
                first_bulk_task_key = (category_name, category_name, 0, product.processing_steps[0].step_id)
                if first_bulk_task_key in self.task_vars and len(product.processing_steps) > 1:
                    second_step_id = product.processing_steps[1].step_id
                    print(f"    - EXPECTED CONSTRAINT: Start of '{second_step_id}' >= End of '{first_bulk_task_key[3]}'")
                else:
                    print("    - Not enough steps to check sequence or no tasks found.")
            else:
                print("    - No processing steps found for this product.")


        print("\n" + "#"*80)
        print("### DIAGNOSTICS COMPLETE ###")
        print("#"*80 + "\n")

    # REMINDER: This diagnostic function requires the _get_all_resources() helper to exist in your class.
    def _get_all_resources(self) -> Dict[str, Any]:
        """A helper to get a single dictionary of all configured resources."""
        all_res = {}
        all_res.update(self.lines)
        all_res.update(self.tanks)
        all_res.update(self.equipments)
        all_res.update(self.rooms)
        return all_res
    
    def run_task_lifecycle_diagnostics(self, result: SchedulingResult):
        """
        [NEW DIAGNOSTIC] A post-mortem analysis tool that reports on the status of
        every potential task in the model to understand why certain tasks were or
        were not scheduled.
        """
        print("\n" + "#"*80)
        print("### RUNNING TASK LIFECYCLE DIAGNOSTICS ###")
        print("#"*80 + "\n")

        # --- Step 1: Get a set of all tasks that actually ran ---
        ran_task_keys = set()
        for task in result.scheduled_tasks:
            # Reconstruct the original task_key from the scheduled task info
            key = (task.order_no, task.sku_id, task.batch_index, task.step_id)
            ran_task_keys.add(key)

        # --- Step 2: Analyze all potential jobs (Bulk and Finishing) ---
        all_jobs = list(self.aggregated_demand.keys()) # Bulk jobs
        all_jobs += [indent.order_no for indent in self.valid_indents] # Finishing jobs

        for job_id in sorted(list(set(all_jobs))):
            print(f"\n--- Analyzing Job: '{job_id}' ---")

            # Determine the steps for this job
            steps = []
            is_bulk_job = job_id in self.products
            
            if is_bulk_job:
                product = self.products[job_id]
                first_pack_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), len(product.processing_steps))
                steps = product.processing_steps[:first_pack_idx]
                sku_id_for_job = job_id
            else: # It's a finishing job
                try:
                    indent = next(i for i in self.valid_indents if i.order_no == job_id)
                    sku_id_for_job = indent.sku_id
                    product = self.products[self.skus[sku_id_for_job].product_category]
                    first_pack_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
                    if first_pack_idx != -1:
                        steps = product.processing_steps[first_pack_idx:]
                except (StopIteration, KeyError):
                    print(f"  [ERROR] Could not find configuration for order '{job_id}'.")
                    continue

            if not steps:
                print("  - No applicable steps found for this job.")
                continue
                
            # Determine the number of potential batches/allocations
            num_batches = len(self.batch_qty.get(job_id, []))
            if is_bulk_job and num_batches == 0:
                # This means the batch size calculation failed for the bulk product.
                num_batches = len(self.batch_qty.get(self.products[job_id].product_category, []))

            # Check each potential task for this job
            for batch_idx in range(num_batches):
                for step in steps:
                    task_key = (job_id, sku_id_for_job, batch_idx, step.step_id)
                    
                    # Check 1: Was the task created?
                    if task_key not in self.task_vars:
                        print(f"  - Task {task_key}:")
                        print("    - STATUS: NOT CREATED")
                        print("    - REASON: The task variable was never generated. This often happens if `_create_resource_vars` fails to find any compatible resources after filtering.")
                        continue
                    
                    # Check 2: Did the task run?
                    if task_key in ran_task_keys:
                        print(f"  - Task {task_key}:")
                        print("    - STATUS: RAN SUCCESSFULLY")
                    else:
                        # Check 3: If not, why not?
                        print(f"  - Task {task_key}:")
                        print("    - STATUS: CREATED BUT NOT RUN")
                        
                        # Infer the reason
                        is_scheduled_var = self._is_scheduled.get(job_id)
                        if is_scheduled_var and not self.solver.BooleanValue(is_scheduled_var):
                            print("    - REASON: The entire job/order was deemed too 'expensive' by the solver and was not scheduled.")
                            print("              (Likely due to low underproduction penalties).")
                        else:
                            volume_var = self.task_vars[task_key]['volume']
                            if self.solver.Value(volume_var) == 0:
                                if is_bulk_job:
                                    print("    - REASON: The solver decided the volume for this specific bulk batch should be zero.")
                                else: # Finishing task
                                    print("    - REASON: The solver decided the allocation from this specific bulk batch should be zero.")
                            else:
                                print("    - REASON: Unknown. The task was created and had volume, but did not appear in the final schedule.")
                                print("              This could indicate a complex resource conflict or a bug in the solution extraction.")

        print("\n" + "#"*80)
        print("### TASK LIFECYCLE DIAGNOSTICS COMPLETE ###")
        print("#"*80 + "\n")

# To make the new diagnostic function work, you'll need to make one small change.
# In `_add_quantity_flow_constraints`, change the first line from:
# bulk_batch_consumers = defaultdict(list)
# to:
# self.bulk_batch_consumers = defaultdict(list)
# and use self.bulk_batch_consumers throughout that function.