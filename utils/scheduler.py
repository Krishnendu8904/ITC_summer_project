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
            self.weights["underproduction_pct_penalty"] = 10000.0
            self.weights["setup_time_penalty"] = 2.0
            self.weights["CIP_time_penalty"] = 5.0
            self.weights["room_utilisation_bonus"] = 5.0
            self.weights["room_underutilisation_penalty"] = 5.0
            self.weights["tank_utilisation_bonus"] = 5.0
            self.weights["tank_underutilisation_penalty_tiers"] = [20, 100, 150]
            self.weights["idle_time_penalty"] = 5000.0
            self.weights["priority_bonus"] = 100.0
            self.weights["otif_bonus"] = 500.0
            self.weights["late_minutes_penalty"] = 500.0
            self.weights["makes_span_penalty"] = 10

        else:
            # Balanced optimization
            self.weights["production_volume"] = 5000.0
            self.weights["overproduction_penalty"] = 100.0
            self.weights["underproduction_pct_penalty"] = 100000.0
            self.weights["setup_time_penalty"] = 1.0
            self.weights["CIP_time_penalty"] = 6.0
            self.weights["room_utilisation_bonus"] = 1000.0
            self.weights["room_underutilisation_penalty"] = 2.0
            self.weights["tank_utilisation_bonus"] = 200.0
            self.weights["tank_underutilisation_penalty_tiers"] = [20, 200, 1000]
            self.weights["idle_time_penalty"] = 10000.0
            self.weights["priority_bonus"] = 100.0
            self.weights["otif_bonus"] = 500.0
            self.weights["late_minutes_penalty"] = 500.0

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
        Groups valid indents by product_category and calculates total required quantity.
        """
        aggregated_demand = defaultdict(lambda: {'total_qty': 0, 'indents': []})
        
        for indent in self.valid_indents:
            product_category = self.skus.get(indent.sku_id).product_category
            if product_category:
                aggregated_demand[product_category]['total_qty'] += indent.qty_required_liters
                aggregated_demand[product_category]['indents'].append(indent)
        return aggregated_demand
    
    def _create_bulk_production_vars(self, product_category: str, total_qty: float):
        """
        Creates batch and task variables for shared, upstream bulk production steps.
        """
        product = self.products[product_category]
        all_steps = product.processing_steps
        
        self._create_batching_and_fulfillment_vars(product_category, total_qty, product_category, product_category)
        
        first_packaging_idx = next((i for i, s in enumerate(all_steps) if s.process_type == ProcessType.PACKAGING), len(all_steps))
        bulk_steps = all_steps[:first_packaging_idx]

        for batch_index, batch_var in enumerate(self.batch_qty[product_category]):
            for step in bulk_steps:
                task_key = (product_category, product_category, batch_index, step.step_id)
                start_var, end_var = self._create_time_vars(product_category, batch_index, step.step_id)
                resource_vars, room_vars = self._create_resource_vars(product_category, batch_index, step)
                
                self._update_resource_usage_map(resource_vars, task_key, start_var, end_var, product_category, step.step_id, batch_index)
                self._update_resource_usage_map(room_vars, task_key, start_var, end_var, product_category, step.step_id, batch_index)
                
                self.task_vars[task_key] = {
                    'start': start_var, 'end': end_var, 'volume': batch_var,
                    'resources': resource_vars, 'rooms': room_vars,
                    'sku_id': product_category, 'step_id': step.step_id, 'batch_index': batch_index,
                    'base_duration': getattr(step, 'duration_minutes', 60),
                    'setup_time': getattr(step, 'setup_time', 0),
                    'priority': Priority.MEDIUM
                }

        # Find the end variable of the last batch of the last bulk step
        #last_step_id = bulk_steps[-1].step_id
        #last_batch_idx = len(self.batch_qty[product_category]) - 1
        #first_batch_idx = 0
        #if len(self.batch_qty[product_category])>0:
        #   first_batch_key = (product_category, product_category, first_batch_idx, last_step_id)
            
            #if first_batch_key in self.task_vars:
            #    bulk_completion_var = self.task_vars[first_batch_key]['end']
            #    return bulk_completion_var
        
        #return self.model.NewConstant(0)
    
    def _create_finishing_vars(self, indents_in_group: List[UserIndent]):
        """
        Creates task variables for SKU-specific finishing steps.
        """
        for indent in indents_in_group:
            order_no = indent.order_no
            sku = self.skus[indent.sku_id]
            product = self.products[sku.product_category]
            
            self._create_batching_and_fulfillment_vars(order_no, indent.qty_required_liters, sku.sku_id, product.product_category)
            
            all_steps = product.processing_steps
            first_packaging_idx = next((i for i, s in enumerate(all_steps) if s.process_type == ProcessType.PACKAGING), -1)

            if first_packaging_idx == -1:
                self.warnings.append(f"Product {product.product_category} has no packaging step; cannot create finishing tasks.")
                continue

            finishing_steps = all_steps[first_packaging_idx:]
            
            for batch_index, batch_var in enumerate(self.batch_qty[order_no]):
                for i, step in enumerate(finishing_steps):
                    task_key = (order_no, sku.sku_id, batch_index, step.step_id)
                    start_var, end_var = self._create_time_vars(order_no, batch_index, step.step_id)
                    resource_vars, room_vars = self._create_resource_vars(order_no, batch_index, step)

                    self._update_resource_usage_map(resource_vars, task_key, start_var, end_var, sku.sku_id, step.step_id, batch_index)
                    self._update_resource_usage_map(room_vars, task_key, start_var, end_var, sku.sku_id, step.step_id, batch_index)
                    
                    self.task_vars[task_key] = {
                        'start': start_var, 'end': end_var, 'volume': batch_var,
                        'resources': resource_vars, 'rooms': room_vars,
                        'sku_id': sku.sku_id, 'step_id': step.step_id, 'batch_index': batch_index,
                        'setup_time': getattr(step, 'setup_time', 0),
                        'priority': indent.priority
                    }
                    

                    # --- THE CRITICAL LINKING CONSTRAINT ---
                    # The FIRST finishing step of the FIRST batch of this order must happen after the bulk is ready.
                    # if batch_index == 0 and i == 0:
                    #    self.model.Add(start_var >= bulk_completion_var)

    # Add this new method to the AdvancedProductionScheduler class

    # Replace your existing function with this final, corrected version.

    # Replace the existing _add_material_flow_constraints function with this definitive version.

    def _add_material_flow_constraints(self):
        """
        [ROBUST V3 - FINAL] Ensures finishing tasks for a product category do not
        start until the FIRST fully completed batch of the corresponding bulk material is ready.
        """
        self.logger.info("Adding robust material flow (bulk-to-finish) constraints V3...")

        for product_category, demand_info in self.aggregated_demand.items():
            product = self.products.get(product_category)
            if not product or not product.processing_steps:
                continue

            # 1. Find the VERY LAST step in the bulk production sequence.
            # This is the step right before the first packaging step.
            first_packaging_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
            
            # If there's no packaging step, or the first step is packaging, there's no bulk phase to link from.
            if first_packaging_idx <= 0:
                self.warnings.append(f"No bulk phase found for {product_category}, skipping material flow constraint.")
                continue

            final_bulk_step_id = product.processing_steps[first_packaging_idx - 1].step_id
            self.logger.info(f"For '{product_category}', identified final bulk step as: '{final_bulk_step_id}'")


            # 2. Find the completion times for ONLY this final bulk step across ALL batches.
            batch_completion_times = [
                task['end'] for key, task in self.task_vars.items()
                if key[0] == product_category and key[3] == final_bulk_step_id
            ]

            if not batch_completion_times:
                self.warnings.append(f"Could not find any tasks for the final bulk step '{final_bulk_step_id}' of '{product_category}'")
                continue

            # 3. Find the time the FIRST of these completed batches is ready.
            # We use AddMinEquality to find the minimum of all batch completion times.
            first_material_ready_time = self.model.NewIntVar(0, self.schedule_horizon, f"first_mat_ready_{product_category}")
            self.model.AddMinEquality(first_material_ready_time, batch_completion_times)

            # 4. Find all finishing tasks for this category.
            finishing_task_start_vars = []
            for indent in demand_info['indents']:
                for key, task in self.task_vars.items():
                    if key[0] == indent.order_no:
                        finishing_task_start_vars.append(task['start'])
            
            if not finishing_task_start_vars:
                continue
            
            # 5. Constrain EVERY finishing task to start only after the first bulk material is ready.
            for start_var in finishing_task_start_vars:
                self.model.Add(start_var >= first_material_ready_time)
            
            self.logger.info(f"SUCCESS: Linked {len(finishing_task_start_vars)} finishing tasks for '{product_category}' to start after {first_material_ready_time.Name()}")
    def _create_batching_and_fulfillment_vars(self, job_id: str, required_qty: float, sku_id: str, product_id: str):
        """
        A generic helper to create batching and fulfillment variables for a job.
        A 'job' can be a bulk production run (where job_id=product_category) or a finishing order (where job_id=order_no).
        """
        # Use the existing _get_batch_size helper to determine the appropriate batch size
        batch_size = self._get_batch_size(sku_id, product_id)
        if batch_size <= 0:
            self.warnings.append(f"Invalid batch size of {batch_size} for job {job_id}. Using default.")
            batch_size = self.DEFAULT_BATCH_SIZE
        
        max_batches = math.ceil(required_qty / batch_size) if batch_size > 0 else 1

        # Batch-level quantity variables
        batch_vars = [self.model.NewIntVar(0, batch_size, f"batch_{job_id}_{b}") for b in range(max_batches)]
        self.batch_qty[job_id] = batch_vars

        # Total produced quantity for this job
        max_possible = batch_size * max_batches
        self.produced_quantity[job_id] = self.model.NewIntVar(0, max_possible, f"produced_qty_{job_id}")
        self.model.Add(self.produced_quantity[job_id] == sum(batch_vars))

        # For individual orders, we track under/overproduction against the original indent
        # For bulk jobs, this can be seen as a buffer.
        self.underproduction[job_id] = self.model.NewIntVar(0, int(required_qty), f"underproduction_{job_id}")
        self.overproduction[job_id] = self.model.NewIntVar(0, max_possible, f"overproduction_{job_id}")

        delta = self.produced_quantity[job_id] - int(required_qty)
        self.model.AddMaxEquality(self.underproduction[job_id], [0, -delta])
        self.model.AddMaxEquality(self.overproduction[job_id], [0, delta])
        
        # We still need a way to decide if the job is scheduled at all
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
    
    def _calculate_requirement_capacity(self, requirement, sku_id: str) -> tuple[float, str]:
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
   
    def _create_resource_vars(self, order_no: str, batch_index: int, step: ProcessingStep):
        """
        [MODIFIED] Creates all resource selection variables for a step, 
        handling multiple simultaneous requirements.
        """
        # These will collect all choices from all requirements
        all_resource_vars = {}
        all_room_vars = {}

        # The main change: Loop through each requirement in the step
        # This assumes 'step' now has a 'requirements' attribute.
        if not hasattr(step, 'requirements'):
            self.warnings.append(f"Step {step.step_id} is missing 'requirement' attribute. Skipping.")
            return {}, {}
        
        print(f' Step is {step.step_id}')

        for requirement in step.requirements:
            if requirement.resource_type == ResourceType.ROOM:
                # Get the choice variables for this specific room requirement
                room_vars = self._create_room_vars(order_no, batch_index, step.step_id, requirement)
                all_room_vars.update(room_vars)
            else:
                # Get the choice variables for this specific equipment/line/tank requirement
                resource_vars = self._create_equipment_vars(order_no, batch_index, step.step_id, requirement)
                all_resource_vars.update(resource_vars)
                print(resource_vars)
        return all_resource_vars, all_room_vars

    def _create_room_vars(self, order_no: str, batch_index: int, step_id: str, requirement: ResourceRequirement):
        """
        [MODIFIED] Create room selection variables for a SINGLE requirement.
        """
        room_vars = {}
        for room_id in requirement.compatible_ids:
            room_vars[room_id] = self.model.NewBoolVar(
                f'use_room_{order_no}_{batch_index}_{step_id}_{room_id}'
            )
        
        if room_vars:
            self.model.AddAtMostOne(room_vars.values())
        
        return room_vars

    def _create_equipment_vars(self, order_no: str, batch_index: int, step_id: str, requirement: ResourceRequirement):
        """
        [MODIFIED] Create equipment selection variables for a SINGLE requirement.
        The AddExactlyOne constraint now applies only to the compatible resources
        for this one requirement.
        """
        resource_vars = {}
        
        # Use the compatible_ids from the requirement object
        for res_id in requirement.compatible_ids:
            bool_var = self.model.NewBoolVar(
                # The name helps in debugging to know which requirement it came from
                f'use_res_{order_no}_{batch_index}_{step_id}_{res_id}'
            )
            resource_vars[res_id] = bool_var
        
        # This constraint now correctly applies to only the resources for this requirement
        # (e.g., choose exactly one of the heaters)
        if resource_vars:
            self.model.AddExactlyOne(resource_vars.values())

        return resource_vars    

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
            Adds all scheduling constraints to the model by calling the specialized helper functions.
            """
            self.logger.info("Adding all scheduling constraints to the model...")
            
            # --- Resource-level Constraints ---
            # Manages that a resource can only be used by one task at a time.
            self._add_dynamic_duration_constraints() #
            self._add_resource_capacity_constraints() #

            # Creates conditional cleaning tasks between production tasks.
            self._add_CIP_constraints() #
            
            # Manages that a CIP circuit can only run one cleaning task at a time.
            self._add_CIP_circuit_capacity_constraints() #
            
            # --- Task & Sequence Constraints ---
            
            # Penalizes idle time between consecutive steps of the same batch.
            self._add_stagnation_constraints() #
            
            # Ensures Batch 2 of an order cannot start until Batch 1 is finished.
            self._add_batch_ordering_constraints()
            
            # Handles tasks with variable durations based on the resource used.
            

            # --- Time & Deadline Constraints ---
            # Ensures tasks are scheduled within defined working windows.
            self._add_time_window_constraints() #
            
            # Adds penalties for finishing orders after their due date.
            self._add_due_date_constraints() #
            
            # --- Quantity & Fulfillment Constraints ---
            # Links the decision to schedule an order with its production quantity.
            self._add_production_quantity_constraints() #
            # self._add_mass_balance_constraints()     
            self._add_material_flow_constraints() #<- added line  
            self._add_batch_to_batch_linking_constraints

    def _add_resource_capacity_constraints(self):
        """
        [CORRECTED] Adds capacity constraints with nuanced logic.
        Fixes a TypeError by creating an explicit duration variable for each task.
        """
        self.logger.info("Adding resource time and volume capacity constraints...")

        for resource_id, task_list in self.resource_usage_map.items():
            if len(task_list) <= 1:
                continue

            resource_type = self._get_resource_type(resource_id)
            intervals = []

            for task_info in task_list:
                task_key = task_info['task_key']
                task_data = self.task_vars.get(task_key)
                print(f'Printing the task key:\n{task_key}\n and task data:\n{task_data}')
                if not task_data:
                    continue

                # --- START OF THE FIX ---
                # 1. Create a dedicated duration variable for the task.
                duration_var = self.model.NewIntVar(0, self.schedule_horizon, f"duration_{task_key}")
                
                # 2. Add a constraint to link it to start and end times.
                self.model.Add(duration_var == task_data['end'] - task_data['start'])
                
                # 3. Store the duration variable for later use (e.g., in dynamic duration constraints)
                self.task_vars[task_key]['duration'] = duration_var

                # 4. Use the new duration_var as the 'size' argument. This is now an affine expression.
                interval_var = self.model.NewOptionalIntervalVar(
                    task_data['start'],
                    duration_var,
                    task_data['end'],
                    task_info['assign_var'],
                    f"interval_{resource_id}_{task_key}"
                )
            
                intervals.append(interval_var)
                self.task_vars[task_key]['interval'] = interval_var

            # The rest of the function remains the same...
                if resource_type == ResourceType.ROOM:
                    demands = []
                    for task_info in task_list:
                        task_key = task_info['task_key']
                        task_data = self.task_vars[task_key]
                        sku_id = task_data['sku_id']
                        sku_object = self.skus.get(sku_id)

                        if sku_object and hasattr(sku_object, 'inventory_size') and sku_object.inventory_size > 0:
                            inventory_size_int = int(sku_object.inventory_size * 100)
                            demand_expr = self.model.NewIntVar(0, 1000000, f"demand_{task_key}")
                            self.model.Add(demand_expr == task_data['volume'] * inventory_size_int)
                            demands.append(demand_expr)
                        else:
                            demands.append(0)
                            self.warnings.append(f"SKU {sku_id} is missing 'inventory_size' for room capacity calculation.")

                    resource_object = self.rooms.get(resource_id)
                    if resource_object and hasattr(resource_object, 'capacity_units'): # Corrected attribute name
                        capacity = int(resource_object.capacity_units * 100)
                        self.model.AddCumulative(intervals, demands, capacity)
                        self.logger.info(f"Added CUMULATIVE constraint for Room '{resource_id}' with capacity {capacity}.")
                else:
                    self.model.AddNoOverlap(intervals)
                    # intervals.clear()

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
    
    # In class AdvancedProductionScheduler

    def _add_CIP_constraints(self):
        """
        [CORRECTED] Enforces conditional CIP and No-Overlap constraints by creating
        explicit CIP tasks without adding them to the primary resource's NoOverlap list,
        which correctly prevents redundant cleaning cycles.
        """
        self.logger.info("Adding corrected conditional CIP constraints...")

        for resource_id, tasks in self.resource_usage_map.items():
            if self._get_resource_type(resource_id) == ResourceType.ROOM or len(tasks) < 2:
                continue

            # This list will ONLY contain the main production tasks for this resource
            all_production_intervals_on_resource = []
            for task_info in tasks:
                task_key = task_info['task_key']
                # The interval for the main task is still needed for the resource's own NoOverlap
                interval = self.task_vars[task_key].get('interval')
                if interval:
                    all_production_intervals_on_resource.append(interval)

            cip_counter = 0
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    t1_info, t2_info = tasks[i], tasks[j]
                    b1, b2 = t1_info['assign_var'], t2_info['assign_var']

                    # This part remains the same, defining the order
                    lit_t1_before_t2 = self.model.NewBoolVar(f"t1_before_t2_{i}_{j}_{resource_id}")
                    self.model.Add(self.task_vars[t1_info['task_key']]['end'] <= self.task_vars[t2_info['task_key']]['start']).OnlyEnforceIf([b1, b2, lit_t1_before_t2])
                    self.model.Add(self.task_vars[t2_info['task_key']]['end'] <= self.task_vars[t1_info['task_key']]['start']).OnlyEnforceIf([b1, b2, lit_t1_before_t2.Not()])
                    self.model.AddBoolOr([lit_t1_before_t2, lit_t1_before_t2.Not()]).OnlyEnforceIf([b1, b2])

                    if self._needs_CIP_between(t1_info, t2_info, resource_id):
                        # Create the conditional CIP task. This part is correct.
                        # Forward Case (t1 -> t2)
                        active_cip_fwd = self.model.NewBoolVar(f"cip_active_{cip_counter}_fwd")
                        self.model.AddBoolOr([b1.Not(), b2.Not(), lit_t1_before_t2.Not(), active_cip_fwd])
                        self._create_CIP_task(resource_id, t1_info, t2_info, f"{cip_counter}_fwd", active_cip_fwd)

                        # Reverse Case (t2 -> t1)
                        active_cip_rev = self.model.NewBoolVar(f"cip_active_{cip_counter}_rev")
                        self.model.AddBoolOr([b1.Not(), b2.Not(), lit_t1_before_t2, active_cip_rev])
                        self._create_CIP_task(resource_id, t2_info, t1_info, f"{cip_counter}_rev", active_cip_rev)
                        
                        cip_counter += 1

            # The NoOverlap constraint is now ONLY applied to the actual production tasks.
            # The CIP tasks are sequenced by their own logic and constrained by the CIP circuit.
            if len(all_production_intervals_on_resource) > 1:
                self.model.AddNoOverlap(all_production_intervals_on_resource)

    def _create_CIP_task(self, resource_id: str, prev_task: dict, next_task: dict, CIP_index: Union[int, str], enforcement_literal: cp_model.BoolVarT) -> cp_model.IntervalVar:
        """
        [MODIFIED] Creates a single, conditional CIP task and returns its interval variable.
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

        self.CIP_circuit_usage[CIP_circuit_id].append(CIP_interval)
        self.model.Add(CIP_end == CIP_start + CIP_time).OnlyEnforceIf(enforcement_literal)
        # The sequencing is now explicitly tied to the CIP interval itself
        self.model.Add(CIP_start >= self.task_vars[prev_task['task_key']]['end']).OnlyEnforceIf(enforcement_literal)
        self.model.Add(self.task_vars[next_task['task_key']]['start'] >= CIP_end).OnlyEnforceIf(enforcement_literal)

        self.CIP_vars[CIP_id] = {
            'start': CIP_start, 'end': CIP_end, 'resource': resource_id,
            'interval': CIP_interval, 'enforced_by': enforcement_literal,
        }
        return CIP_interval # <-- Return the interval                     
    
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

    def _add_batch_ordering_constraints(self):
        """
        [CORRECTED] Adds a soft penalty for scheduling batches out of numerical
        order. This version correctly identifies the first step of any job
        (bulk or finishing) to resolve KeyErrors.
        """
        self.logger.info("Adding SOFT batch ordering constraints...")
        
        if "batch_out_of_sequence_penalty" not in self.weights:
            self.weights["batch_out_of_sequence_penalty"] = 10

        out_of_sequence_penalties = []
        valid_order_nos = {i.order_no for i in self.valid_indents}

        for job_id, batches in self.batch_qty.items():
            if len(batches) <= 1:
                continue

            # --- START OF FIX ---
            # Robustly determine the product, sku_id, and relevant first step for this job.
            first_step_id = None
            sku_id = None

            if job_id in valid_order_nos: # This is a Finishing Job
                try:
                    indent = next(i for i in self.valid_indents if i.order_no == job_id)
                    sku_id = indent.sku_id
                    product = self.products[self.skus[sku_id].product_category]
                    
                    # For finishing jobs, the "first step" is the first packaging step.
                    first_packaging_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
                    if first_packaging_idx != -1:
                        first_step_id = product.processing_steps[first_packaging_idx].step_id
                except StopIteration:
                    continue # Should not happen if valid_order_nos is built correctly

            else: # This is a Bulk Job
                sku_id = job_id # For bulk jobs, sku_id is the same as the product_category/job_id
                product = self.products.get(job_id)
                if product and product.processing_steps:
                    first_step_id = product.processing_steps[0].step_id
            
            if not first_step_id or not sku_id:
                self.warnings.append(f"Could not determine first step for job {job_id} in batch ordering.")
                continue
            # --- END OF FIX ---

            for i in range(len(batches) - 1):
                # We compare the start times of the first step of consecutive batches
                prev_key = (job_id, sku_id, i, first_step_id)
                next_key = (job_id, sku_id, i + 1, first_step_id)

                if prev_key in self.task_vars and next_key in self.task_vars:
                    start_prev = self.task_vars[prev_key]['start']
                    start_next = self.task_vars[next_key]['start']

                    is_out_of_sequence = self.model.NewBoolVar(f'is_ooo_{job_id}_b{i+1}')
                    self.model.Add(start_next < start_prev).OnlyEnforceIf(is_out_of_sequence)
                    out_of_sequence_penalties.append(is_out_of_sequence)

        if out_of_sequence_penalties:
            # Add penalty to the objective terms list for later summation
            penalty = sum(out_of_sequence_penalties) * int(self.weights["batch_out_of_sequence_penalty"])
            self.cost_vars['out_of_sequence_penalty'] = penalty

    def _add_dynamic_duration_constraints(self):
        """
        [REVISED & CORRECTED] Adds constraints to dynamically calculate task duration.
        This version now correctly handles both dynamic (speed-based) and fixed durations,
        and creates the duration variable itself to prevent KeyErrors.
        """
        self.logger.info("Adding dynamic and fixed duration constraints...")
        OEE_FACTOR = 0.85 # Overall Equipment Effectiveness

        for task_key, task_data in self.task_vars.items():
            order_no, sku_id, batch_index, step_id = task_key

            # --- FIX: Create the duration variable here to avoid dependency issues ---
            # This ensures every task gets a duration variable.
            duration_expr = self.model.NewIntVar(0, self.schedule_horizon, f"duration_{task_key}")
            self.model.Add(task_data['end'] == task_data['start'] + duration_expr)
            task_data['duration'] = duration_expr # Store for other functions to use
            # --- END FIX ---
            
            batch_volume_var = task_data['volume']
            
            # Combine all possible resources for this step
            all_possible_resources = {**task_data.get('resources', {}), **task_data.get('rooms', {})}
            
            # Flag to see if any resource has a dynamic speed
            has_dynamic_duration_constraint = False

            for resource_id, resource_var in all_possible_resources.items():
                resource = (self.lines.get(resource_id) or 
                            self.tanks.get(resource_id) or 
                            self.equipments.get(resource_id))
                
                if not resource:
                    continue

                setup_time = getattr(resource, 'setup_time_minutes', task_data.get('setup_time', 0))
                resource_type = self._get_resource_type(resource_id)
                
                # --- DURATION CALCULATION LOGIC ---
                dynamic_speed = None
                if resource_type == ResourceType.LINE and hasattr(resource, 'compatible_skus_max_production'):
                    dynamic_speed = resource.compatible_skus_max_production.get(sku_id)
                elif hasattr(resource, 'processing_speed'):
                    dynamic_speed = resource.processing_speed

                if dynamic_speed is not None and dynamic_speed > 0:
                    has_dynamic_duration_constraint = True
                    effective_rate = int(dynamic_speed * OEE_FACTOR * 100)
                    # Using multiplication to avoid division with model variables.
                    # volume * 100 == (duration - setup) * effective_rate
                    self.model.Add(batch_volume_var * 100 == (duration_expr - setup_time) * effective_rate).OnlyEnforceIf(resource_var)
            
            # If NO resource for this task has a dynamic speed, enforce the base duration.
            if not has_dynamic_duration_constraint:
                is_task_active = self.model.NewBoolVar(f"is_active_{task_key}")
                self.model.Add(batch_volume_var > 0).OnlyEnforceIf(is_task_active)
                self.model.Add(batch_volume_var == 0).OnlyEnforceIf(is_task_active.Not())
                
                base_duration = task_data.get('base_duration', 0)
                # Find a resource to get setup time from, assuming it's the same for all in this step
                any_resource_id = next(iter(all_possible_resources.keys()), None)
                any_resource = self.lines.get(any_resource_id) or self.tanks.get(any_resource_id) or self.equipments.get(any_resource_id)
                setup_time = getattr(any_resource, 'setup_time_minutes', task_data.get('setup_time', 0)) if any_resource else 0

                # Enforce fixed duration only if the task is active
                self.model.Add(duration_expr == base_duration + setup_time).OnlyEnforceIf(is_task_active)
                self.model.Add(duration_expr == 0).OnlyEnforceIf(is_task_active.Not())

    def _add_stagnation_constraints(self):
        """
        [CORRECTED] Adds stagnation penalties AND hard sequencing constraints for consecutive steps.
        This fix ensures tasks are processed in the correct physical order.
        """
        self.logger.info("Adding inter-step sequence and stagnation constraints...")
        valid_order_nos = {i.order_no for i in self.valid_indents}

        for job_id, batches in self.batch_qty.items():
            if not batches: continue

            # Identify the correct product and sequence of steps for this job
            # (This logic correctly separates bulk and finishing steps)
            if job_id in valid_order_nos:
                indent = next((i for i in self.valid_indents if i.order_no == job_id), None)
                if not indent: continue
                sku_id = indent.sku_id
                product = self.products[self.skus[sku_id].product_category]
                first_packaging_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
                if first_packaging_idx == -1: continue
                steps_to_sequence = product.processing_steps[first_packaging_idx:]
            else: # This is a BULK job
                sku_id = job_id
                product = self.products.get(job_id)
                if not product: continue
                first_packaging_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), len(product.processing_steps))
                steps_to_sequence = product.processing_steps[:first_packaging_idx]

            # Apply constraints for each batch
            for batch_idx in range(len(batches)):
                for i in range(len(steps_to_sequence) - 1):
                    current_step = steps_to_sequence[i]
                    next_step = steps_to_sequence[i+1]
                    
                    current_task_key = (job_id, sku_id, batch_idx, current_step.step_id)
                    next_task_key = (job_id, sku_id, batch_idx, next_step.step_id)

                    if current_task_key in self.task_vars and next_task_key in self.task_vars:
                        current_task_end = self.task_vars[current_task_key]['end']
                        next_task_start = self.task_vars[next_task_key]['start']

                        # --- THE CRITICAL FIX ---
                        # 1. Add the missing HARD constraint to enforce the sequence.
                        self.model.Add(next_task_start >= current_task_end)

                        # 2. Keep the soft constraint to penalize idle time (the gap).
                        gap_var = self.model.NewIntVar(0, self.schedule_horizon, f"gap_{job_id}_{batch_idx}_{i}")
                        self.model.Add(gap_var == next_task_start - current_task_end)
                        self.stagnation_vars.append(gap_var)

    def _add_time_window_constraints(self):
        """
        [REVISED] Ensures tasks are scheduled only within working windows by creating
        No-Overlap constraints against fixed non-working intervals.
        """
        self.logger.info("Adding time window (working hours) constraints...")
        
        # Get a list of all non-working time slots from the TimeManager
        non_working_periods = self.time_manager._create_non_working_intervals()
        
        # Create fixed, non-optional interval variables for each non-working period
        non_working_intervals = []
        for i, (start, duration) in enumerate(non_working_periods):
            if duration > 0:
                non_working_intervals.append(
                    self.model.NewIntervalVar(start, duration, start + duration, f'non_working_{i}')
                )

        if not non_working_intervals:
            return # No constraints to add if the factory is open 24/7

        # For every task, add a constraint that its interval cannot overlap with any non-working interval
        for task_data in self.task_vars.values():
            task_interval = task_data.get('interval')
            if task_interval:
                self.model.AddNoOverlap(non_working_intervals + [task_interval])
   
    def _add_production_quantity_constraints(self):
        """
        Add constraints for production quantities. If an order is not scheduled,
        the quantity for all its batches must be zero.
        """
        for order_no, batches in self.batch_qty.items():
            is_scheduled_var = self._is_scheduled[order_no]
            # Link is_scheduled to actual production
            self.model.Add(self.produced_quantity[order_no] > 0).OnlyEnforceIf(is_scheduled_var)
            self.model.Add(self.produced_quantity[order_no] == 0).OnlyEnforceIf(is_scheduled_var.Not())

            for batch_var in batches:
                # If not scheduled, batch quantity must be 0
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
    
    def _add_batch_to_batch_linking_constraints(self):
        """
        [ROBUST V6 - FINAL] Creates a direct 1-to-1 time dependency between
        bulk production batches and finishing consumption batches.
        e.g., Finishing_Batch_2 cannot start until Bulk_Batch_2 is complete.
        """
        self.logger.info("Adding direct Batch-to-Batch linking constraints...")

        for product_category, demand_info in self.aggregated_demand.items():
            product = self.products.get(product_category)
            if not product or not product.processing_steps:
                continue

            # 1. Find the final step_id of the bulk process
            first_packaging_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
            if first_packaging_idx <= 0: continue
            final_bulk_step_id = product.processing_steps[first_packaging_idx - 1].step_id

            # 2. Collect all bulk batch completion times, in order by batch_index.
            bulk_batches = sorted([
                task for key, task in self.task_vars.items()
                if key[0] == product_category and key[3] == final_bulk_step_id
            ], key=lambda x: x['batch_index'])
            
            # 3. Collect all finishing batch start times, in order.
            finishing_batches = []
            for indent in demand_info['indents']:
                # Get the first packaging step for this specific indent's SKU
                sku_product = self.products.get(self.skus[indent.sku_id].product_category)
                if not sku_product: continue
                first_pack_step_id = next((s.step_id for s in sku_product.processing_steps if s.process_type == ProcessType.PACKAGING), None)
                if not first_pack_step_id: continue
                
                # Sort by order number and then by batch index for a consistent, predictable order
                finishing_batches.extend(sorted([
                    task for key, task in self.task_vars.items()
                    if key[0] == indent.order_no and key[3] == first_pack_step_id
                ], key=lambda x: (x['sku_id'], x['batch_index'])))

            if not bulk_batches or not finishing_batches:
                continue

            # 4. Add the 1-to-1 linking constraints for as many pairs as possible
            num_links = min(len(bulk_batches), len(finishing_batches))
            for i in range(num_links):
                bulk_batch_end_var = bulk_batches[i]['end']
                finishing_batch_start_var = finishing_batches[i]['start']
                self.model.Add(finishing_batch_start_var >= bulk_batch_end_var)
            
            # 5. [NEW] Constraint for remaining finishing batches: they must wait for the last bulk batch.
            # This handles cases where there are more finishing batches than bulk batches (e.g. 3 finish, 2 bulk).
            if len(finishing_batches) > len(bulk_batches) and bulk_batches:
                last_bulk_batch_end_var = bulk_batches[-1]['end']
                for i in range(len(bulk_batches), len(finishing_batches)):
                    self.model.Add(finishing_batches[i]['start'] >= last_bulk_batch_end_var)

            self.logger.info(f"SUCCESS: Created {len(finishing_batches)} batch dependency links for '{product_category}'.")

    # Replace the previous reservoir function with this corrected version.

    def _add_reservoir_mass_balance_constraints(self):
        """
        [ROBUST V5 - FINAL & CORRECTED] Uses a reservoir constraint to ensure
        that the quantity of bulk material consumed never exceeds the quantity
        produced at any point in time. This version uses fixed demands to create
        a valid model for the solver.
        """
        self.logger.info("Adding Reservoir Mass Balance constraints (Corrected)...")

        for product_category, demand_info in self.aggregated_demand.items():
            product = self.products.get(product_category)
            if not product or not product.processing_steps:
                continue

            # --- FIX: Determine the standard batch size to use as a fixed demand ---
            # We use the batch size for the product category itself.
            batch_size = self._get_batch_size(product_category, product_category)
            if batch_size <= 0:
                self.warnings.append(f"Invalid batch size ({batch_size}) for {product_category}, skipping reservoir constraint.")
                continue

            # Find the final step of the bulk process
            first_packaging_idx = next((i for i, s in enumerate(product.processing_steps) if s.process_type == ProcessType.PACKAGING), -1)
            if first_packaging_idx <= 0:
                continue
            final_bulk_step_id = product.processing_steps[first_packaging_idx - 1].step_id

            times = []
            demands = []
            
            # 1. Add PRODUCTION events to the reservoir
            production_batches = {
                key: task for key, task in self.task_vars.items()
                if key[0] == product_category and key[3] == final_bulk_step_id
            }
            for key, task in production_batches.items():
                times.append(task['end'])
                # Use the fixed integer batch_size for the demand
                demands.append(batch_size)

            # 2. Add CONSUMPTION events to the reservoir
            for indent in demand_info['indents']:
                sku_product = self.products.get(self.skus[indent.sku_id].product_category)
                if not sku_product: continue
                
                first_pack_step_id = next((s.step_id for s in sku_product.processing_steps if s.process_type == ProcessType.PACKAGING), None)
                if not first_pack_step_id: continue

                # Determine the batch size for this specific finishing SKU
                finishing_batch_size = self._get_batch_size(indent.sku_id, sku_product.product_category)
                if finishing_batch_size <= 0: continue

                consumption_batches = {
                    key: task for key, task in self.task_vars.items()
                    if key[0] == indent.order_no and key[3] == first_pack_step_id
                }
                for key, task in consumption_batches.items():
                    times.append(task['start'])
                    # Use the fixed integer batch_size for the demand
                    demands.append(-finishing_batch_size)

            # 3. Add the reservoir constraint for this product category
            if times:
                self.model.AddReservoirConstraint(times, demands, min_level=0, max_level=9999999)
                self.logger.info(f"SUCCESS: Added reservoir constraint for '{product_category}' with {len(times)} events.")

    def _get_order_completion_time(self, order_no: str) -> Optional[cp_model.IntVar]:
        """
        [FINAL REVISED] Get the completion time variable for an order by finding the
        end time of the true final step in its finishing process.
        """
        try:
            # Find the specific indent to get the SKU and product info
            indent = next(i for i in self.valid_indents if i.order_no == order_no)
            sku_id = indent.sku_id
            product = self.products[self.skus[sku_id].product_category]
            all_steps = product.processing_steps

            # --- START OF CORRECTION ---
            # Find the index of the first packaging step in the product's recipe.
            first_packaging_idx = -1
            for i, step in enumerate(all_steps):
                if step.process_type == ProcessType.PACKAGING:
                    first_packaging_idx = i
                    break

            # If a product has no packaging step, we cannot determine its finishing time.
            if first_packaging_idx == -1:
                self.warnings.append(f"Order {order_no} has no packaging steps; cannot determine its completion time.")
                return None

            # The finishing process is defined as the first packaging step and all subsequent steps.
            finishing_process_steps = all_steps[first_packaging_idx:]
            
            # The true final step for the order is the last one in this finishing sequence.
            last_step_id = finishing_process_steps[-1].step_id
            # --- END OF CORRECTION ---

            num_batches = len(self.batch_qty.get(order_no, []))
            if num_batches == 0:
                return None
            
            last_batch_idx = num_batches - 1
            last_task_key = (order_no, sku_id, last_batch_idx, last_step_id)
            
            if last_task_key in self.task_vars:
                return self.task_vars[last_task_key]['end']
                
        except StopIteration:
            self.logger.warning(f"Could not find indent for order_no {order_no} in _get_order_completion_time.")
            return None
        
        return None

    def _add_mass_balance_constraints(self):
        """
        [NEW] Ensures that the total quantity of finished goods produced for a category
        does not exceed the total quantity of the parent bulk material produced.
        """
        self.logger.info("Adding mass-balance constraints (Bulk >= Sum of Finish)...")
        
        # self.aggregated_demand holds the grouping of indents by category
        for product_category, demand_info in self.aggregated_demand.items():
            
            # Get the single solver variable for the total bulk production of this category
            if product_category not in self.produced_quantity:
                continue
            bulk_produced_var = self.produced_quantity[product_category]
            
            # Create a list of all the finished good production variables for this category
            finish_produced_vars = []
            for indent in demand_info['indents']:
                if indent.order_no in self.produced_quantity:
                    finish_produced_vars.append(self.produced_quantity[indent.order_no])
            
            # Add the constraint: Bulk Production >= Sum of all related Finish Production
            if finish_produced_vars:
                self.model.Add(sum(finish_produced_vars) <= bulk_produced_var)

    def _create_objective(self):
        """
        [CORRECTED] Create a comprehensive objective function that aligns with the "Bulk-then-Finish" strategy.
        """
        self.logger.info("Creating fulfillment-focused objective function...")
        
        objective_terms = []
        SCALE = 1000

        # --- 1. Fulfillment-Driven Value (Bonuses and Penalties for Final Orders) ---
        # A single, consolidated loop for all indent-specific objectives.
        for indent in self.valid_indents:
            order_no = indent.order_no
            
            # Defensive check to ensure we have variables for this order
            if order_no not in self.produced_quantity:
                continue

            # a. Production Bonus: Applied ONLY to the produced quantity of a final, sellable good.
            required_qty = indent.qty_required_liters
            if required_qty > 0:
                # Create a new variable to represent the fulfillment percentage (scaled by 1000)
                # Its domain is 0 to 1000, representing 0% to 100%
                fulfillment_pct_scaled = self.model.NewIntVar(0, SCALE, f"fulfillment_pct_{order_no}")

                # Add a constraint that defines the fulfillment percentage:
                # fulfillment_pct_scaled / 1000 = produced_quantity / required_qty
                # To avoid floats, we rearrange to:
                # fulfillment_pct_scaled * required_qty = produced_quantity * 1000
                self.model.Add(
                    fulfillment_pct_scaled * int(required_qty) <= self.produced_quantity[order_no] * SCALE
                )
                
                # a. Production Bonus: Applied to the scaled fulfillment percentage.
                objective_terms.append(
                    fulfillment_pct_scaled * int(self.weights["production_volume"])
                )
            
            # b. Under/Overproduction Penalties: MOVED INSIDE the loop to apply to every order.
            self._add_piecewise_underproduction_penalty(order_no, objective_terms)
            objective_terms.append(
                -self.overproduction[order_no] * int(self.weights["overproduction_penalty"]* int(self.weights["overproduction_penalty"]))
            )

            # c. Priority Bonus: CONSOLIDATED into this loop.
            priority_multiplier = indent.priority.value
            objective_terms.append(
                self._is_scheduled[order_no] *priority_multiplier * priority_multiplier * int(self.weights["priority_bonus"])
            )
        
        # --- 2. Global Costs and Penalties ---

        # a. CIP time penalties
        for CIP_data in self.CIP_vars.values():
            CIP_duration = CIP_data['interval'].SizeExpr()
            objective_terms.append(-CIP_duration * int(self.weights["CIP_time_penalty"]*10.0))

        # b. Resource utilization
        self._add_resource_utilization_objectives(objective_terms)
        
        # c. Stagnation (idle time between steps) penalty
        for gap_var in self.stagnation_vars:
            objective_terms.append(-gap_var * int(self.weights["idle_time_penalty"]))
        
        # d. Late delivery penalty
        for tardiness_var in self.tardiness_vars.values():
            objective_terms.append(-tardiness_var * int(self.weights["late_minutes_penalty"]))

        makespan_var = self.model.NewIntVar(0, self.schedule_horizon, 'makespan')
        all_end_times = [task['end'] for task in self.task_vars.values()]
        
        if all_end_times:
            self.model.AddMaxEquality(makespan_var, all_end_times)
            # Penalize the makespan to force the solver to finish everything as early as possible.
            objective_terms.append(-makespan_var * int(self.weights["makespan_penalty"]))

        # --- 3. Finalize Objective ---
        if objective_terms:
            self.model.Maximize(sum(objective_terms))
        else:
            self.logger.warning("No objective terms created!")  

    def _add_piecewise_underproduction_penalty(self, order_no: str, objective_terms: list):
        SCALE=1000
        """Add piecewise linear underproduction penalty"""
        under_var = self.underproduction[order_no]
        indent = next(i for i in self.valid_indents if i.order_no == order_no)
        required_qty = int(indent.qty_required_liters)
        under_frac_scaled = self.model.NewIntVar(0, SCALE, f"{order_no}_under_frac_scaled")
        self.model.Add(under_frac_scaled * required_qty == under_var * SCALE)
        under_frac_squared = self.model.NewIntVar(0, SCALE*SCALE, f"{order_no}_under_frac_squared")
        self.model.AddMultiplicationEquality(under_frac_squared, [under_frac_scaled, under_frac_scaled])

        penalty_per_pct = int(self.weights["underproduction_pct_penalty"])
        objective_terms.append(-under_frac_scaled * penalty_per_pct )
        objective_terms.append(-under_frac_squared * penalty_per_pct )

    def _add_resource_utilization_objectives(self, objective_terms: list):
        """
        [CORRECTED] Add resource utilization bonuses and penalties.
        The bonus for mere utilization has been removed to prevent incentivizing inefficiency.
        """
        all_resources = list(config.ROOMS.keys()) + list(config.EQUIPMENTS.keys()) + list(config.LINES.keys()) + list(config.TANKS.keys())

        for resource_id in all_resources:
            # Calculate the total time the resource is actually used
            total_used_time = self._calculate_resource_utilization(resource_id)
            
            # --- REMOVED ---
            # The bonus for utilization has been removed. We should not reward the model
            # for simply being busy; we should reward it for producing volume (done elsewhere)
            # while penalizing the costs (time, CIP, etc).
            # bonus_weight = ...
            # objective_terms.append(total_used_time * int(bonus_weight))
            
            # The penalty for a resource being idle is still valuable.
            available_time = self._get_available_time_for_resource(resource_id)
            if available_time > 0:
                underutil_var = self.model.NewIntVar(0, available_time, f"underutil_{resource_id}")
                self.model.Add(underutil_var >= available_time - total_used_time)
                
                # This logic remains correct.
                penalty_weight = self.weights.get("room_underutilisation_penalty") if self._get_resource_type(resource_id) == ResourceType.ROOM else self.weights.get("tank_underutilisation_penalty_tiers", [10])[0]
                objective_terms.append(-underutil_var * int(penalty_weight))

    def _calculate_resource_utilization(self, resource_id: str) -> cp_model.IntVar:
        """
        [CORRECTED] Calculate total utilization time for a resource.
        Fixes a bug by calculating duration from start/end variables instead of a non-existent key.
        """
        if resource_id not in self.resource_usage_map:
            return self.model.NewConstant(0)
        
        total_time = self.model.NewIntVar(0, self.schedule_horizon, f"total_time_{resource_id}")
        
        task_durations = []
        for task_info in self.resource_usage_map[resource_id]:
            task_key = task_info['task_key']
            
            # --- FIX ---
            # Calculate duration directly from the task's start and end variables.
            # The key 'duration' did not exist in self.task_vars.
            task_data = self.task_vars[task_key]
            duration_expr = task_data['end'] - task_data['start']
            
            # Create a variable to hold the duration if the task is assigned to this resource
            assigned_duration = self.model.NewIntVar(0, self.schedule_horizon, f"assigned_duration_{resource_id}_{task_key}")
            
            self.model.Add(assigned_duration == duration_expr).OnlyEnforceIf(task_info['assign_var'])
            self.model.Add(assigned_duration == 0).OnlyEnforceIf(task_info['assign_var'].Not())
            task_durations.append(assigned_duration)
        
        if task_durations:
            self.model.Add(total_time == sum(task_durations))
        else:
            self.model.Add(total_time == 0)
        
        return total_time
    
    def _get_available_time_for_resource(self, resource_id: str) -> int:
        """Get total available time for a resource within the schedule horizon"""
        total_available = 0
        for start_time, end_time in self.time_manager.working_windows:
            total_available += (end_time - start_time)
        return total_available
    
    def _extract_enhanced_solution(self, status) -> SchedulingResult:
        """
        Extract comprehensive solution from the solved model.
        """
        self.logger.info("Extracting solution...")
        
        scheduled_tasks = []
        resource_utilization = defaultdict(list)
        production_summary = {}
        CIP_schedules = []
        
        # Extract task schedules

        for task_key, task_data in self.task_vars.items():
            volume = self.solver.Value(task_data['volume'])
            if volume == 0:
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
                    resource_id=assigned_resource, volume=volume, priority=task_data['priority']
                )
                scheduled_tasks.append(task_schedule)
                resource_utilization[assigned_resource].append({'start': start_time, 'end': end_time, 'task_id': task_schedule.task_id})
        
        # Extract CIP schedules
        for CIP_id, CIP_data in self.CIP_vars.items():
            if self.solver.BooleanValue(CIP_data['enforced_by']):
                start = self.solver.Value(CIP_data['start'])
                end = self.solver.Value(CIP_data['end'])
                CIP_schedules.append(CIPSchedule(
                    CIP_id=CIP_id,
                    resource_id=CIP_data['resource'],
                    start_time=self.time_manager.schedule_start + timedelta(minutes=start),
                    end_time=self.time_manager.schedule_start + timedelta(minutes=end),
                    duration_minutes=int((end - start)),
                    preceding_task_id= 'unknown',
                    following_task_id= 'unknown'
                ))

        # Extract production summary
        for order_no, produced_var in self.produced_quantity.items():
            produced_qty = self.solver.Value(produced_var)
            if produced_qty > 0:
                production_summary[order_no] = {
                    'produced_quantity': produced_qty,
                    'underproduction': self.solver.Value(self.underproduction[order_no]),
                    'overproduction': self.solver.Value(self.overproduction[order_no]),
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
        """Calculate a comprehensive score for the solution."""
        if not result or not result.scheduled_tasks:
            return float('-inf')
        
        score = 0.0
        score += result.objective_value
        score += len(result.production_summary) * 1000
        score -= len(result.warnings) * 50
        
        total_utilization_rate = 0
        if result.resource_utilization:
            num_resources = len(self.resource_usage_map)
            for resource_id, tasks in result.resource_utilization.items():
                if tasks:
                    total_time = sum(task['end'] - task['start'] for task in tasks)
                    available_time = self._get_available_time_for_resource(resource_id)
                    if available_time > 0:
                        total_utilization_rate += total_time / available_time
            
            avg_utilization = total_utilization_rate / num_resources if num_resources > 0 else 0
            score += avg_utilization * 1000
        
        return result.objective_value

    def _get_resource_type(self, resource_id: str) -> ResourceType:
        """Get resource type with graceful error handling"""
        if resource_id in config.TANKS: return ResourceType.TANK
        if resource_id in config.LINES: return ResourceType.LINE
        if resource_id in config.EQUIPMENTS: return ResourceType.EQUIPMENT
        if resource_id in config.ROOMS: return ResourceType.ROOM
        self.logger.warning(f"Unknown resource ID: {resource_id}. Defaulting to EQUIPMENT type.")
        return ResourceType.EQUIPMENT

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
    
    # Add this new method to the AdvancedProductionScheduler class
    # In class AdvancedProductionScheduler

    def generate_schedule_log_file(self, result: SchedulingResult, file_path: str = "schedule_log.txt"):
        """
        [NEW DIAGNOSTIC TOOL] Generates a human-readable text file detailing the
        final production schedule, organized chronologically and by resource.
        """
        self.logger.info(f"Generating detailed schedule log file at: {file_path}")
        
        # Ensure the directory for the log file exists
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except FileExistsError:
            pass # Directory already exists
        except Exception as e:
            self.logger.error(f"Could not create directory for log file: {e}")
            # Continue anyway, it might write to the root directory
            
        with open(file_path, "w") as f:
            f.write("="*80 + "\n")
            f.write(" " * 25 + "FINAL PRODUCTION SCHEDULE LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Schedule Start Time: {self.time_manager.schedule_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Solver Status: {self.solver.StatusName(result.status)}\n")
            f.write(f"Objective Value: {result.objective_value:.2f}\n")
            f.write("\n")

            if not result.is_feasible or not result.scheduled_tasks:
                f.write("No feasible schedule was generated.\n")
                return

            # --- 1. Chronological Full-Factory Schedule ---
            f.write("="*80 + "\n")
            f.write(" chronological full-factory schedule\n".upper())
            f.write("="*80 + "\n\n")
            
            all_events = result.scheduled_tasks + result.CIP_schedules
            all_events.sort(key=lambda x: x.start_time)

            for event in all_events:
                start_str = event.start_time.strftime('%m-%d %H:%M')
                end_str = event.end_time.strftime('%m-%d %H:%M')
                duration = int((event.end_time - event.start_time).total_seconds() / 60)
                
                if isinstance(event, TaskSchedule):
                    # It's a production task
                    event_type = "TASK"
                    details = f"{event.order_no} | {event.sku_id} | Batch {event.batch_index} | Step: {event.step_id}"
                else:
                    # It's a CIP task
                    event_type = "-- CLEANING (CIP) --"
                    details = f"Duration: {duration} mins"

                f.write(f"[{start_str} -> {end_str}] {event_type:<20} | Resource: {event.resource_id:<20} | {details}\n")
            
            # --- 2. Schedule per Resource ---
            f.write("\n\n" + "="*80 + "\n")
            f.write(" schedule by resource\n".upper())
            f.write("="*80 + "\n\n")

            # Group events by resource
            resource_schedule = defaultdict(list)
            for event in all_events:
                resource_schedule[event.resource_id].append(event)
            
            sorted_resources = sorted(resource_schedule.keys())

            for resource_id in sorted_resources:
                f.write(f"--- Resource: {resource_id} ---\n")
                
                total_busy_minutes = 0
                # Sort events for this resource chronologically
                sorted_events = sorted(resource_schedule[resource_id], key=lambda x: x.start_time)

                for event in sorted_events:
                    start_str = event.start_time.strftime('%m-%d %H:%M')
                    end_str = event.end_time.strftime('%m-%d %H:%M')
                    duration = int((event.end_time - event.start_time).total_seconds() / 60)
                    total_busy_minutes += duration

                    if isinstance(event, TaskSchedule):
                        details = f"Task: {event.task_id}"
                    else:
                        details = "Task: Cleaning (CIP)"
                    f.write(f"  [{start_str} -> {end_str}] ({duration: >3} min) | {details}\n")

                # Calculate utilization
                available_time = self._get_available_time_for_resource(resource_id)
                utilization_pct = (total_busy_minutes / available_time * 100) if available_time > 0 else 0
                f.write(f"  Resource Utilization: {utilization_pct:.1f}%\n\n")
            
            # --- 3. Production Fulfillment Summary ---
            f.write("\n" + "="*80 + "\n")
            f.write(" production fulfillment summary\n".upper())
            f.write("="*80 + "\n\n")
            
            for indent in self.valid_indents:
                order_no = indent.order_no
                summary = result.production_summary.get(order_no)
                
                if summary:
                    produced = summary['produced_quantity']
                    under = summary['underproduction']
                    over = summary['overproduction']
                    fulfillment_pct = (produced / indent.qty_required_liters * 100) if indent.qty_required_liters > 0 else 100
                    status = f"{fulfillment_pct:.1f}% Fulfilled"
                    if under > 0: status += f" (UNDER by {under}L)"
                    if over > 0: status += f" (OVER by {over}L)"

                    f.write(f"Order: {order_no:<10} | Required: {indent.qty_required_liters:<5}L | Produced: {produced:<5}L | Status: {status}\n")
                else:
                    f.write(f"Order: {order_no:<10} | Required: {indent.qty_required_liters:<5}L | Produced: 0L | Status: NOT PRODUCED\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
    # Replace your _run_diagnostics function with this complete, corrected version.

    def _run_diagnostics(self, categories_to_debug: Optional[List[str]] = None):
        """
        [Corrected & Improved] A special function to print detailed debugging
        information to diagnose sequencing issues. It can run on specific categories
        or on all of them by default.
        """
        print("\n" + "#"*60)
        print("### RUNNING SCHEDULER DIAGNOSTICS ###")
        print("#"*60 + "\n")

        # If no specific categories are provided, run diagnostics on ALL products.
        product_keys = categories_to_debug or list(self.products.keys())

        for category_name in product_keys:
            print("\n" + "="*50)
            print(f"DIAGNOSING CATEGORY: '{category_name}'")
            print("="*50)

            # --- 1. Check for BULK tasks ---
            # Bulk tasks are identified by having their key[0] and key[1] be the category name.
            print("--- 1. Checking for BULK tasks ---")
            bulk_tasks = {
                key: task for key, task in self.task_vars.items()
                if key[0] == category_name and key[1] == category_name
            }
            if not bulk_tasks:
                print(f"  [!!] DIAGNOSTIC WARNING: No bulk tasks were found for '{category_name}'. This is a likely cause of the issue.\n")
            else:
                print(f"  [OK] Found {len(bulk_tasks)} bulk tasks:")
                for key, task in bulk_tasks.items():
                    print(f"    - Bulk Task: {key}, End Var: {task['end'].Name()}")
            print("-" * 20)

            # --- 2. Check Material Ready Time ---
            print("--- 2. Checking Material Ready Time variable ---")
            bulk_tasks_end_vars = [task['end'] for task in bulk_tasks.values()]
            material_ready_time_name = "NOT_CREATED"

            if not bulk_tasks_end_vars:
                print("  [!!] DIAGNOSTIC ERROR: Cannot create a material ready time as no bulk tasks were found.\n")
            else:
                # This logic mirrors the constraint function to see what it *should* do.
                material_ready_time = self.model.NewIntVar(0, self.schedule_horizon, f"mat_ready_{category_name}_debug")
                self.model.AddMinEquality(material_ready_time, bulk_tasks_end_vars)
                material_ready_time_name = material_ready_time.Name()
                print(f"  [OK] Material Ready Time variable '{material_ready_time_name}' defined.")
                print(f"  It is linked to the completion of: {[var.Name() for var in bulk_tasks_end_vars]}")
            print("-" * 20)

            # --- 3. Check for FINISHING tasks and their constraints ---
            print("--- 3. Checking for FINISHING tasks & their constraints ---")
            demand_info = self.aggregated_demand.get(category_name)
            if not demand_info:
                print(f"  [!!] DIAGNOSTIC ERROR: No demand info found for '{category_name}'.")
                continue

            all_finishing_tasks_found = False
            for indent in demand_info['indents']:
                # Finishing tasks are identified by having their key[0] be an order number.
                finishing_tasks = {
                    key: task for key, task in self.task_vars.items()
                    if key[0] == indent.order_no
                }
                if finishing_tasks:
                    all_finishing_tasks_found = True
                    print(f"  For Order No '{indent.order_no}':")
                    for key, task in finishing_tasks.items():
                        start_var_name = task['start'].Name()
                        print(f"    - Finishing Task: {key}, Start Var: {start_var_name}")
                        if material_ready_time_name != "NOT_CREATED":
                            print(f"      - EXPECTED CONSTRAINT: '{start_var_name}' >= '{material_ready_time_name}'")
                        else:
                            print(f"      - EXPECTED CONSTRAINT: SKIPPED (No bulk material ready time).")
            
            if not all_finishing_tasks_found:
                print(f"  [!!] DIAGNOSTIC WARNING: No finishing tasks were found for any orders in this category.")

        print("\n" + "#"*60)
        print("### DIAGNOSTICS COMPLETE ###")
        print("#"*60 + "\n")