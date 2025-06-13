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
from models.data_models import *
import math
from collections import defaultdict

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
        
        self.shift_starts = self._calculate_shift_start_points()
        self.working_windows = self._calculate_working_windows()
        self.schedule_horizon = schedule_horizon
        self.logger = logger
    
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
                    if shift.end_time <= shift.start_time: # This indicates the shift wraps around to the next day
                        shift_end_dt += timedelta(days=1)
                    
                    if shift_start_dt > self.schedule_start:
                        actual_window_start = max(shift_start_dt, self.schedule_start)

                        # Calculate minutes offset from schedule_start
                        start_minutes = int((actual_window_start - self.schedule_start).total_seconds() / 60)
                        end_minutes = int((shift_end_dt - self.schedule_start).total_seconds() / 60) 
                        
                        # Only add the window if it has a positive duration
                        if end_minutes > start_minutes:
                            windows.append((start_minutes, end_minutes))
        

        return sorted(windows)
    
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
        self.cip_circuits = config.CIP_CIRCUIT
        self.equipments = config.EQUIPMENTS
        self.schedule_horizon = schedule_horizon*24*60
        self.time_manager = TimeManager(schedule_start, working_days, holidays, schedule_horizon=schedule_horizon)

        self.weights = {
            "production_volume": 10.0,
            "overproduction_penalty": 1.0,
            "underproduction_penalty_tiers": [ 1, 50, 100, 500 ],  # piecewise <100%, <90%, <70%, <50%
            "setup_time_penalty": 5.0,
            "cip_time_penalty": 20.0,
            "room_utilisation_bonus": 10.0,
            "room_underutilisation_penalty": 10.0,
            "tank_utilisation_bonus": 10.0,
            "tank_underutilisation_penalty_tiers": [10, 50, 200], # piecewise <100%, <75%, <50%
            "idle_time_penalty": 20.0,
            "priority_bonus": 25.0,
            "otif_bonus": 30.0,
            "late_minutes_penalty": 100.0,
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
        self.cip_vars = {}   # Initialize CIP variables dict
        self.cost_vars = {}  # Initialize cost variables dict
        
        # Tracking
        self.solve_start_time = None
        self.warnings = []

        self._is_scheduled = {}
        self.produced_quantity = {}
        
        # Logging
        self.logger = logger

    def schedule_production(self, time_limit: int = 600, max_iterations: int = 5) -> SchedulingResult:
        """
        Schedule production using multi-stage optimization approach.
        
        Args:
            time_limit: Total time limit for optimization in seconds
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            SchedulingResult containing the best solution found
        """
        self.solver_start_time = datetime.now()
        try:
            # Validate setup before starting optimization
            if not self.validate_schedule_setup():
                self.logger.error("Schedule validation failed - cannot proceed")
                return SchedulingResult(
                    status=cp_model.INFEASIBLE,
                    objective_value=0,
                    schedule_items=[],
                    resource_utilization={},
                    production_summary={},
                    solve_time=0,
                    warnings=self.warnings.copy()
                )
            
            # Multi-stage optimization
            best_result = None
            best_score = float('-inf')
            
            # Configure solver parameters for scalability
            self.solver.parameters.num_search_workers = 8
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.random_seed = 42  # For reproducibility
            self.solver.parameters.linearization_level = 2  # Better performance for complex models
            
            for iter in range(max_iterations):
                self._adjust_parameters_for_iterations(iter=iter)

                result = self._solve_iteration(time_limit=(time_limit // max_iterations))
                if result:
                    score = self._calculate_score(result)

                    if score > best_score:
                        best_score = score
                        best_result = result
                        self.logger.info(f'Best score currently: {best_score:.2f}')
                    
                self._reset_model()
            return best_result or None

        except Exception as e:
            self.logger.error(f"Error in scheduling: {e}")
            self.warnings.append(f"Scheduling error: {str(e)}")
            return None

    def _adjust_parameters_for_iterations(self, iter: int):
        """
        Adjusts the weights used in the objective function based on the current optimization iteration.
        """

        if iter == 0:
            # Focus on meeting the target (fulfillment and delivery)
            
            self.weights["overproduction_penalty"]= 1.0
            self.weights["underproduction_penalty_tiers"]= [ 1, 50, 200, 1000 ]  # piecewise <100%, <90%, <70%, <50%
            self.weights["production_volume"]= 20.0
            self.weights["priority_bonus"]= 50.0
            self.weights["otif_bonus"]= 60.0
            self.weights["late_minutes_penalty"]= 200.0

        elif iter == 1:
            # Focus on maximizing utilization and efficiency

            self.weights["production_volume"]= 10.0
            self.weights["overproduction_penalty"]= 5.0
            self.weights["underproduction_penalty_tiers"]= [ 1, 50, 100, 500 ]  # piecewise <100%, <90%, <70%, <50%
            self.weights["setup_time_penalty"]= 5.0
            self.weights["cip_time_penalty"]= 20.0
            self.weights["room_utilisation_bonus"]= 10.0
            self.weights["room_underutilisation_penalty"]= 10.0
            self.weights["tank_utilisation_bonus"]= 20.0
            self.weights["tank_underutilisation_penalty_tiers"]= [10, 50, 500] # piecewise <100%, <75%, <50%
            self.weights["idle_time_penalty"]= 200.0
            self.weights["priority_bonus"]= 25.0
            self.weights["otif_bonus"]= 30.0
            self.weights["late_minutes_penalty"]= 100.0

        else:
            # Balanced optimization
            self.weights["production_volume"]= 10.0
            self.weights["overproduction_penalty"]= 5.0
            self.weights["underproduction_penalty_tiers"]= [ 1, 50, 150, 750 ]  # piecewise <100%, <90%, <70%, <50%
            self.weights["setup_time_penalty"]= 10.0
            self.weights["cip_time_penalty"]= 20.0
            self.weights["room_utilisation_bonus"]= 10.0
            self.weights["room_underutilisation_penalty"]= 10.0
            self.weights["tank_utilisation_bonus"]= 10.0
            self.weights["tank_underutilisation_penalty_tiers"]= [10, 50, 250] # piecewise <100%, <75%, <50%
            self.weights["idle_time_penalty"]= 20.0
            self.weights["priority_bonus"]= 25.0
            self.weights["otif_bonus"]= 30.0
            self.weights["late_minutes_penalty"]= 100.0

    def _solve_iteration(self, time_limit: int) -> Optional[SchedulingResult]:
        try:
            # Pass current datetime for testability
            current_time = datetime.now()
            self.valid_indents = self._sort_indent(current_time)
            
            if not self.valid_indents:
                self.logger.warning("No valid indents to schedule")
                self.warnings.append("No valid orders found for scheduling")
                return None
            
            self._set_variables()
            self._add_constraints()
            self._create_objective()

            self.solver.parameters.max_time_in_seconds = time_limit
            self.solver.parameters.num_search_workers = 8
            self.solver.parameters.log_search_progress = True

            status = self.solver.Solve(self.model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return self._extract_enhanced_solution(status)
            else:
                self.logger.warning(f"Solver status: {self.solver.StatusName(status)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in iteration: {e}")
            self.warnings.append(f"Iteration error: {str(e)}")
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
            return due_date_weight * days_until_due + priority_weight * priority_score
        
        return sorted(valid_indents, key=score)
            
    def _set_variables(self):
        """Fixed version with proper method calls"""
        self.logger.info("Creating enhanced decision variables...")

        indent_list = self.valid_indents.copy()
        for indent in indent_list:
            order_no = indent.order_no
            sku_id = indent.sku_id
            product_id = config.SKUS[sku_id].product_category
            
            self._is_scheduled[order_no] = self.model.NewBoolVar(f'is_scheduled_{order_no}') 
            self._create_batch_and_under_over_vars(indent, config.SKUS[sku_id], config.PRODUCTS[product_id])
            self._create_task_vars(indent, config.SKUS[sku_id], config.PRODUCTS[product_id])
            
        # CIP variables with enhanced modeling
        self._add_cip_constraints()
        # Cost tracking variables
        self._create_cost_variables()

    def _create_batch_and_under_over_vars(self, indent: UserIndent, sku: SKU, product: Product):
        """
        Creates batch production variables, total produced quantity, and under/overproduction penalties.
        
        Args:
            indent: User order indent
            sku: SKU information
            product: Product information with max_batch_size attribute
        """
        order_no = indent.order_no
        # Ensure product has max_batch_size attribute
        batch_size = getattr(product, 'max_batch_size', getattr(product, 'batch_size', 1000))
        required_qty = int(indent.qty_required_liters)
        max_batches = math.ceil(required_qty / batch_size)

        # 1. Create batch-level variables
        batch_vars = []
        for b in range(max_batches):
            var = self.model.NewIntVar(0, batch_size, f"batch_{order_no}_{b}")
            batch_vars.append(var)
        self.batch_qty[order_no] = batch_vars

        # 2. Total produced quantity (can exceed requirement)
        max_possible = batch_size * max_batches
        self.produced_quantity[order_no] = self.model.NewIntVar(0, max_possible, f"produced_qty_{order_no}")
        self.model.Add(self.produced_quantity[order_no] == sum(self.batch_qty[order_no]))

        # 3. Underproduction and overproduction variables
        self.underproduction[order_no] = self.model.NewIntVar(0, required_qty, f"underproduction_{order_no}")
        self.overproduction[order_no] = self.model.NewIntVar(0, max_possible - required_qty, f"overproduction_{order_no}")

        # 4. Delta and penalty logic
        delta = self.model.NewIntVar(-max_possible, max_possible, f"delta_production_{order_no}")
        self.model.Add(delta == self.produced_quantity[order_no] - required_qty)

        self.model.AddMaxEquality(self.underproduction[order_no], [0, required_qty - self.produced_quantity[order_no]])
        self.model.AddMaxEquality(self.overproduction[order_no], [0, self.produced_quantity[order_no] - required_qty])

    def _create_task_vars(self, indent: UserIndent, sku: SKU, product: Product):
        """Create task variables for scheduling optimization."""
        order_no = indent.order_no
        batch_list = self.batch_qty[order_no]
        sku_id = sku.sku_id
        
        for batch_index, batch_var in enumerate(batch_list):
            for step in product.processing_steps:
                task_key = (order_no, sku_id, batch_index, step.step_id)
                
                # Create time variables
                start_var, end_var = self._create_time_vars(order_no, batch_index, step.step_id)
                
                # Create resource variables based on type
                resource_vars, room_vars, selected_resource = self._create_resource_vars(
                    order_no, batch_index, step
                )
                
                # Update resource usage mapping
                self._update_resource_usage_map(
                    resource_vars, task_key, start_var, end_var, sku_id, step.step_id, batch_index
                )
                
                # Store task variables
                self.task_vars[task_key] = {
                    'start': start_var,
                    'end': end_var,
                    'volume': batch_var,
                    'resources': resource_vars,
                    'rooms': room_vars,
                    'sku_id': sku_id,
                    'step_id': step.step_id,
                    'batch_index': batch_index,
                    'priority': indent.priority,
                    'duration': step.duration_minutes_per_batch,
                    'setup_time': step.setup_time,
                    'selected_resource': selected_resource
                }

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
        """Create resource variables based on step resource type."""
        resource_vars = {}
        room_vars = {}
        selected_resource = None
        
        if step.resource_type == ResourceType.ROOM:
            room_vars = self._create_room_vars(order_no, batch_index, step)
        else:
            resource_vars, selected_resource = self._create_equipment_vars(order_no, batch_index, step)
        
        return resource_vars, room_vars, selected_resource

    def _create_room_vars(self, order_no: str, batch_index: int, step: ProcessingStep):
        """Create room selection variables with at-most-one constraint."""
        room_vars = {}
        
        for room_id in step.compatible_resource_ids:
            room_vars[room_id] = self.model.NewBoolVar(
                f'use_room_{order_no}_{batch_index}_{step.step_id}_{room_id}'
            )
        
        if room_vars:
            self.model.AddAtMostOne(room_vars.values())
        
        return room_vars

    def _create_equipment_vars(self, order_no: str, batch_index: int, step: ProcessingStep):
        """Create equipment selection variables with direct resource mapping"""
        resource_vars = {}
        resource_id_map = {}  # Map from boolean var to resource_id
        
        for res_id in step.compatible_resource_ids:
            bool_var = self.model.NewBoolVar(
                f'use_res_{order_no}_{batch_index}_{step.step_id}_{res_id}'
            )
            resource_vars[res_id] = bool_var
            resource_id_map[bool_var] = res_id
        
        selected_resource_id = None
        if resource_vars:
            self.model.AddExactlyOne(resource_vars.values())
            # Store the resource ID mapping for easier lookup
            selected_resource_id = resource_id_map
        
        return resource_vars, selected_resource_id

    def _create_resource_index_var(self, order_no: str, batch_index: int, step_id: str, resource_vars: dict):
        """Create an integer variable representing the selected resource index."""
        selected_resource = self.model.NewIntVar(
            0, len(resource_vars) - 1, f'selected_res_idx_{order_no}_{batch_index}_{step_id}'
        )
        
        # Link the index to the boolean variables
        resource_list = list(resource_vars.values())
        self.model.Add(selected_resource == sum(
            idx * var for idx, var in enumerate(resource_list)
        ))
        
        return selected_resource

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

    def _add_cip_constraints(self):
        self.logger.info("Adding CIP constraints...")
        cip_counter = 0

        for resource_id, task_list in self.resource_usage_map.items():
            # Sort by start time via AddNoOverlap or sequencing constraints later
            for i in range(len(task_list)):
                for j in range(i+1, len(task_list)):
                    t1, t2 = task_list[i], task_list[j]
                    # Only if both tasks are actually assigned to this resource
                    self.model.AddImplication(t1['assign_var'], t2['assign_var'])  # Optional

                    if self._needs_cip_between(t1, t2, resource_id):
                        self._create_cip_task(resource_id, t1, t2, cip_counter)
                        cip_counter += 1

    def _create_cost_variables(self):
        """Create cost tracking variables"""
        # Total cost components
        self.cost_vars = {
            'production_cost': self.model.NewIntVar(0, 1000000, 'production_cost'),
            'setup_cost': self.model.NewIntVar(0, 100000, 'setup_cost'),
            'cip_cost': self.model.NewIntVar(0, 50000, 'cip_cost'),
            'quality_cost': self.model.NewIntVar(0, 50000, 'quality_cost'),
            'total_cost': self.model.NewIntVar(0, 1200000, 'total_cost')
        }

    def _add_constraints(self):
        """Add all scheduling constraints"""
        self.logger.info("Adding scheduling constraints...")
        
        # 1. Task sequencing within batches
        self._add_task_sequencing_constraints()
        
        # 2. Resource capacity constraints
        self._add_resource_capacity_constraints()
        
        # 3. Time window constraints
        self._add_time_window_constraints()
        
        # 4. Batch ordering constraints
        self._add_batch_ordering_constraints()
        
        # 5. Production quantity constraints
        self._add_production_quantity_constraints()
        
        # 6. Due date constraints
        self._add_due_date_constraints()

    def _add_task_sequencing_constraints(self):
        """Ensure tasks within a batch follow the correct sequence"""
        for order_no in self.batch_qty.keys():
            for batch_index in range(len(self.batch_qty[order_no])):
                # Get all tasks for this batch
                batch_tasks = []
                for task_key, task_data in self.task_vars.items():
                    if task_key[0] == order_no and task_key[2] == batch_index:
                        batch_tasks.append((task_key, task_data))
                
                # Sort by step sequence
                batch_tasks.sort(key=lambda x: self._get_step_sequence(x[0][3]))
                
                # Add sequencing constraints
                for i in range(len(batch_tasks) - 1):
                    current_task = batch_tasks[i][1]
                    next_task = batch_tasks[i + 1][1]
                    # Next task can only start after current task ends
                    self.model.Add(next_task['start'] >= current_task['end'])

    def _get_step_sequence(self, step_id: str) -> int:
        """Get the sequence number of a processing step"""
        # This assumes you have a way to determine step sequence
        # You may need to modify based on your actual data structure
        for product in config.PRODUCTS.values():
            for i, step in enumerate(product.processing_steps):
                if step.step_id == step_id:
                    return i
        return 0

    def _add_resource_capacity_constraints(self):
        """Add no-overlap constraints for resources"""
        for resource_id, task_list in self.resource_usage_map.items():
            if len(task_list) <= 1:
                continue
                
            # Create interval variables for tasks using this resource
            intervals = []
            for task_info in task_list:
                # Create interval only when task is assigned to this resource
                interval_var = self.model.NewOptionalIntervalVar(
                    task_info['start'],
                    task_info['end'] - task_info['start'],
                    task_info['end'],
                    task_info['assign_var'],
                    f"interval_{resource_id}_{task_info['task_key']}"
                )
                intervals.append(interval_var)
            
            # No overlap constraint
            self.model.AddNoOverlap(intervals)

    def _add_time_window_constraints(self):
        """Ensure tasks are scheduled within working hours using inverse intervals approach"""
        # Check if we have valid shift starts and working windows
        if not self.time_manager.shift_starts:
            self.logger.warning("No valid shift start points found. Tasks may not be properly constrained.")
            self.warnings.append("No valid shift start points - scheduling may be unrestricted")
        
        if not self.time_manager.working_windows:
            self.logger.warning("No working time windows found. All tasks will be unconstrained by working hours.")
            self.warnings.append("No working time windows - tasks not constrained by working hours")
            # Still add basic duration constraints even without working windows
            for task_key, task_data in self.task_vars.items():
                self.model.Add(task_data['end'] == task_data['start'] + task_data['duration'])
            return
        
        # Create non-working time intervals (inverse approach)
        non_working_intervals = self._create_non_working_intervals()
        
        for task_key, task_data in self.task_vars.items():
            # Tasks can only start at valid shift start points (if available)
            if self.time_manager.shift_starts:
                start_options = []
                for start_point in self.time_manager.shift_starts:
                    if start_point + task_data['duration'] <= self.schedule_horizon:
                        start_options.append(start_point)
                
                if start_options:
                    # Task start must be one of the valid start points
                    self.model.AddAllowedAssignments(
                        [task_data['start']], 
                        [(point,) for point in start_options]
                    )
                else:
                    self.logger.warning(f"No valid start options for task {task_key}")
                    self.warnings.append(f"No valid start options for task {task_key}")
            
            # Duration constraint (always add this)
            self.model.Add(task_data['end'] == task_data['start'] + task_data['duration'])
            
            # Create task interval
            task_interval = self.model.NewIntervalVar(
                task_data['start'],
                task_data['duration'],
                task_data['end'],
                f"task_interval_{task_key}"
            )
            
            # Ensure task doesn't overlap with non-working intervals
            if non_working_intervals:
                all_intervals = non_working_intervals + [task_interval]
                self.model.AddNoOverlap(all_intervals)

    def _add_batch_ordering_constraints(self):
        """Ensure batches are processed in order within each order"""
        for order_no, batch_list in self.batch_qty.items():
            if len(batch_list) <= 1:
                continue
                
            for batch_idx in range(len(batch_list) - 1):
                # Get last task of current batch and first task of next batch
                current_batch_tasks = self._get_batch_tasks(order_no, batch_idx)
                next_batch_tasks = self._get_batch_tasks(order_no, batch_idx + 1)
                
                if current_batch_tasks and next_batch_tasks:
                    # Last task of current batch must finish before first task of next batch
                    current_last = max(current_batch_tasks, key=lambda x: self._get_step_sequence(x[0][3]))
                    next_first = min(next_batch_tasks, key=lambda x: self._get_step_sequence(x[0][3]))
                    
                    self.model.Add(next_first[1]['start'] >= current_last[1]['end'])

    def _get_batch_tasks(self, order_no: str, batch_index: int) -> List[Tuple]:
        """Get all tasks for a specific batch"""
        batch_tasks = []
        for task_key, task_data in self.task_vars.items():
            if task_key[0] == order_no and task_key[2] == batch_index:
                batch_tasks.append((task_key, task_data))
        return batch_tasks

    def _add_production_quantity_constraints(self):
        """Add constraints for production quantities"""
        for order_no in self.batch_qty.keys():
            # Only produce if scheduled
            for batch_var in self.batch_qty[order_no]:
                scheduled_var = self._is_scheduled[order_no]
                # If not scheduled, batch quantity must be 0
                self.model.Add(batch_var == 0).OnlyEnforceIf(scheduled_var.Not())

    def _add_due_date_constraints(self):
        """Add soft constraints for due dates"""
        for indent in self.valid_indents:
            order_no = indent.order_no
            due_date_minutes = int((indent.due_date - self.time_manager.schedule_start).total_seconds() / 60)
            
            if due_date_minutes > 0:
                # Get the last task for this order
                last_task_end = self._get_order_completion_time(order_no)
                if last_task_end:
                    # Create tardiness variable
                    tardiness = self.model.NewIntVar(0, self.schedule_horizon, f"tardiness_{order_no}")
                    self.model.AddMaxEquality(tardiness, [0, last_task_end - due_date_minutes])

    def _get_order_completion_time(self, order_no: str):
        """Get the completion time variable for an order"""
        order_tasks = [(k, v) for k, v in self.task_vars.items() if k[0] == order_no]
        if not order_tasks:
            return None
        
        # Return the end time of the last task (highest batch index, highest step sequence)
        last_task = max(order_tasks, key=lambda x: (x[0][2], self._get_step_sequence(x[0][3])))
        return last_task[1]['end']

    def _create_objective(self):
        """Create the comprehensive objective function with all weight components"""
        self.logger.info("Creating enhanced objective function...")
        
        objective_terms = []
        
        # 1. Production volume bonus
        for order_no, produced_var in self.produced_quantity.items():
            objective_terms.append(produced_var * int(self.weights["production_volume"]))
        
        # 2. Enhanced under/overproduction penalties with proper piecewise handling
        for order_no in self.underproduction.keys():
            indent = next((i for i in self.valid_indents if i.order_no == order_no), None)
            if indent:
                required_qty = int(indent.qty_required_liters)
                self._add_piecewise_underproduction_penalty(order_no, required_qty, objective_terms)
            
            # Overproduction penalty
            over_var = self.overproduction[order_no]
            objective_terms.append(-over_var * int(self.weights["overproduction_penalty"]))
        
        # 3. Setup and CIP time penalties
        for task_key, task_data in self.task_vars.items():
            if task_data.get('setup_time', 0) > 0:
                objective_terms.append(-task_data['setup_time'] * int(self.weights["setup_time_penalty"]))
        
        for cip_id, cip_data in self.cip_vars.items():
            cip_duration = cip_data['end'] - cip_data['start']
            objective_terms.append(-cip_duration * int(self.weights["cip_time_penalty"]))
        
        # 4. Resource utilization bonuses and penalties
        self._add_resource_utilization_objectives(objective_terms)
        
        # 5. Idle time penalty
        self._add_idle_time_penalty(objective_terms)
        
        # 6. Priority bonus
        for indent in self.valid_indents:
            order_no = indent.order_no
            if order_no in self._is_scheduled:
                priority_multiplier = indent.priority.value
                scheduled_var = self._is_scheduled[order_no]
                objective_terms.append(scheduled_var * priority_multiplier * int(self.weights["priority_bonus"]))
        
        # 7. Enhanced OTIF bonus with partial credit
        self._add_otif_objectives(objective_terms)
        
        # 8. Late delivery penalty
        for indent in self.valid_indents:
            order_no = indent.order_no
            completion_time = self._get_order_completion_time(order_no)
            due_date_minutes = int((indent.due_date - self.time_manager.schedule_start).total_seconds() / 60)
            
            if completion_time and due_date_minutes > 0:
                lateness = self.model.NewIntVar(0, self.schedule_horizon, f"lateness_{order_no}")
                self.model.AddMaxEquality(lateness, [0, completion_time - due_date_minutes])
                objective_terms.append(-lateness * int(self.weights["late_minutes_penalty"]))
        
        # Set the objective
        if objective_terms:
            self.model.Maximize(sum(objective_terms))
        else:
            self.logger.warning("No objective terms created!")

    def _add_piecewise_underproduction_penalty(self, order_no: str, required_qty: int, objective_terms: list):
        """Add piecewise linear underproduction penalty"""
        under_var = self.underproduction[order_no]
        penalty_tiers = self.weights["underproduction_penalty_tiers"]
        
        # Create breakpoints: 100%, 90%, 70%, 50% of required quantity
        breakpoints = [0, int(0.1 * required_qty), int(0.3 * required_qty), int(0.5 * required_qty)]
        
        for i, penalty in enumerate(penalty_tiers):
            if i < len(breakpoints) - 1:
                # Penalty for underprodction in this tier
                tier_under = self.model.NewIntVar(0, breakpoints[i+1] - breakpoints[i], 
                                                f"tier_under_{order_no}_{i}")
                # Constrain tier underprodction
                self.model.AddMaxEquality(tier_under, [0, 
                    self.model.NewConstant(min(breakpoints[i+1], required_qty)) - 
                    self.model.NewConstant(breakpoints[i]) - 
                    (under_var - self.model.NewConstant(breakpoints[i]))])
                
                objective_terms.append(-tier_under * penalty)

    def _add_resource_utilization_objectives(self, objective_terms: list):
        """Add resource utilization bonuses and penalties"""
        # Room utilization
        for room_id, room in config.ROOMS.items():
            total_room_time = self._calculate_resource_utilization(room_id)
            if total_room_time:
                # Bonus for high utilization
                objective_terms.append(total_room_time * int(self.weights["room_utilisation_bonus"]))
                
                # Penalty for underutilization
                available_time = self._get_available_time_for_resource(room_id)
                if available_time:
                    underutil = self.model.NewIntVar(0, available_time, f"room_underutil_{room_id}")
                    self.model.AddMaxEquality(underutil, [0, available_time - total_room_time])
                    objective_terms.append(-underutil * int(self.weights["room_underutilisation_penalty"]))
        
        # Tank utilization with piecewise penalties
        for tank_id, tank in config.TANKS.items():
            total_tank_time = self._calculate_resource_utilization(tank_id)
            if total_tank_time:
                objective_terms.append(total_tank_time * int(self.weights["tank_utilisation_bonus"]))
                
                # Piecewise underutilization penalty
                available_time = self._get_available_time_for_resource(tank_id)
                if available_time:
                    self._add_piecewise_tank_underutilization_penalty(tank_id, available_time, 
                                                                    total_tank_time, objective_terms)

    def _add_piecewise_tank_underutilization_penalty(self, tank_id: str, available_time, 
                                                used_time, objective_terms: list):
        """Add piecewise tank underutilization penalty"""
        penalty_tiers = self.weights["tank_underutilisation_penalty_tiers"]
        
        # Create breakpoints: 100%, 75%, 50% utilization
        breakpoints = [available_time, int(0.75 * available_time), int(0.5 * available_time), 0]
        
        for i, penalty in enumerate(penalty_tiers):
            if i < len(breakpoints) - 1:
                tier_underutil = self.model.NewIntVar(0, breakpoints[i] - breakpoints[i+1], 
                                                    f"tank_underutil_{tank_id}_{i}")
                # Logic for piecewise underutilization
                self.model.AddMaxEquality(tier_underutil, [0, 
                    min(breakpoints[i], available_time) - max(breakpoints[i+1], used_time)])
                
                objective_terms.append(-tier_underutil * penalty)

    def _add_idle_time_penalty(self, objective_terms: list):
        """Add idle time penalty for resources"""
        for resource_id, task_list in self.resource_usage_map.items():
            if len(task_list) > 1:
                # Calculate idle time between consecutive tasks
                for i in range(len(task_list) - 1):
                    idle_time = self.model.NewIntVar(0, self.schedule_horizon, 
                                                f"idle_{resource_id}_{i}")
                    # Idle time is the gap between end of task i and start of task i+1
                    self.model.Add(idle_time == task_list[i+1]['start'] - task_list[i]['end'])
                    objective_terms.append(-idle_time * int(self.weights["idle_time_penalty"]))

    def _add_otif_objectives(self, objective_terms: list):
        """Add enhanced OTIF objectives with partial credit"""
        for indent in self.valid_indents:
            order_no = indent.order_no
            if order_no in self.produced_quantity:
                required_qty = int(indent.qty_required_liters)
                produced_var = self.produced_quantity[order_no]
                
                # Full OTIF bonus
                due_date_minutes = int((indent.due_date - self.time_manager.schedule_start).total_seconds() / 60)
                completion_time = self._get_order_completion_time(order_no)
                
                if completion_time and due_date_minutes > 0:
                    # Full OTIF
                    full_otif = self.model.NewBoolVar(f"full_otif_{order_no}")
                    self.model.Add(produced_var >= required_qty).OnlyEnforceIf(full_otif)
                    self.model.Add(completion_time <= due_date_minutes).OnlyEnforceIf(full_otif)
                    objective_terms.append(full_otif * int(self.weights["otif_bonus"]))
                    
                    # Partial OTIF (delivered on time but not full quantity)
                    partial_otif = self.model.NewBoolVar(f"partial_otif_{order_no}")
                    self.model.Add(produced_var >= int(0.8 * required_qty)).OnlyEnforceIf(partial_otif)
                    self.model.Add(produced_var < required_qty).OnlyEnforceIf(partial_otif)
                    self.model.Add(completion_time <= due_date_minutes).OnlyEnforceIf(partial_otif)
                    objective_terms.append(partial_otif * int(self.weights["otif_bonus"] * 0.5))

    def _calculate_resource_utilization(self, resource_id: str):
        """Calculate total utilization time for a resource"""
        if resource_id not in self.resource_usage_map:
            return self.model.NewConstant(0)
        
        total_time = self.model.NewIntVar(0, self.schedule_horizon, f"total_time_{resource_id}")
        
        # Sum up all task durations assigned to this resource
        task_durations = []
        for task_info in self.resource_usage_map[resource_id]:
            duration = task_info['end'] - task_info['start']
            # Only count if task is actually assigned to this resource
            assigned_duration = self.model.NewIntVar(0, self.schedule_horizon, 
                                                    f"assigned_duration_{resource_id}_{task_info['task_key']}")
            self.model.Add(assigned_duration == duration).OnlyEnforceIf(task_info['assign_var'])
            self.model.Add(assigned_duration == 0).OnlyEnforceIf(task_info['assign_var'].Not())
            task_durations.append(assigned_duration)
        
        if task_durations:
            self.model.Add(total_time == sum(task_durations))
        else:
            self.model.Add(total_time == 0)
        
        return total_time

    def _get_available_time_for_resource(self, resource_id: str) -> int:
        """Get total available time for a resource within the schedule horizon"""
        # This is a simplified calculation - in reality, you'd consider working windows
        # For now, assume resource is available during all working windows
        total_available = 0
        for start_time, end_time in self.time_manager.working_windows:
            total_available += (end_time - start_time)
        return total_available
    
    def _extract_enhanced_solution(self, status) -> SchedulingResult:
        """Extract comprehensive solution from the solved model"""
        self.logger.info("Extracting solution...")
        
        scheduled_tasks = []
        resource_utilization = defaultdict(list)
        production_summary = {}
        
        # Extract task schedules
        for task_key, task_data in self.task_vars.items():
            order_no, sku_id, batch_index, step_id = task_key
            
            start_time = self.solver.Value(task_data['start'])
            end_time = self.solver.Value(task_data['end'])
            volume = self.solver.Value(task_data['volume'])
            
            # Find assigned resource
            assigned_resource = None
            for resource_id, resource_var in task_data['resources'].items():
                if self.solver.BooleanValue(resource_var):
                    assigned_resource = resource_id
                    break
            
            if volume > 0 and assigned_resource:  # Only include tasks with positive volume
                task_schedule = TaskSchedule(
                    task_id=f"{order_no}_{batch_index}_{step_id}",
                    order_no=order_no,
                    sku_id=sku_id,
                    batch_index=batch_index,
                    step_id=step_id,
                    start_time=self.time_manager.schedule_start + timedelta(minutes=start_time),
                    end_time=self.time_manager.schedule_start + timedelta(minutes=end_time),
                    resource_id=assigned_resource,
                    volume=volume,
                    priority=task_data['priority']
                )
                scheduled_tasks.append(task_schedule)
                
                # Track resource utilization
                resource_utilization[assigned_resource].append({
                    'start': start_time,
                    'end': end_time,
                    'task_id': task_schedule.task_id,
                    'volume': volume
                })
        
        # Extract production summary
        for order_no in self.produced_quantity.keys():
            produced_qty = self.solver.Value(self.produced_quantity[order_no])
            is_scheduled = self.solver.BooleanValue(self._is_scheduled[order_no])
            
            if is_scheduled:
                production_summary[order_no] = {
                    'produced_quantity': produced_qty,
                    'underproduction': self.solver.Value(self.underproduction[order_no]),
                    'overproduction': self.solver.Value(self.overproduction[order_no]),
                    'scheduled': True
                }
        
        # Calculate metrics
        total_solve_time = (datetime.now() - self.solver_start_time).total_seconds()
        objective_value = self.solver.ObjectiveValue() if status == cp_model.OPTIMAL else self.solver.BestObjectiveBound()
        
        return SchedulingResult(
            status=status,
            objective_value=objective_value,
            schedule_items=scheduled_tasks,
            resource_utilization=dict(resource_utilization),
            production_summary=production_summary,
            solve_time=total_solve_time,
            warnings=self.warnings.copy()
        )

    def _calculate_score(self, result: SchedulingResult) -> float:
        """Calculate a comprehensive score for the solution"""
        if not result or not result.schedule_items:
            return float('-inf')
        
        score = 0.0
        
        # Base score from objective value
        score += result.objective_value
        
        # Bonus for number of scheduled orders
        scheduled_orders = len(result.production_summary)
        score += scheduled_orders * 100
        
        # Penalty for warnings
        score -= len(result.warnings) * 50
        
        # Resource utilization score
        total_utilization = 0
        for resource_id, tasks in result.resource_utilization.items():
            if tasks:
                total_time = sum(task['end'] - task['start'] for task in tasks)
                utilization_rate = total_time / self.schedule_horizon
                total_utilization += utilization_rate
        
        avg_utilization = total_utilization / len(result.resource_utilization) if result.resource_utilization else 0
        score += avg_utilization * 1000
        
        return score

    def _needs_cip_between(self, prev_task, next_task, resource_id: str) -> bool:
        resource_type = self._get_resource_type(resource_id)

        # Tanks: Always require CIP between uses
        if resource_type == ResourceType.TANK:
            return True

        # Lines: Only if the SKU or variant changes (customize as needed)
        if resource_type == ResourceType.LINE:
            return prev_task['sku_id'] != next_task['sku_id']

        # Other resources: Customize more if needed
        return False
    
    def _get_resource_type(self, resource_id: str) -> ResourceType:
        """Get resource type with graceful error handling"""
        try:
            if resource_id in config.TANKS:
                return ResourceType.TANK
            elif resource_id in config.LINES:
                return ResourceType.LINE
            elif resource_id in config.EQUIPMENTS:
                return ResourceType.EQUIPMENT
            elif resource_id in config.ROOMS:
                return ResourceType.ROOM
            else:
                self.logger.warning(f"Unknown resource ID: {resource_id}. Defaulting to EQUIPMENT type.")
                self.warnings.append(f"Unknown resource ID: {resource_id}")
                return ResourceType.EQUIPMENT  # Default fallback
        except Exception as e:
            self.logger.error(f"Error determining resource type for {resource_id}: {e}")
            self.warnings.append(f"Error with resource {resource_id}: {str(e)}")
            return ResourceType.EQUIPMENT
        
    def _create_cip_task(self, resource_id: str, prev_task: dict, next_task: dict, cip_index: int):
        cip_id = f"cip_{cip_index}_{resource_id}"

        cip_time = self._get_cip_time(resource_id)
        
        cip_start = self.model.NewIntVar(0, self.schedule_horizon, f"{cip_id}_start")
        cip_end = self.model.NewIntVar(0, self.schedule_horizon, f"{cip_id}_end")

        self.model.Add(cip_end == cip_start + cip_time)
        self.model.Add(cip_start >= prev_task['end'])
        self.model.Add(next_task['start'] >= cip_end)

        # Optional: Track CIP effectiveness etc.
        self.cip_vars[cip_id] = {
            'start': cip_start,
            'end': cip_end,
            'resource': resource_id,
            'between': (prev_task['task_key'], next_task['task_key']),
        }

    def _get_cip_time(self, resource_id: str) -> int:
        """Get CIP duration for a resource with standardized naming"""
        resource_type = self._get_resource_type(resource_id)
        if resource_type == ResourceType.TANK:
            return config.TANKS[resource_id].cip_duration_minutes
        elif resource_type == ResourceType.LINE:
            # Standardize to use cip_duration_minutes
            return getattr(config.LINES[resource_id], 'cip_duration_minutes', 
                          getattr(config.LINES[resource_id], 'cip_duration', 60))
        elif resource_type == ResourceType.EQUIPMENT:
            return config.EQUIPMENTS[resource_id].cip_duration_minutes
        return 60  # fallback default

    def validate_schedule_setup(self) -> bool:
        """Validate that the scheduling setup is viable"""
        validation_passed = True
        
        # Check if we have any working windows
        if not self.time_manager.working_windows:
            self.logger.error("No working time windows available - scheduling impossible")
            self.warnings.append("Critical: No working time windows available")
            validation_passed = False
        
        # Check if we have any shift start points
        if not self.time_manager.shift_starts:
            self.logger.error("No valid shift start points - tasks cannot be scheduled")
            self.warnings.append("Critical: No valid shift start points")
            validation_passed = False
        
        # Check if we have valid indents
        if not hasattr(self, 'valid_indents') or not self.valid_indents:
            self.logger.error("No valid orders to schedule")
            self.warnings.append("Critical: No valid orders available")
            validation_passed = False
        
        return validation_passed
    
    def _create_non_working_intervals(self) -> List:
        """Create intervals for non-working times to use with AddNoOverlap constraint"""
        non_working_intervals = []
        
        if not self.time_manager.working_windows:
            return non_working_intervals
        
        # Sort working windows by start time
        sorted_windows = sorted(self.time_manager.working_windows)
        
        # Create non-working intervals between working windows
        current_time = 0
        
        for start_time, end_time in sorted_windows:
            # Add non-working interval before this working window
            if current_time < start_time:
                non_working_start = self.model.NewConstant(current_time)
                non_working_size = self.model.NewConstant(start_time - current_time)
                non_working_end = self.model.NewConstant(start_time)
                
                non_working_interval = self.model.NewIntervalVar(
                    non_working_start,
                    non_working_size,
                    non_working_end,
                    f"non_working_{current_time}_{start_time}"
                )
                non_working_intervals.append(non_working_interval)
            
            current_time = max(current_time, end_time)

    def _reset_model(self):
            """Reset the model for next iteration"""
            self.model = cp_model.CpModel()
            self.solver = cp_model.CpSolver()
            self.resource_usage_map = defaultdict(list)
            self.task_vars = {}
            self.batch_qty = {}
            self.produced_quantity = {}
            self.underproduction = {}
            self.overproduction = {}
            self._is_scheduled = {}
            self.cip_vars = {}
            self.cost_vars = {}
            self.warnings = []
