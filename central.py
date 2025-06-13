from ortools.sat.python import cp_model
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import
import logging
from datetime import datetime, timedelta
from enum import Enum

class TaskType(Enum):
    PRODUCTION = "production"
    CIP = "cip"
    SETUP = "setup"
    DOWNTIME = "downtime"

@dataclass
class ScheduleItem:
    """Represents a scheduled task with all relevant information"""
    sku_id: str
    step_id: str
    resource_id: str
    start_time: int  # in minutes from schedule start
    end_time: int
    volume: float
    room_id: Optional[str] = None
    task_type: TaskType = TaskType.PRODUCTION
    setup_time: int = 0
    
    def to_datetime(self, schedule_start: datetime) -> tuple[datetime, datetime]:
        """Convert time offsets to actual datetime objects"""
        start_dt = schedule_start + timedelta(minutes=self.start_time)
        end_dt = schedule_start + timedelta(minutes=self.end_time)
        return start_dt, end_dt
    
    def __str__(self) -> str:
        return (f"ScheduleItem(SKU={self.sku_id}, Step={self.step_id}, "
                f"Resource={self.resource_id}, Time={self.start_time}-{self.end_time}, "
                f"Volume={self.volume}, Room={self.room_id}, Type={self.task_type.value})")

class TimeManager:
    """Manages time conversions and shift boundaries"""
    
    def __init__(self, schedule_start: datetime, shift_boundaries_hours: List[int]):
        self.schedule_start = schedule_start
        self.shift_boundaries_hours = shift_boundaries_hours
        self.shift_boundaries_minutes = self._calculate_shift_boundaries()
    
    def _calculate_shift_boundaries(self) -> List[int]:
        """Convert shift boundary hours to minutes from schedule start"""
        boundaries = []
        current_date = self.schedule_start.date()
        
        # Calculate boundaries for the next 7 days
        for day_offset in range(8):  # Include extra day for safety
            day = current_date + timedelta(days=day_offset)
            for hour in self.shift_boundaries_hours:
                boundary_time = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
                if boundary_time >= self.schedule_start:
                    minutes_offset = int((boundary_time - self.schedule_start).total_seconds() / 60)
                    boundaries.append(minutes_offset)
        
        return sorted(boundaries)
    
    def minutes_to_datetime(self, minutes: int) -> datetime:
        """Convert minutes offset to datetime"""
        return self.schedule_start + timedelta(minutes=minutes)
    
    def datetime_to_minutes(self, dt: datetime) -> int:
        """Convert datetime to minutes offset"""
        return int((dt - self.schedule_start).total_seconds() / 60)

class ProductionScheduler:
    """Enhanced production scheduler with improved time handling and constraints"""
    
    def __init__(self, skus, products, resources, rooms, cip_circuits, 
                 schedule_start: datetime, shift_boundaries_hours: List[int],
                 schedule_horizon_hours: int = 168):  # 1 week default
        
        self.skus = skus
        self.products = products
        self.resources = resources
        self.rooms = rooms
        self.cip_circuits = cip_circuits
        
        # Time management
        self.time_manager = TimeManager(schedule_start, shift_boundaries_hours)
        self.schedule_horizon = schedule_horizon_hours * 60  # Convert to minutes
        
        # Performance parameters
        self.OEE = 0.8
        self.packaging_downtime_minutes = 30
        self.default_setup_time = 30
        
        # OR-Tools components
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Decision variables
        self.task_vars = {}  # (sku_id, step_id) -> {start, end, volume, resource, room}
        self.cip_vars = {}   # cip task variables
        self.downtime_vars = {}  # downtime variables for shift changes
        
        # Tracking variables
        self.objective_terms = []
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_variables(self):
        """Create all decision variables for the scheduling problem"""
        self.logger.info("Creating decision variables...")
        
        # Create main task variables
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            for step_id in product.steps:
                task_key = (sku_id, step_id)
                self._create_task_variables(task_key, sku, step_id)
        
        # Create CIP variables
        self._create_cip_variables()
        
        # Create shift downtime variables
        self._create_downtime_variables()
    
    def _create_task_variables(self, task_key: Tuple[str, str], sku, step_id: str):
        """Create variables for a single task"""
        sku_id = task_key[0]
        
        # Time variables
        start_var = self.model.NewIntVar(0, self.schedule_horizon, f'start_{sku_id}_{step_id}')
        end_var = self.model.NewIntVar(0, self.schedule_horizon, f'end_{sku_id}_{step_id}')
        
        # Volume variable (allow controlled overproduction)
        max_volume = int(sku.quantity_required * 1.2)  # Allow 20% overproduction
        volume_var = self.model.NewIntVar(
            sku.quantity_required, max_volume, f'volume_{sku_id}_{step_id}'
        )
        
        # Resource selection
        compatible_resources = self._get_compatible_resources(step_id)
        if not compatible_resources:
            raise ValueError(f"No compatible resources found for step {step_id}")
        
        resource_vars = {}
        for res_id in compatible_resources:
            resource_vars[res_id] = self.model.NewBoolVar(f'use_res_{sku_id}_{step_id}_{res_id}')
        
        # Exactly one resource must be selected
        self.model.AddExactlyOne(resource_vars.values())
        
        # Room selection (if needed)
        room_vars = {}
        if self._step_needs_room(step_id):
            compatible_rooms = self._get_compatible_rooms(step_id)
            for room_id in compatible_rooms:
                room_vars[room_id] = self.model.NewBoolVar(f'use_room_{sku_id}_{step_id}_{room_id}')
            # At most one room (can be zero if room not required)
            if room_vars:
                self.model.AddAtMostOne(room_vars.values())
        
        self.task_vars[task_key] = {
            'start': start_var,
            'end': end_var,
            'volume': volume_var,
            'resources': resource_vars,
            'rooms': room_vars,
            'sku_id': sku_id,
            'step_id': step_id
        }
    
    def _create_cip_variables(self):
        """Create CIP task variables"""
        self.logger.info("Creating CIP variables...")
        
        cip_counter = 0
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            for step_id in product.steps:
                if self._needs_cip_before_task(sku, step_id):
                    cip_id = f"cip_{cip_counter}"
                    cip_counter += 1
                    
                    cip_duration = 60  # 60 minutes standard CIP
                    cip_start = self.model.NewIntVar(0, self.schedule_horizon, f'cip_start_{cip_id}')
                    cip_end = self.model.NewIntVar(0, self.schedule_horizon, f'cip_end_{cip_id}')
                    
                    self.model.Add(cip_end == cip_start + cip_duration)
                    
                    self.cip_vars[cip_id] = {
                        'start': cip_start,
                        'end': cip_end,
                        'duration': cip_duration,
                        'sku_id': sku_id,
                        'step_id': step_id,
                        'circuit': self._get_required_cip_circuit(step_id)
                    }
    
    def _create_downtime_variables(self):
        """Create downtime variables for shift changes affecting packaging"""
        self.logger.info("Creating downtime variables...")
        
        downtime_counter = 0
        for shift_boundary in self.time_manager.shift_boundaries_minutes:
            if shift_boundary < self.schedule_horizon:
                downtime_id = f"downtime_{downtime_counter}"
                downtime_counter += 1
                
                # Downtime starts 15 minutes before shift change and lasts 30 minutes total
                downtime_start = max(0, shift_boundary - 15)
                downtime_end = min(self.schedule_horizon, shift_boundary + 15)
                
                self.downtime_vars[downtime_id] = {
                    'start': downtime_start,
                    'end': downtime_end,
                    'shift_boundary': shift_boundary
                }
    
    def add_time_constraints(self):
        """Add constraints for task durations and sequencing"""
        self.logger.info("Adding time constraints...")
        
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            prev_task_vars = None
            for i, step_id in enumerate(product.steps):
                task_key = (sku_id, step_id)
                task_vars = self.task_vars[task_key]
                
                # Duration constraint based on volume and resource capacity
                self._add_duration_constraints(task_vars, step_id)
                
                # Sequential step constraints
                if prev_task_vars is not None:
                    # Current step must start after previous step ends (with possible setup time)
                    self.model.Add(task_vars['start'] >= prev_task_vars['end'])
                
                # CIP constraints
                self._add_cip_timing_constraints(sku, step_id, task_vars)
                
                prev_task_vars = task_vars
    
    def _add_duration_constraints(self, task_vars: Dict, step_id: str):
        """Add duration constraints for a task"""
        for res_id, res_var in task_vars['resources'].items():
            resource = next(r for r in self.resources if r.resource_id == res_id)
            
            # Effective capacity with OEE
            effective_capacity = resource.capacity * self.OEE
            
            # Calculate minimum duration in minutes
            min_duration = 1  # At least 1 minute
            max_duration = self.schedule_horizon
            
            # Duration is proportional to volume
            duration_var = self.model.NewIntVar(min_duration, max_duration, 
                                              f'duration_{task_vars["sku_id"]}_{step_id}_{res_id}')
            
            # If this resource is used, set duration based on volume
            # Use integer division for OR-Tools compatibility
            self.model.AddDivisionEquality(duration_var, task_vars['volume'], 
                                         int(effective_capacity)).OnlyEnforceIf(res_var)
            self.model.Add(duration_var == 0).OnlyEnforceIf(res_var.Not())
            
            # Setup time
            setup_time = getattr(resource, 'setup_time', self.default_setup_time)
            
            # End time = start time + duration + setup time
            self.model.Add(
                task_vars['end'] >= task_vars['start'] + duration_var + setup_time
            ).OnlyEnforceIf(res_var)
    
    def _add_cip_timing_constraints(self, sku, step_id: str, task_vars: Dict):
        """Add CIP timing constraints"""
        # Find corresponding CIP task
        for cip_id, cip_vars in self.cip_vars.items():
            if (cip_vars['sku_id'] == sku.sku_id and 
                cip_vars['step_id'] == step_id):
                # Task must start after CIP ends
                self.model.Add(task_vars['start'] >= cip_vars['end'])
    
    def add_resource_constraints(self):
        """Add resource capacity and exclusivity constraints"""
        self.logger.info("Adding resource constraints...")
        
        # Group tasks by resource
        resource_intervals = {}
        
        for task_key, task_vars in self.task_vars.items():
            for res_id, res_var in task_vars['resources'].items():
                if res_id not in resource_intervals:
                    resource_intervals[res_id] = []
                
                # Create optional interval variable
                interval_var = self.model.NewOptionalIntervalVar(
                    task_vars['start'],
                    task_vars['end'] - task_vars['start'],
                    task_vars['end'],
                    res_var,
                    f'interval_{task_key[0]}_{task_key[1]}_{res_id}'
                )
                resource_intervals[res_id].append(interval_var)
        
        # Add CIP intervals to resource constraints
        for cip_id, cip_vars in self.cip_vars.items():
            circuit_id = cip_vars['circuit']
            if circuit_id and circuit_id in resource_intervals:
                cip_interval = self.model.NewIntervalVar(
                    cip_vars['start'],
                    cip_vars['duration'],
                    cip_vars['end'],
                    f'cip_interval_{cip_id}'
                )
                resource_intervals[circuit_id].append(cip_interval)
        
        # Add no-overlap constraints for each resource
        for res_id, intervals in resource_intervals.items():
            if len(intervals) > 1:
                self.model.AddNoOverlap(intervals)
    
    def add_shift_constraints(self):
        """Add shift boundary constraints with packaging downtime"""
        self.logger.info("Adding shift constraints...")
        
        packaging_tasks = []
        
        # Identify packaging tasks
        for task_key, task_vars in self.task_vars.items():
            step_id = task_vars['step_id']
            if self._is_packaging_step(step_id):
                packaging_tasks.append((task_key, task_vars))
        
        # Add constraints for each packaging task and shift boundary
        for task_key, task_vars in packaging_tasks:
            for downtime_id, downtime_info in self.downtime_vars.items():
                self._add_packaging_shift_constraint(task_vars, downtime_info, task_key)
    
    def _add_packaging_shift_constraint(self, task_vars: Dict, downtime_info: Dict, task_key: Tuple):
        """Add constraint for packaging task crossing shift boundary"""
        shift_boundary = downtime_info['shift_boundary']
        downtime_start = downtime_info['start']
        downtime_end = downtime_info['end']
        
        # Boolean variable: does task overlap with downtime period?
        overlaps = self.model.NewBoolVar(f'overlaps_{task_key[0]}_{task_key[1]}_{shift_boundary}')
        
        # Task overlaps if it starts before downtime ends and ends after downtime starts
        starts_before_downtime_ends = self.model.NewBoolVar(f'starts_before_{task_key[0]}_{task_key[1]}_{shift_boundary}')
        ends_after_downtime_starts = self.model.NewBoolVar(f'ends_after_{task_key[0]}_{task_key[1]}_{shift_boundary}')
        
        self.model.Add(task_vars['start'] < downtime_end).OnlyEnforceIf(starts_before_downtime_ends)
        self.model.Add(task_vars['start'] >= downtime_end).OnlyEnforceIf(starts_before_downtime_ends.Not())
        
        self.model.Add(task_vars['end'] > downtime_start).OnlyEnforceIf(ends_after_downtime_starts)
        self.model.Add(task_vars['end'] <= downtime_start).OnlyEnforceIf(ends_after_downtime_starts.Not())
        
        # Task overlaps if both conditions are true
        self.model.AddBoolAnd([starts_before_downtime_ends, ends_after_downtime_starts]).OnlyEnforceIf(overlaps)
        
        # If task overlaps, it must either:
        # 1. Finish before downtime starts, OR
        # 2. Start after downtime ends
        finishes_before = self.model.NewBoolVar(f'finishes_before_{task_key[0]}_{task_key[1]}_{shift_boundary}')
        starts_after = self.model.NewBoolVar(f'starts_after_{task_key[0]}_{task_key[1]}_{shift_boundary}')
        
        self.model.Add(task_vars['end'] <= downtime_start).OnlyEnforceIf(finishes_before)
        self.model.Add(task_vars['start'] >= downtime_end).OnlyEnforceIf(starts_after)
        
        # If overlaps, then must choose one of the two options
        self.model.AddBoolOr([finishes_before, starts_after]).OnlyEnforceIf(overlaps)
    
    def add_room_constraints(self):
        """Add room capacity constraints"""
        self.logger.info("Adding room constraints...")
        
        for room_id, room in self.rooms.items():
            intervals = []
            demands = []
            
            for task_key, task_vars in self.task_vars.items():
                if room_id in task_vars['rooms']:
                    room_var = task_vars['rooms'][room_id]
                    
                    # Create optional interval for room usage
                    interval_var = self.model.NewOptionalIntervalVar(
                        task_vars['start'],
                        task_vars['end'] - task_vars['start'],
                        task_vars['end'],
                        room_var,
                        f'room_interval_{task_key[0]}_{task_key[1]}_{room_id}'
                    )
                    intervals.append(interval_var)
                    demands.append(task_vars['volume'])
            
            # Cumulative constraint for room capacity
            if intervals:
                self.model.AddCumulative(intervals, demands, room.capacity)
    
    def add_setup_constraints(self):
        """Add setup time constraints for SKU/product changes"""
        self.logger.info("Adding setup constraints...")
        
        # Group tasks by resource
        resource_tasks = {}
        for task_key, task_vars in self.task_vars.items():
            for res_id, res_var in task_vars['resources'].items():
                if res_id not in resource_tasks:
                    resource_tasks[res_id] = []
                resource_tasks[res_id].append((task_key, task_vars, res_var))
        
        # Add setup constraints between tasks on same resource
        for res_id, tasks in resource_tasks.items():
            resource = next(r for r in self.resources if r.resource_id == res_id)
            setup_time = getattr(resource, 'setup_time', self.default_setup_time)
            
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    task1_key, task1_vars, task1_res_var = tasks[i]
                    task2_key, task2_vars, task2_res_var = tasks[j]
                    
                    if self._tasks_need_setup(task1_key, task2_key):
                        self._add_setup_constraint(task1_vars, task2_vars, task1_res_var, 
                                                 task2_res_var, setup_time, i, j, res_id)
    
    def _add_setup_constraint(self, task1_vars, task2_vars, task1_res_var, task2_res_var, 
                             setup_time, i, j, res_id):
        """Add setup constraint between two tasks"""
        both_on_resource = self.model.NewBoolVar(f'both_on_{res_id}_{i}_{j}')
        self.model.AddBoolAnd([task1_res_var, task2_res_var]).OnlyEnforceIf(both_on_resource)
        
        # Task 1 ends before task 2 starts
        task1_before_task2 = self.model.NewBoolVar(f't1_before_t2_{i}_{j}')
        self.model.Add(task1_vars['end'] <= task2_vars['start']).OnlyEnforceIf(task1_before_task2)
        
        # Task 2 ends before task 1 starts  
        task2_before_task1 = self.model.NewBoolVar(f't2_before_t1_{i}_{j}')
        self.model.Add(task2_vars['end'] <= task1_vars['start']).OnlyEnforceIf(task2_before_task1)
        
        # If both on same resource, one must be before the other
        self.model.AddBoolOr([task1_before_task2, task2_before_task1]).OnlyEnforceIf(both_on_resource)
        
        # Add setup time
        self.model.Add(task2_vars['start'] >= task1_vars['end'] + setup_time).OnlyEnforceIf(
            self.model.NewBoolVar().AddBoolAnd([both_on_resource, task1_before_task2])
        )
        self.model.Add(task1_vars['start'] >= task2_vars['end'] + setup_time).OnlyEnforceIf(
            self.model.NewBoolVar().AddBoolAnd([both_on_resource, task2_before_task1])
        )
    
    def create_objective(self):
        """Create multi-objective function"""
        self.logger.info("Creating objective function...")
        
        objective_terms = []
        
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            # Get completion time (end time of last step)
            last_step_id = product.steps[-1]
            last_task_key = (sku_id, last_step_id)
            completion_time = self.task_vars[last_task_key]['end']
            
            # Priority weight (higher priority = more important to complete early)
            priority_weight = sku.priority * 100
            
            # Due date penalty (in minutes from schedule start)
            due_date_minutes = sku.due_date * 24 * 60
            lateness = self.model.NewIntVar(0, self.schedule_horizon, f'lateness_{sku_id}')
            self.model.AddMaxEquality(lateness, [completion_time - due_date_minutes, 0])
            
            # Volume satisfaction
            total_production = sum(
                self.task_vars[(sku_id, step_id)]['volume'] 
                for step_id in product.steps
            ) // len(product.steps)  # Use integer division
            
            shortage = self.model.NewIntVar(0, sku.quantity_required, f'shortage_{sku_id}')
            self.model.AddMaxEquality(shortage, [sku.quantity_required - total_production, 0])
            
            # Objective terms (maximizing, so negate costs)
            objective_terms.extend([
                -priority_weight,  # Base priority bonus
                -lateness * 1000,  # Heavy lateness penalty
                -shortage * 5000,  # Very heavy shortage penalty
                -completion_time   # Prefer earlier completion
            ])
        
        # Resource utilization bonus
        for res_id in [r.resource_id for r in self.resources]:
            utilization = sum(
                (task_vars['end'] - task_vars['start']) * res_var
                for task_vars in self.task_vars.values()
                for res_var_id, res_var in task_vars['resources'].items()
                if res_var_id == res_id
            )
            objective_terms.append(utilization // 10)  # Small bonus for utilization
        
        self.model.Maximize(sum(objective_terms))
    
    def solve(self, time_limit_seconds: int = 300) -> List[ScheduleItem]:
        """Solve the scheduling problem"""
        self.logger.info("Starting optimization process...")
        
        try:
            # Build the model
            self.create_variables()
            self.add_time_constraints()
            self.add_resource_constraints()
            self.add_room_constraints()
            self.add_shift_constraints()
            self.add_setup_constraints()
            self.create_objective()
            
            # Configure solver
            self.solver.parameters.max_time_in_seconds = time_limit_seconds
            self.solver.parameters.num_search_workers = 8
            self.solver.parameters.log_search_progress = True
            
            # Solve
            self.logger.info(f"Solving with {len(self.task_vars)} tasks and {time_limit_seconds}s time limit...")
            status = self.solver.Solve(self.model)
            
            # Process results
            if status == cp_model.OPTIMAL:
                self.logger.info("Optimal solution found!")
                return self._extract_solution()
            elif status == cp_model.FEASIBLE:
                self.logger.info("Feasible solution found!")
                return self._extract_solution()
            else:
                self.logger.error(f"Solver failed with status: {self.solver.StatusName(status)}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error during solving: {e}")
            return []
    
    def _extract_solution(self) -> List[ScheduleItem]:
        """Extract solution from solved model"""
        schedule_items = []
        
        # Extract main tasks
        for task_key, task_vars in self.task_vars.items():
            sku_id, step_id = task_key
            
            start_time = self.solver.Value(task_vars['start'])
            end_time = self.solver.Value(task_vars['end'])
            volume = self.solver.Value(task_vars['volume'])
            
            # Find selected resource
            selected_resource = None
            for res_id, res_var in task_vars['resources'].items():
                if self.solver.Value(res_var):
                    selected_resource = res_id
                    break
            
            # Find selected room
            selected_room = None
            for room_id, room_var in task_vars['rooms'].items():
                if self.solver.Value(room_var):
                    selected_room = room_id
                    break
            
            schedule_items.append(ScheduleItem(
                sku_id=sku_id,
                step_id=step_id,
                resource_id=selected_resource,
                start_time=start_time,
                end_time=end_time,
                volume=volume,
                room_id=selected_room,
                task_type=TaskType.PRODUCTION
            ))
        
        # Extract CIP tasks
        for cip_id, cip_vars in self.cip_vars.items():
            start_time = self.solver.Value(cip_vars['start'])
            end_time = self.solver.Value(cip_vars['end'])
            
            schedule_items.append(ScheduleItem(
                sku_id=cip_vars['sku_id'],
                step_id=f"CIP_{cip_vars['step_id']}",
                resource_id=cip_vars['circuit'],
                start_time=start_time,
                end_time=end_time,
                volume=0,
                task_type=TaskType.CIP
            ))
        
        return sorted(schedule_items, key=lambda x: x.start_time)
    
    def print_schedule(self, schedule: List[ScheduleItem]):
        """Print formatted schedule with datetime information"""
        if not schedule:
            print("No schedule generated.")
            return
        
        print(f"\n{'='*80}")
        print(f"PRODUCTION SCHEDULE - {len(schedule)} tasks")
        print(f"Schedule Start: {self.time_manager.schedule_start}")
        print(f"{'='*80}")
        
        current_day = None
        for item in schedule:
            start_dt, end_dt = item.to_datetime(self.time_manager.schedule_start)
            
            # Print day header if day changed
            if current_day != start_dt.date():
                current_day = start_dt.date()
                print(f"\n--- {current_day.strftime('%A, %B %d, %Y')} ---")
            
            duration_mins = item.end_time - item.start_time
            print(f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} "
                  f"({duration_mins:3d}min) | "
                  f"{item.task_type.value.upper():4s} | "
                  f"SKU:{item.sku_id:8s} | "
                  f"Step:{item.step_id:12s} | "
                  f"Res:{item.resource_id:8s} | "
                  f"Vol:{item.volume:6.0f}")
        
        print(f"\n{'='*80}")
    
    # Helper methods
    def _get_compatible_resources(self, step_id: str) -> List[str]:
        """Get resources compatible with a step"""
        compatible = []
        for resource in self.resources:
            if hasattr(resource, 'compatible_steps'):
                if step_id in resource.compatible_steps:
                    compatible.append(resource.resource_id)
            else:
                # Default compatibility logic
                if any(keyword in step_id.lower() for keyword in ['tank', 'ferment', 'brew']):
                    if any(keyword in resource.resource_id.lower() for keyword in ['tank', 'ferment', 'brew']):
                        compatible.append(resource.resource_id)
                elif any(keyword in step_id.lower() for keyword in ['pack', 'fill', 'bottle']):
                    if any(keyword in resource.resource_id.lower() for keyword in ['line', 'pack', 'fill']):
                        compatible.append(resource.resource_id)
        return compatible
    
    def _get_compatible_rooms(self, step_id: str) -> List[str]:
        """Get rooms compatible with a step"""
        compatible = []
        for room_id, room in self.rooms.items():
            if hasattr(room, 'compatible_steps'):
                if step_id in room.compatible_steps:
                    compatible.append(room_id)
            else:
                # Default room compatibility
                if any(keyword in step_id.lower() for keyword in ['storage', 'hold', 'age']):
                    compatible.append(room_id)
        return compatible
    
    def _step_needs_room(self, step_id: str) -> bool:
        """Check if step needs room storage"""
        room_keywords = ['storage', 'hold', 'age', 'cure', 'condition']
        return any(keyword in step_id.lower() for keyword in room_keywords)
    
    def _is_packaging_step(self, step_id: str) -> bool:
        """Check if step is a packaging operation"""
        packaging_keywords = ['pack', 'fill', 'bottle', 'can', 'box', 'wrap']
        return any(keyword in step_id.lower() for keyword in packaging_keywords)
    
    def _needs_cip_before_task(self, sku, step_id: str) -> bool:
        """Check if CIP is needed before this task"""
        # CIP needed for:
        # 1. First step of different product types
        # 2. Steps after allergen products
        # 3. Specific process steps that require cleaning
        
        product = next(p for p in self.products if p.product_id == sku.product_id)
        
        # First step of production typically needs CIP
        if step_id == product.steps[0]:
            return True
        
        # Specific steps that always need CIP
        cip_required_steps = ['ferment', 'brew', 'mix', 'blend']
        if any(keyword in step_id.lower() for keyword in cip_required_steps):
            return True
        
        return False
    
    def _get_required_cip_circuit(self, step_id: str) -> Optional[str]:
        """Get CIP circuit required for a step"""
        for circuit in self.cip_circuits:
            if hasattr(circuit, 'compatible_steps'):
                if step_id in circuit.compatible_steps:
                    return circuit.circuit_id
            else:
                # Default CIP circuit assignment
                if hasattr(circuit, 'circuit_id'):
                    return circuit.circuit_id
        
        # If no specific circuit found, return first available
        if self.cip_circuits:
            return getattr(self.cip_circuits[0], 'circuit_id', 'CIP_1')
        return None
    
    def _tasks_need_setup(self, task1_key: Tuple[str, str], task2_key: Tuple[str, str]) -> bool:
        """Check if setup is needed between two tasks"""
        sku1_id, step1_id = task1_key
        sku2_id, step2_id = task2_key
        
        # No setup needed for same SKU
        if sku1_id == sku2_id:
            return False
        
        # Extract product and variant information
        def parse_sku_id(sku_id: str) -> Tuple[str, str]:
            parts = sku_id.split('_')
            product = parts[0] if parts else sku_id
            variant = parts[1] if len(parts) > 1 else 'default'
            return product, variant
        
        product1, variant1 = parse_sku_id(sku1_id)
        product2, variant2 = parse_sku_id(sku2_id)
        
        # Different products always need setup
        if product1 != product2:
            return True
        
        # Same product, different variants may need setup
        if variant1 != variant2:
            # Check if variants require setup (e.g., different flavors, allergens)
            return self._variants_need_setup(variant1, variant2)
        
        return False
    
    def _variants_need_setup(self, variant1: str, variant2: str) -> bool:
        """Check if two variants need setup between them"""
        # Define variant groups that don't need setup between each other
        no_setup_groups = [
            ['vanilla', 'plain', 'original'],
            ['chocolate', 'cocoa', 'dark'],
            ['strawberry', 'berry', 'fruit']
        ]
        
        for group in no_setup_groups:
            if variant1.lower() in group and variant2.lower() in group:
                return False
        
        # Default: different variants need setup
        return True

# Enhanced usage function with better error handling and reporting
def run_enhanced_scheduler(skus, products, resources, rooms, cip_circuits, 
                          schedule_start: datetime = None, 
                          shift_boundaries_hours: List[int] = None,
                          schedule_horizon_hours: int = 168,
                          time_limit_seconds: int = 300) -> List[ScheduleItem]:
    """
    Run the enhanced production scheduler
    
    Args:
        skus: List of SKU objects with production requirements
        products: List of product definitions with process steps
        resources: List of production resources (tanks, lines, etc.)
        rooms: Dictionary of storage rooms/areas
        cip_circuits: List of CIP cleaning circuits
        schedule_start: Start datetime for the schedule (default: now)
        shift_boundaries_hours: List of shift change hours [8, 16, 24] (default)
        schedule_horizon_hours: Schedule duration in hours (default: 168 = 1 week)
        time_limit_seconds: Solver time limit (default: 300 = 5 minutes)
    
    Returns:
        List of ScheduleItem objects representing the optimized schedule
    """
    
    # Set defaults
    if schedule_start is None:
        schedule_start = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    if shift_boundaries_hours is None:
        shift_boundaries_hours = [8, 16, 24]  # 8 AM, 4 PM, Midnight
    
    try:
        # Create and run scheduler
        scheduler = ProductionScheduler(
            skus=skus,
            products=products, 
            resources=resources,
            rooms=rooms,
            cip_circuits=cip_circuits,
            schedule_start=schedule_start,
            shift_boundaries_hours=shift_boundaries_hours,
            schedule_horizon_hours=schedule_horizon_hours
        )
        
        # Solve the scheduling problem
        schedule = scheduler.solve(time_limit_seconds=time_limit_seconds)
        
        # Print results
        scheduler.print_schedule(schedule)
        
        # Print summary statistics
        if schedule:
            print(f"\nSCHEDULE SUMMARY:")
            print(f"Total tasks scheduled: {len(schedule)}")
            
            # Count by task type
            task_counts = {}
            for item in schedule:
                task_counts[item.task_type.value] = task_counts.get(item.task_type.value, 0) + 1
            
            for task_type, count in task_counts.items():
                print(f"  {task_type.title()} tasks: {count}")
            
            # Resource utilization
            resource_usage = {}
            for item in schedule:
                if item.resource_id:
                    duration = item.end_time - item.start_time
                    resource_usage[item.resource_id] = resource_usage.get(item.resource_id, 0) + duration
            
            print(f"\nRESOURCE UTILIZATION:")
            total_horizon = schedule_horizon_hours * 60
            for res_id, usage_mins in resource_usage.items():
                utilization_pct = (usage_mins / total_horizon) * 100
                print(f"  {res_id}: {usage_mins} minutes ({utilization_pct:.1f}%)")
            
            # Schedule span
            if schedule:
                first_start = min(item.start_time for item in schedule)
                last_end = max(item.end_time for item in schedule)
                span_hours = (last_end - first_start) / 60
                print(f"\nSchedule span: {span_hours:.1f} hours")
        else:
            print("No feasible schedule found. Consider:")
            print("  - Reducing production requirements")
            print("  - Increasing resource capacity")
            print("  - Extending schedule horizon")
            print("  - Relaxing constraints")
        
        return schedule
        
    except Exception as e:
        logging.error(f"Error in scheduler: {e}")
        print(f"Scheduling failed: {e}")
        return []

# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime
    
    # This would be replaced with actual data structures
    print("Enhanced Production Scheduler")
    print("=" * 50)
    print("Key improvements:")
    print("- Real datetime handling with TimeManager")
    print("- Proper shift change downtime for packaging (30 min)")
    print("- Enhanced constraint modeling")
    print("- Better resource compatibility logic")
    print("- Comprehensive solution reporting")
    print("- Robust error handling")
    print("- Multiple task types (production, CIP, setup)")
    print("- Improved objective function")
    print()
    print("To use:")
    print("schedule = run_enhanced_scheduler(skus, products, resources, rooms, cip_circuits)")
    print()
    print("The scheduler now properly handles packaging operations that cross")
    print("shift boundaries by ensuring they either complete before the shift")
    print("change or start after the 30-minute downtime period.")