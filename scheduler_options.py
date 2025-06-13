from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import logging
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from collections import defaultdict
import heapq
import numpy as np

class TaskType(Enum):
    PRODUCTION = "production"
    CIP = "cip"
    SETUP = "setup"
    DOWNTIME = "downtime"
    MAINTENANCE = "maintenance"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScheduleItem:
    """Enhanced schedule item with comprehensive tracking"""
    sku_id: str
    step_id: str
    resource_id: str
    start_time: int  # minutes from schedule start
    end_time: int
    volume: float
    room_id: Optional[str] = None
    task_type: TaskType = TaskType.PRODUCTION
    setup_time: int = 0
    cip_time: int = 0
    priority: Priority = Priority.MEDIUM
    cost: float = 0.0
    efficiency: float = 1.0
    quality_score: float = 1.0
    
    def to_datetime(self, schedule_start: datetime) -> Tuple[datetime, datetime]:
        """Convert time offsets to actual datetime objects"""
        start_dt = schedule_start + timedelta(minutes=self.start_time)
        end_dt = schedule_start + timedelta(minutes=self.end_time)
        return start_dt, end_dt
    
    @property
    def duration(self) -> int:
        """Task duration in minutes"""
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return (f"ScheduleItem(SKU={self.sku_id}, Step={self.step_id}, "
                f"Resource={self.resource_id}, Time={self.start_time}-{self.end_time}, "
                f"Volume={self.volume}, Type={self.task_type.value})")

@dataclass
class SchedulingResult:
    """Comprehensive scheduling result with metrics"""
    schedule_items: List[ScheduleItem]
    total_production: float
    efficiency_score: float
    cost_score: float
    utilization_score: float
    quality_score: float
    makespan: int  # Total schedule duration
    unfulfilled_demand: Dict[str, float]
    warnings: List[str]
    solver_status: str
    solve_time: float
    
    def print_summary(self):
        """Print comprehensive scheduling summary"""
        print(f"\n{'='*80}")
        print(f"PRODUCTION SCHEDULE SUMMARY")
        print(f"{'='*80}")
        print(f"Total items scheduled: {len(self.schedule_items)}")
        print(f"Total production volume: {self.total_production:,.2f}")
        print(f"Makespan: {self.makespan} minutes ({self.makespan/60:.1f} hours)")
        print(f"Efficiency score: {self.efficiency_score:.2f}%")
        print(f"Cost score: {self.cost_score:.2f}")
        print(f"Utilization score: {self.utilization_score:.2f}%")
        print(f"Quality score: {self.quality_score:.2f}")
        print(f"Solver status: {self.solver_status}")
        print(f"Solve time: {self.solve_time:.2f} seconds")
        
        if self.unfulfilled_demand:
            print(f"\nUnfulfilled demand:")
            for sku_id, volume in self.unfulfilled_demand.items():
                print(f"  {sku_id}: {volume:,.2f}")
        
        if self.warnings:
            print(f"\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

class TimeManager:
    """Advanced time management with multiple calendars and constraints"""
    
    def __init__(self, schedule_start: datetime, 
                 shift_boundaries_hours: List[int],
                 working_days: List[int] = None,
                 holidays: List[datetime] = None):
        self.schedule_start = schedule_start
        self.shift_boundaries_hours = shift_boundaries_hours
        self.working_days = working_days or [0, 1, 2, 3, 4]  # Mon-Fri
        self.holidays = holidays or []
        self.shift_boundaries_minutes = self._calculate_shift_boundaries()
        self.working_windows = self._calculate_working_windows()
    
    def _calculate_shift_boundaries(self) -> List[int]:
        """Calculate shift boundaries with day/night considerations"""
        boundaries = []
        current_date = self.schedule_start.date()
        
        for day_offset in range(14):  # Two weeks
            day = current_date + timedelta(days=day_offset)
            day_dt = datetime.combine(day, datetime.min.time())
            
            # Skip non-working days
            if day.weekday() not in self.working_days:
                continue
            
            # Skip holidays
            if day_dt in self.holidays:
                continue
            
            for hour in self.shift_boundaries_hours:
                boundary_time = day_dt + timedelta(hours=hour)
                if boundary_time >= self.schedule_start:
                    minutes_offset = int((boundary_time - self.schedule_start).total_seconds() / 60)
                    boundaries.append(minutes_offset)
        
        return sorted(boundaries)
    
    def _calculate_working_windows(self) -> List[Tuple[int, int]]:
        """Calculate available working time windows"""
        windows = []
        current_date = self.schedule_start.date()
        
        for day_offset in range(14):
            day = current_date + timedelta(days=day_offset)
            
            if day.weekday() not in self.working_days:
                continue
            
            day_dt = datetime.combine(day, datetime.min.time())
            if day_dt in self.holidays:
                continue
            
            # Assume 3 shifts: 6-14, 14-22, 22-6
            for i in range(len(self.shift_boundaries_hours) - 1):
                start_hour = self.shift_boundaries_hours[i]
                end_hour = self.shift_boundaries_hours[i + 1]
                
                start_time = day_dt + timedelta(hours=start_hour)
                end_time = day_dt + timedelta(hours=end_hour)
                
                if start_time >= self.schedule_start:
                    start_minutes = int((start_time - self.schedule_start).total_seconds() / 60)
                    end_minutes = int((end_time - self.schedule_start).total_seconds() / 60)
                    windows.append((start_minutes, end_minutes))
        
        return windows
    
    def is_working_time(self, time_minutes: int) -> bool:
        """Check if given time is within working hours"""
        for start, end in self.working_windows:
            if start <= time_minutes < end:
                return True
        return False

class AdvancedProductionScheduler:
    """State-of-the-art production scheduler with multiple optimization techniques"""
    
    def __init__(self, skus, products, resources, rooms, cip_circuits,
                 schedule_start: datetime,
                 shift_boundaries_hours: List[int],
                 schedule_horizon_hours: int = 168,
                 working_days: List[int] = None,
                 holidays: List[datetime] = None):
        
        # Core data
        self.skus = skus
        self.products = products
        self.resources = resources
        self.rooms = rooms
        self.cip_circuits = cip_circuits
        
        # Time management
        self.time_manager = TimeManager(schedule_start, shift_boundaries_hours, 
                                      working_days, holidays)
        self.schedule_horizon = schedule_horizon_hours * 60
        
        # Performance parameters
        self.OEE = 0.85
        self.packaging_downtime_minutes = 30
        self.default_setup_time = 30
        self.default_cip_time = 60
        
        # Optimization weights
        self.weights = {
            'production_volume': 1.0,
            'due_date_penalty': 100.0,
            'priority_bonus': 50.0,
            'setup_cost': 10.0,
            'cip_cost': 15.0,
            'quality_bonus': 25.0,
            'efficiency_bonus': 20.0,
            'utilization_bonus': 5.0
        }
        
        # OR-Tools components
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Decision variables
        self.task_vars = {}
        self.cip_vars = {}
        self.setup_vars = {}
        self.downtime_vars = {}
        self.maintenance_vars = {}
        
        # Auxiliary variables for advanced constraints
        self.sequence_vars = {}
        self.quality_vars = {}
        self.cost_vars = {}
        
        # Tracking
        self.solve_start_time = None
        self.warnings = []
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def schedule_production(self, time_limit_seconds: int = 600,
                          quality_threshold: float = 0.8,
                          max_iterations: int = 3) -> SchedulingResult:
        """Main scheduling method with iterative improvement"""
        self.solve_start_time = datetime.now()
        
        try:
            # Multi-stage optimization
            best_result = None
            best_score = float('-inf')
            
            for iteration in range(max_iterations):
                self.logger.info(f"Starting optimization iteration {iteration + 1}")
                
                # Adjust parameters for each iteration
                self._adjust_parameters_for_iteration(iteration)
                
                # Build and solve model
                result = self._solve_iteration(time_limit_seconds // max_iterations)
                
                if result and result.schedule_items:
                    # Calculate composite score
                    score = self._calculate_composite_score(result)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        self.logger.info(f"New best score: {score:.2f}")
                
                # Reset model for next iteration
                self._reset_model()
            
            return best_result or self._create_empty_result()
            
        except Exception as e:
            self.logger.error(f"Error in scheduling: {e}")
            self.warnings.append(f"Scheduling error: {str(e)}")
            return self._create_empty_result()
    
    def _adjust_parameters_for_iteration(self, iteration: int):
        """Adjust optimization parameters for different iterations"""
        if iteration == 0:
            # First iteration: Focus on feasibility
            self.weights['production_volume'] = 2.0
            self.weights['due_date_penalty'] = 50.0
        elif iteration == 1:
            # Second iteration: Focus on quality
            self.weights['quality_bonus'] = 50.0
            self.weights['efficiency_bonus'] = 30.0
        else:
            # Final iteration: Balanced optimization
            self.weights['production_volume'] = 1.0
            self.weights['due_date_penalty'] = 100.0
            self.weights['quality_bonus'] = 25.0
            self.weights['efficiency_bonus'] = 20.0
    
    def _solve_iteration(self, time_limit: int) -> Optional[SchedulingResult]:
        """Solve a single iteration"""
        try:
            # Create model
            self._create_enhanced_variables()
            self._add_enhanced_constraints()
            self._create_advanced_objective()
            
            # Configure solver
            self.solver.parameters.max_time_in_seconds = time_limit
            self.solver.parameters.num_search_workers = 8
            self.solver.parameters.log_search_progress = True
            
            # Solve
            status = self.solver.Solve(self.model)
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return self._extract_enhanced_solution(status)
            else:
                self.logger.warning(f"Solver status: {self.solver.StatusName(status)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in iteration: {e}")
            return None
    
    def _create_enhanced_variables(self):
        """Create comprehensive decision variables"""
        self.logger.info("Creating enhanced decision variables...")
        
        # Main production variables
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            for step_id in product.steps:
                task_key = (sku_id, step_id)
                self._create_enhanced_task_variables(task_key, sku, step_id)
        
        # CIP variables with enhanced modeling
        self._create_enhanced_cip_variables()
        
        # Setup variables with sequence dependencies
        self._create_enhanced_setup_variables()
        
        # Quality and efficiency variables
        self._create_quality_variables()
        
        # Cost tracking variables
        self._create_cost_variables()
    
    def _create_enhanced_task_variables(self, task_key: Tuple[str, str], sku, step_id: str):
        """Create enhanced task variables with quality and efficiency tracking"""
        sku_id = task_key[0]
        
        # Time variables with working hour constraints
        start_var = self.model.NewIntVar(0, self.schedule_horizon, f'start_{sku_id}_{step_id}')
        end_var = self.model.NewIntVar(0, self.schedule_horizon, f'end_{sku_id}_{step_id}')
        
        # Volume with quality considerations
        base_required = sku.quantity_required
        max_volume = int(base_required * 1.3)  # Allow 30% overproduction
        volume_var = self.model.NewIntVar(base_required, max_volume, f'volume_{sku_id}_{step_id}')
        
        # Resource selection with capability scoring
        compatible_resources = self._get_enhanced_compatible_resources(step_id, sku)
        resource_vars = {}
        
        for res_id, capability_score in compatible_resources.items():
            resource_vars[res_id] = self.model.NewBoolVar(f'use_res_{sku_id}_{step_id}_{res_id}')
        
        if resource_vars:
            self.model.AddExactlyOne(resource_vars.values())
        
        # Room selection with capacity and compatibility
        room_vars = {}
        if self._step_needs_room(step_id):
            compatible_rooms = self._get_enhanced_compatible_rooms(step_id, sku)
            for room_id, capacity_score in compatible_rooms.items():
                room_vars[room_id] = self.model.NewBoolVar(f'use_room_{sku_id}_{step_id}_{room_id}')
            
            if room_vars:
                self.model.AddAtMostOne(room_vars.values())
        
        # Quality score variable
        quality_var = self.model.NewIntVar(0, 100, f'quality_{sku_id}_{step_id}')
        
        # Efficiency variable
        efficiency_var = self.model.NewIntVar(0, 100, f'efficiency_{sku_id}_{step_id}')
        
        self.task_vars[task_key] = {
            'start': start_var,
            'end': end_var,
            'volume': volume_var,
            'resources': resource_vars,
            'rooms': room_vars,
            'quality': quality_var,
            'efficiency': efficiency_var,
            'sku_id': sku_id,
            'step_id': step_id,
            'priority': getattr(sku, 'priority', Priority.MEDIUM)
        }
    
    def _create_enhanced_cip_variables(self):
        """Create CIP variables with scheduling optimization"""
        self.logger.info("Creating enhanced CIP variables...")
        
        cip_counter = 0
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            for step_id in product.steps:
                if self._needs_cip_before_task(sku, step_id):
                    cip_id = f"cip_{cip_counter}"
                    cip_counter += 1
                    
                    # Variable CIP duration based on contamination level
                    min_cip_duration = 45
                    max_cip_duration = 120
                    cip_duration_var = self.model.NewIntVar(
                        min_cip_duration, max_cip_duration, f'cip_duration_{cip_id}'
                    )
                    
                    cip_start = self.model.NewIntVar(0, self.schedule_horizon, f'cip_start_{cip_id}')
                    cip_end = self.model.NewIntVar(0, self.schedule_horizon, f'cip_end_{cip_id}')
                    
                    self.model.Add(cip_end == cip_start + cip_duration_var)
                    
                    # CIP effectiveness variable
                    cip_effectiveness = self.model.NewIntVar(70, 100, f'cip_effectiveness_{cip_id}')
                    
                    self.cip_vars[cip_id] = {
                        'start': cip_start,
                        'end': cip_end,
                        'duration': cip_duration_var,
                        'effectiveness': cip_effectiveness,
                        'sku_id': sku_id,
                        'step_id': step_id,
                        'circuit': self._get_required_cip_circuit(step_id)
                    }
    
    def _create_enhanced_setup_variables(self):
        """Create setup variables with sequence optimization"""
        self.logger.info("Creating enhanced setup variables...")
        
        # Track setup sequences for each resource
        for resource in self.resources:
            res_id = resource.resource_id
            
            # Get all tasks that could use this resource
            resource_tasks = []
            for task_key, task_vars in self.task_vars.items():
                if res_id in task_vars['resources']:
                    resource_tasks.append(task_key)
            
            # Create setup variables for each pair of tasks
            for i, task1 in enumerate(resource_tasks):
                for j, task2 in enumerate(resource_tasks):
                    if i != j:
                        setup_key = f"setup_{res_id}_{i}_{j}"
                        
                        # Binary variable: is there a setup between task1 and task2?
                        setup_var = self.model.NewBoolVar(setup_key)
                        
                        # Setup time variable (depends on product changeover)
                        setup_time_var = self.model.NewIntVar(0, 240, f'setup_time_{res_id}_{i}_{j}')
                        
                        self.setup_vars[setup_key] = {
                            'setup': setup_var,
                            'setup_time': setup_time_var,
                            'task1': task1,
                            'task2': task2,
                            'resource': res_id
                        }
    
    def _create_quality_variables(self):
        """Create quality tracking variables"""
        for sku in self.skus:
            sku_id = sku.sku_id
            
            # Overall quality score for the SKU
            sku_quality = self.model.NewIntVar(0, 100, f'sku_quality_{sku_id}')
            
            # Quality consistency variable
            quality_consistency = self.model.NewIntVar(0, 100, f'quality_consistency_{sku_id}')
            
            self.quality_vars[sku_id] = {
                'overall_quality': sku_quality,
                'consistency': quality_consistency
            }
    
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
    
    def _add_enhanced_constraints(self):
        """Add comprehensive constraints"""
        self.logger.info("Adding enhanced constraints...")
        
        # Core constraints
        self._add_enhanced_time_constraints()
        self._add_enhanced_resource_constraints()
        self._add_enhanced_setup_constraints()
        self._add_working_time_constraints()
        
        # Advanced constraints
        self._add_sequence_constraints()
        self._add_load_balancing_constraints()
        self._add_energy_efficiency_constraints()
    
    def _add_enhanced_time_constraints(self):
        """Add time constraints with quality considerations"""
        for sku in self.skus:
            sku_id = sku.sku_id
            product = next(p for p in self.products if p.product_id == sku.product_id)
            
            prev_task_vars = None
            for i, step_id in enumerate(product.steps):
                task_key = (sku_id, step_id)
                task_vars = self.task_vars[task_key]
                
                # Enhanced duration constraints
                self._add_enhanced_duration_constraints(task_vars, step_id, sku)
                
                # Sequential constraints with buffer time
                if prev_task_vars is not None:
                    buffer_time = self._calculate_buffer_time(prev_task_vars['step_id'], step_id)
                    self.model.Add(task_vars['start'] >= prev_task_vars['end'] + buffer_time)
                
                # CIP timing with effectiveness
                self._add_enhanced_cip_timing_constraints(sku, step_id, task_vars)
                
                prev_task_vars = task_vars
    
    def _add_enhanced_duration_constraints(self, task_vars: Dict, step_id: str, sku):
        """Add duration constraints with quality and efficiency factors"""
        for res_id, res_var in task_vars['resources'].items():
            resource = next(r for r in self.resources if r.resource_id == res_id)
            
            # Base capacity with OEE
            base_capacity = resource.capacity * self.OEE
            
            # Quality factor affects processing time
            quality_factor = self.model.NewIntVar(80, 120, f'quality_factor_{task_vars["sku_id"]}_{step_id}_{res_id}')
            
            # Efficiency factor
            efficiency_factor = self.model.NewIntVar(70, 100, f'efficiency_factor_{task_vars["sku_id"]}_{step_id}_{res_id}')
            
            # Adjusted processing time
            base_duration = self.model.NewIntVar(1, self.schedule_horizon, f'base_duration_{task_vars["sku_id"]}_{step_id}_{res_id}')
            
            # Calculate duration with quality and efficiency factors
            adjusted_duration = self.model.NewIntVar(1, self.schedule_horizon, f'adjusted_duration_{task_vars["sku_id"]}_{step_id}_{res_id}')
            
            # If resource is selected, calculate proper duration
            self.model.AddDivisionEquality(base_duration, task_vars['volume'], int(base_capacity)).OnlyEnforceIf(res_var)
            
            # Adjust for quality (higher quality = longer time)
            self.model.AddMultiplicationEquality(adjusted_duration, base_duration, quality_factor).OnlyEnforceIf(res_var)
            
            # Connect quality factor to quality variable
            self.model.Add(task_vars['quality'] >= quality_factor - 20).OnlyEnforceIf(res_var)
            
            # Connect efficiency factor to efficiency variable
            self.model.Add(task_vars['efficiency'] >= efficiency_factor).OnlyEnforceIf(res_var)
            
            # Set end time
            setup_time = getattr(resource, 'setup_time', self.default_setup_time)
            self.model.Add(task_vars['end'] >= task_vars['start'] + adjusted_duration + setup_time).OnlyEnforceIf(res_var)
    
    def _add_working_time_constraints(self):
        """Ensure tasks are scheduled within working hours"""
        for task_key, task_vars in self.task_vars.items():
            start_time = task_vars['start']
            end_time = task_vars['end']
            
            # Task must start and end within working windows
            working_window_vars = []
            
            for i, (window_start, window_end) in enumerate(self.time_manager.working_windows):
                # Boolean variable: is task in this window?
                in_window = self.model.NewBoolVar(f'in_window_{task_key[0]}_{task_key[1]}_{i}')
                working_window_vars.append(in_window)
                
                # If in this window, task must fit within it
                self.model.Add(start_time >= window_start).OnlyEnforceIf(in_window)
                self.model.Add(end_time <= window_end).OnlyEnforceIf(in_window)
            
            # Task must be in exactly one working window
            if working_window_vars:
                self.model.AddExactlyOne(working_window_vars)
    
    def _add_enhanced_quality_constraints(self):
        """Add quality-related constraints"""
        for sku_id, quality_vars in self.quality_vars.items():
            product = next(p for p in self.products if any(s.sku_id == sku_id for s in self.skus))
            
            # Overall quality is weighted average of step qualities
            step_qualities = []
            step_volumes = []
            
            for step_id in product.steps:
                task_key = (sku_id, step_id)
                if task_key in self.task_vars:
                    step_qualities.append(self.task_vars[task_key]['quality'])
                    step_volumes.append(self.task_vars[task_key]['volume'])
            
            # Calculate weighted average quality
            if step_qualities:
                total_volume = sum(step_volumes)
                weighted_quality = self.model.NewIntVar(0, 100, f'weighted_quality_{sku_id}')
                
                # Simplified weighted average constraint
                quality_sum = sum(q * v for q, v in zip(step_qualities, step_volumes))
                self.model.AddDivisionEquality(weighted_quality, quality_sum, total_volume)
                
                self.model.Add(quality_vars['overall_quality'] == weighted_quality)
    
    def _add_sequence_constraints(self):
        """Add sequence optimization constraints"""
        # Minimize sequence-dependent setups
        for res_id, resource in {r.resource_id: r for r in self.resources}.items():
            # Get tasks that use this resource
            resource_tasks = [(k, v) for k, v in self.task_vars.items() if res_id in v['resources']]
            
            if len(resource_tasks) <= 1:
                continue
            
            # Create sequence variables
            for i, (task1_key, task1_vars) in enumerate(resource_tasks):
                for j, (task2_key, task2_vars) in enumerate(resource_tasks):
                    if i != j:
                        # Sequence variable: task1 immediately before task2
                        seq_var = self.model.NewBoolVar(f'seq_{res_id}_{i}_{j}')
                        
                        # If both tasks use this resource and are sequential
                        both_on_resource = self.model.NewBoolVar(f'both_on_res_{res_id}_{i}_{j}')
                        self.model.AddBoolAnd([
                            task1_vars['resources'][res_id],
                            task2_vars['resources'][res_id]
                        ]).OnlyEnforceIf(both_on_resource)
                        
                        # Sequence implies both on resource
                        self.model.AddImplication(seq_var, both_on_resource)
                        
                        # If sequential, task2 starts after task1 ends
                        self.model.Add(task2_vars['start'] >= task1_vars['end']).OnlyEnforceIf(seq_var)
                        
                        self.sequence_vars[f'{res_id}_{i}_{j}'] = seq_var
    
    def _add_load_balancing_constraints(self):
        """Add load balancing constraints across resources"""
        # Balance utilization across similar resources
        resource_groups = self._group_similar_resources()
        
        for group_name, resource_list in resource_groups.items():
            if len(resource_list) <= 1:
                continue
            
            # Calculate utilization for each resource in the group
            utilizations = []
            for res_id in resource_list:
                total_usage = self.model.NewIntVar(0, self.schedule_horizon, f'usage_{res_id}')
                
                # Sum all task durations on this resource
                usage_terms = []
                for task_key, task_vars in self.task_vars.items():
                    if res_id in task_vars['resources']:
                        duration = task_vars['end'] - task_vars['start']
                        usage_terms.append(duration * task_vars['resources'][res_id])
                
                if usage_terms:
                    self.model.Add(total_usage == sum(usage_terms))
                else:
                    self.model.Add(total_usage == 0)
                
                utilizations.append(total_usage)
            
            # Add balance constraints (utilization difference should be small)
            max_utilization = self.model.NewIntVar(0, self.schedule_horizon, f'max_util_{group_name}')
            min_utilization = self.model.NewIntVar(0, self.schedule_horizon, f'min_util_{group_name}')
            
            self.model.AddMaxEquality(max_utilization, utilizations)
            self.model.AddMinEquality(min_utilization, utilizations)
            
            # Balance constraint: difference should be less than 20% of average
            balance_threshold = self.schedule_horizon // 5  # 20% of horizon
            self.model.Add(max_utilization - min_utilization <= balance_threshold)
    
    def _add_energy_efficiency_constraints(self):
        """Add energy efficiency constraints"""
        # Prefer scheduling tasks during off-peak hours
        peak_hours = [(8*60, 18*60)]  # 8 AM to 6 PM
        
        for task_key, task_vars in self.task_vars.items():
            start_time = task_vars['start']
            
            # Add bonus for off-peak scheduling
            for peak_start, peak_end in peak_hours:
                off_peak_bonus = self.model.NewBoolVar(f'off_peak_{task_key[0]}_{task_key[1]}')
                
                # Task is off-peak if it starts before or after peak hours
                self.model.Add(start_time < peak_start).OnlyEnforceIf(off_peak_bonus)
                self.model.Add(start_time >= peak_end).OnlyEnforceIf(off_peak_bonus.Not())
    
    def _add_enhanced_resource_constraints(self):
        """Add enhanced resource constraints"""
        # Resource capacity constraints
        for resource in self.resources:
            res_id = resource.resource_id
            
            # No overlapping tasks on same resource
            tasks_on_resource = []
            for task_key, task_vars in self.task_vars.items():
                if res_id in task_vars['resources']:
                    tasks_on_resource.append((task_vars['start'], task_vars['end'], task_vars['resources'][res_id]))
            
            # Add no-overlap constraints for tasks on same resource
            if len(tasks_on_resource) > 1:
                intervals = []
                for i, (start, end, use_var) in enumerate(tasks_on_resource):
                    interval = self.model.NewOptionalIntervalVar(
                        start, end - start, end, use_var,
                        f'interval_{res_id}_{i}'
                    )
                    intervals.append(interval)
                
                self.model.AddNoOverlap(intervals)
    
    def _add_enhanced_setup_constraints(self):
        """Add enhanced setup constraints"""
        for setup_key, setup_vars in self.setup_vars.items():
            task1_key = setup_vars['task1']
            task2_key = setup_vars['task2']
            resource_id = setup_vars['resource']
            
            task1_vars = self.task_vars[task1_key]
            task2_vars = self.task_vars[task2_key]
            
            # If both tasks use the same resource and are sequential
            both_use_resource = self.model.NewBoolVar(f'both_use_{setup_key}')
            self.model.AddBoolAnd([
                task1_vars['resources'][resource_id],
                task2_vars['resources'][resource_id]
            ]).OnlyEnforceIf(both_use_resource)
            
            # Setup time depends on product changeover
            setup_time = self._calculate_setup_time(task1_key, task2_key)
            
            # If setup is needed, add time constraint
            self.model.Add(
                task2_vars['start'] >= task1_vars['end'] + setup_time
            ).OnlyEnforceIf(setup_vars['setup'])
    
    def _add_enhanced_cip_timing_constraints(self, sku, step_id: str, task_vars: Dict):
        """Add CIP timing constraints"""
        # Find CIP tasks for this step
        for cip_id, cip_vars in self.cip_vars.items():
            if cip_vars['sku_id'] == sku.sku_id and cip_vars['step_id'] == step_id:
                # CIP must complete before task starts
                self.model.Add(cip_vars['end'] <= task_vars['start'])
                
                # CIP effectiveness affects task quality
                self.model.Add(task_vars['quality'] >= cip_vars['effectiveness'] - 10)
    
    def _create_advanced_objective(self):
        """Create advanced multi-objective function"""
        objective_terms = []
        
        # Production volume maximization
        total_production = self.model.NewIntVar(0, 1000000, 'total_production')
        production_terms = []
        for task_key, task_vars in self.task_vars.items():
            production_terms.append(task_vars['volume'])
        if production_terms:
            self.model.Add(total_production == sum(production_terms))
            objective_terms.append(total_production * int(self.weights['production_volume']))
        
        # Quality bonus
        total_quality = self.model.NewIntVar(0, 10000, 'total_quality')
        quality_terms = []
        for sku_id, quality_vars in self.quality_vars.items():
            quality_terms.append(quality_vars['overall_quality'])
        if quality_terms:
            self.model.Add(total_quality == sum(quality_terms))
            objective_terms.append(total_quality * int(self.weights['quality_bonus']))
        
        # Setup cost minimization
        total_setup_cost = self.model.NewIntVar(0, 100000, 'total_setup_cost')
        setup_cost_terms = []
        for setup_key, setup_vars in self.setup_vars.items():
            setup_cost_terms.append(setup_vars['setup'] * 100)  # Cost per setup
        if setup_cost_terms:
            self.model.Add(total_setup_cost == sum(setup_cost_terms))
            objective_terms.append(-total_setup_cost * int(self.weights['setup_cost']))
        
        # CIP cost minimization
        total_cip_cost = self.model.NewIntVar(0, 50000, 'total_cip_cost')
        cip_cost_terms = []
        for cip_id, cip_vars in self.cip_vars.items():
            cip_cost_terms.append(cip_vars['duration'] * 2)  # Cost per minute
        if cip_cost_terms:
            self.model.Add(total_cip_cost == sum(cip_cost_terms))
            objective_terms.append(-total_cip_cost * int(self.weights['cip_cost']))
        
        # Priority bonus
        priority_bonus = self.model.NewIntVar(0, 50000, 'priority_bonus')
        priority_terms = []
        for task_key, task_vars in self.task_vars.items():
            priority_value = task_vars['priority'].value if hasattr(task_vars['priority'], 'value') else 1
            priority_terms.append(task_vars['volume'] * priority_value)
        if priority_terms:
            self.model.Add(priority_bonus == sum(priority_terms))
            objective_terms.append(priority_bonus * int(self.weights['priority_bonus']))
        
        # Combine all objective terms
        if objective_terms:
            total_objective = self.model.NewIntVar(-1000000, 1000000, 'total_objective')
            self.model.Add(total_objective == sum(objective_terms))
            self.model.Maximize(total_objective)
    
    def _extract_enhanced_solution(self, status) -> SchedulingResult:
        """Extract enhanced solution with comprehensive metrics"""
        solve_time = (datetime.now() - self.solve_start_time).total_seconds()
        
        schedule_items = []
        total_production = 0
        unfulfilled_demand = {}
        
        # Extract task solutions
        for task_key, task_vars in self.task_vars.items():
            sku_id, step_id = task_key
            
            # Find selected resource
            selected_resource = None
            for res_id, res_var in task_vars['resources'].items():
                if self.solver.Value(res_var):
                    selected_resource = res_id
                    break
            
            # Find selected room
            selected_room = None
            if task_vars['rooms']:
                for room_id, room_var in task_vars['rooms'].items():
                    if self.solver.Value(room_var):
                        selected_room = room_id
                        break
            
            if selected_resource:
                start_time = self.solver.Value(task_vars['start'])
                end_time = self.solver.Value(task_vars['end'])
                volume = self.solver.Value(task_vars['volume'])
                quality = self.solver.Value(task_vars['quality'])
                efficiency = self.solver.Value(task_vars['efficiency'])
                
                schedule_item = ScheduleItem(
                    sku_id=sku_id,
                    step_id=step_id,
                    resource_id=selected_resource,
                    start_time=start_time,
                    end_time=end_time,
                    volume=volume,
                    room_id=selected_room,
                    task_type=TaskType.PRODUCTION,
                    quality_score=quality / 100.0,
                    efficiency=efficiency / 100.0,
                    priority=task_vars['priority']
                )
                
                schedule_items.append(schedule_item)
                total_production += volume
        
        # Calculate metrics
        efficiency_score = self._calculate_efficiency_score(schedule_items)
        cost_score = self._calculate_cost_score(schedule_items)
        utilization_score = self._calculate_utilization_score(schedule_items)
        quality_score = self._calculate_quality_score(schedule_items)
        makespan = max([item.end_time for item in schedule_items]) if schedule_items else 0
        
        # Calculate unfulfilled demand
        for sku in self.skus:
            produced = sum(item.volume for item in schedule_items if item.sku_id == sku.sku_id)
            if produced < sku.quantity_required:
                unfulfilled_demand[sku.sku_id] = sku.quantity_required - produced
        
        return SchedulingResult(
            schedule_items=schedule_items,
            total_production=total_production,
            efficiency_score=efficiency_score,
            cost_score=cost_score,
            utilization_score=utilization_score,
            quality_score=quality_score,
            makespan=makespan,
            unfulfilled_demand=unfulfilled_demand,
            warnings=self.warnings.copy(),
            solver_status=self.solver.StatusName(status),
            solve_time=solve_time
        )
    
    def _calculate_composite_score(self, result: SchedulingResult) -> float:
        """Calculate composite optimization score"""
        score = 0.0
        
        # Production volume score (normalized)
        if result.total_production > 0:
            score += result.total_production * 0.01
        
        # Quality score
        score += result.quality_score * 100
        
        # Efficiency score
        score += result.efficiency_score * 10
        
        # Utilization score
        score += result.utilization_score * 5
        
        # Penalty for unfulfilled demand
        unfulfilled_penalty = sum(result.unfulfilled_demand.values()) * 50
        score -= unfulfilled_penalty
        
        # Makespan penalty (prefer shorter schedules)
        score -= result.makespan * 0.01
        
        return score
    
    def _calculate_efficiency_score(self, schedule_items: List[ScheduleItem]) -> float:
        """Calculate overall efficiency score"""
        if not schedule_items:
            return 0.0
        
        total_efficiency = sum(item.efficiency for item in schedule_items)
        return (total_efficiency / len(schedule_items)) * 100
    
    def _calculate_cost_score(self, schedule_items: List[ScheduleItem]) -> float:
        """Calculate cost score (lower is better)"""
        total_cost = 0.0
        
        for item in schedule_items:
            # Base production cost
            total_cost += item.volume * 0.1
            
            # Setup cost
            total_cost += item.setup_time * 0.5
            
            # CIP cost
            total_cost += item.cip_time * 0.8
        
        return total_cost
    
    def _calculate_utilization_score(self, schedule_items: List[ScheduleItem]) -> float:
        """Calculate resource utilization score"""
        if not schedule_items:
            return 0.0
        
        resource_usage = defaultdict(int)
        
        for item in schedule_items:
            resource_usage[item.resource_id] += item.duration
        
        if not resource_usage:
            return 0.0
        
        total_available = len(self.resources) * self.schedule_horizon
        total_used = sum(resource_usage.values())
        
        return (total_used / total_available) * 100
    
    def _calculate_quality_score(self, schedule_items: List[ScheduleItem]) -> float:
        """Calculate overall quality score"""
        if not schedule_items:
            return 0.0
        
        total_quality = sum(item.quality_score for item in schedule_items)
        return total_quality / len(schedule_items)
    
    def _create_empty_result(self) -> SchedulingResult:
        """Create empty result for failed optimization"""
        return SchedulingResult(
            schedule_items=[],
            total_production=0.0,
            efficiency_score=0.0,
            cost_score=0.0,
            utilization_score=0.0,
            quality_score=0.0,
            makespan=0,
            unfulfilled_demand={sku.sku_id: sku.quantity_required for sku in self.skus},
            warnings=self.warnings.copy(),
            solver_status="FAILED",
            solve_time=0.0
        )
    
    def _reset_model(self):
        """Reset model for next iteration"""
        self.model = cp_model.CpModel()
        self.task_vars.clear()
        self.cip_vars.clear()
        self.setup_vars.clear()
        self.sequence_vars.clear()
        self.quality_vars.clear()
        self.cost_vars.clear()
    
    # Helper methods
    def _get_enhanced_compatible_resources(self, step_id: str, sku) -> Dict[str, float]:
        """Get compatible resources with capability scores"""
        compatible = {}
        
        for resource in self.resources:
            if self._is_resource_compatible(resource, step_id, sku):
                # Calculate capability score based on capacity, efficiency, etc.
                capability_score = self._calculate_capability_score(resource, step_id, sku)
                compatible[resource.resource_id] = capability_score
        
        return compatible
    
    def _get_enhanced_compatible_rooms(self, step_id: str, sku) -> Dict[str, float]:
        """Get compatible rooms with capacity scores"""
        compatible = {}
        
        for room in self.rooms:
            if self._is_room_compatible(room, step_id, sku):
                capacity_score = getattr(room, 'capacity', 100)
                compatible[room.room_id] = capacity_score
        
        return compatible
    
    def _is_resource_compatible(self, resource, step_id: str, sku) -> bool:
        """Check if resource is compatible with step"""
        # Basic compatibility check
        return hasattr(resource, 'compatible_steps') and step_id in resource.compatible_steps
    
    def _is_room_compatible(self, room, step_id: str, sku) -> bool:
        """Check if room is compatible with step"""
        return hasattr(room, 'compatible_steps') and step_id in room.compatible_steps
    
    def _calculate_capability_score(self, resource, step_id: str, sku) -> float:
        """Calculate resource capability score"""
        score = 1.0
        
        # Base capacity score
        score *= getattr(resource, 'capacity', 100) / 100
        
        # Efficiency multiplier
        score *= getattr(resource, 'efficiency', 1.0)
        
        return score
    
    def _step_needs_room(self, step_id: str) -> bool:
        """Check if step requires a room"""
        room_required_steps = ['fermentation', 'aging', 'storage', 'packaging']
        return any(req_step in step_id.lower() for req_step in room_required_steps)
    
    def _needs_cip_before_task(self, sku, step_id: str) -> bool:
        """Check if CIP is needed before task"""
        cip_required_steps = ['fermentation', 'mixing', 'filling']
        return any(req_step in step_id.lower() for req_step in cip_required_steps)
    
    def _get_required_cip_circuit(self, step_id: str) -> str:
        """Get required CIP circuit for step"""
        if 'fermentation' in step_id.lower():
            return 'fermentation_cip'
        elif 'filling' in step_id.lower():
            return 'filling_cip'
        else:
            return 'general_cip'
    
    def _calculate_buffer_time(self, prev_step: str, current_step: str) -> int:
        """Calculate buffer time between steps"""
        # Default buffer time
        buffer = 15
        
        # Longer buffer for critical transitions
        if 'fermentation' in prev_step.lower() and 'packaging' in current_step.lower():
            buffer = 60
        elif 'mixing' in prev_step.lower():
            buffer = 30
        
        return buffer
    
    def _calculate_setup_time(self, task1_key: Tuple[str, str], task2_key: Tuple[str, str]) -> int:
        """Calculate setup time between tasks"""
        # Default setup time
        setup_time = self.default_setup_time
        
        # Longer setup for different products
        sku1_product = next(p for p in self.products if any(s.sku_id == task1_key[0] for s in self.skus) and s.product_id == p.product_id)
        sku2_product = next(p for p in self.products if any(s.sku_id == task2_key[0] for s in self.skus) and s.product_id == p.product_id)
        
        if sku1_product.product_id != sku2_product.product_id:
            setup_time = 90  # Longer setup for product changeover
        
        return setup_time
    
    def _group_similar_resources(self) -> Dict[str, List[str]]:
        """Group similar resources for load balancing"""
        groups = defaultdict(list)
        
        for resource in self.resources:
            # Group by resource type or capability
            resource_type = getattr(resource, 'resource_type', 'general')
            groups[resource_type].append(resource.resource_id)
        
        return dict(groups)
        