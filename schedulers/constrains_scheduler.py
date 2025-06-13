import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
from collections import defaultdict
import heapq
from itertools import combinations
from schedulers import max_flow_scheduler, proportional_scheduler, pulp_scheduler
from scheduler import ProductionScheduler
import config
from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductTypeRegistry, FlowEdge, ProductionSlot
)

logger = logging.getLogger(__name__)

# Additional OR Technique: Constraint Programming Scheduler
class ConstraintProgrammingScheduler:
    """Alternative scheduler using Constraint Programming principles"""
    
    def __init__(self):
        self.variables = {}  # Decision variables
        self.constraints = []  # Constraint set
        self.objective_terms = []  # Objective function terms
        
    def schedule_production_cp(self) -> SchedulingResult:
        """Schedule using constraint programming approach"""
        logger.info("Starting constraint programming scheduling")
        
        # Step 1: Define decision variables
        self._define_variables()
        
        # Step 2: Add constraints
        self._add_constraints()
        
        # Step 3: Define objective function
        self._define_objective()
        
        # Step 4: Solve using backtracking with constraint propagation
        solution = self._solve_with_backtracking()
        
        # Step 5: Convert solution to schedule
        return self._convert_cp_solution(solution)
    
    def _define_variables(self):
        """Define decision variables for CP model"""
        # Binary variables: x[i,j,k,t] = 1 if indent i is assigned to line j, tank k at time slot t
        self.variables = {}
        
        time_slots = list(range(0, 24 * 60 * self.max_lookahead_days, 60))  # Hourly slots
        
        for indent_id in config.USER_INDENTS:
            for line_id in config.LINES:
                for tank_id in config.TANKS:
                    for t in time_slots:
                        var_name = f"x_{indent_id}_{line_id}_{tank_id}_{t}"
                        self.variables[var_name] = {
                            'domain': [0, 1],
                            'value': None,
                            'indent_id': indent_id,
                            'line_id': line_id,
                            'tank_id': tank_id,
                            'time_slot': t
                        }
    
    def _add_constraints(self):
        """Add constraints to the CP model"""
        # Constraint 1: Each indent must be assigned exactly once (or not at all)
        for indent_id in config.USER_INDENTS:
            constraint_vars = [var for var_name, var in self.variables.items() 
                             if var['indent_id'] == indent_id]
            self.constraints.append({
                'type': 'sum_leq',
                'variables': constraint_vars,
                'limit': 1,
                'description': f'Indent {indent_id} assigned at most once'
            })
        
        # Constraint 2: Line capacity constraints
        for line_id in config.LINES:
            for t in range(0, 24 * 60 * self.max_lookahead_days, 60):
                constraint_vars = [var for var_name, var in self.variables.items() 
                                 if var['line_id'] == line_id and var['time_slot'] == t]
                self.constraints.append({
                    'type': 'sum_leq',
                    'variables': constraint_vars,
                    'limit': 1,
                    'description': f'Line {line_id} capacity at time {t}'
                })
        
        # Constraint 3: Tank capacity constraints
        for tank_id in config.TANKS:
            tank = config.TANKS[tank_id]
            for t in range(0, 24 * 60 * self.max_lookahead_days, 60):
                constraint_vars = []
                for var_name, var in self.variables.items():
                    if var['tank_id'] == tank_id and var['time_slot'] <= t:
                        indent = config.USER_INDENTS[var['indent_id']]
                        constraint_vars.append((var, indent.qty_required))
                
                self.constraints.append({
                    'type': 'weighted_sum_leq',
                    'variables': constraint_vars,
                    'limit': tank.capacity_liters,
                    'description': f'Tank {tank_id} capacity at time {t}'
                })
        
        # Constraint 4: Compatibility constraints
        for var_name, var in self.variables.items():
            indent = config.USER_INDENTS[var['indent_id']]
            sku = config.SKUS.get(indent.sku_id)
            line = config.LINES[var['line_id']]
            tank = config.TANKS[var['tank_id']]
            
            if not sku or sku.sku_id not in line.compatible_skus_max_production:
                var['domain'] = [0]  # Force to 0 if incompatible
            
            if not sku or not tank.can_store(indent.qty_required, sku):
                var['domain'] = [0]  # Force to 0 if tank incompatible
    
    def _define_objective(self):
        """Define objective function for maximizing production and utilization"""
        self.objective_terms = []
        
        for var_name, var in self.variables.items():
            if var['domain'] == [0]:  # Skip infeasible variables
                continue
                
            indent = config.USER_INDENTS[var['indent_id']]
            sku = config.SKUS.get(indent.sku_id)
            
            if not sku:
                continue
            
            # Positive coefficient for production volume
            volume_benefit = indent.qty_required * config.PENALTY_WEIGHTS.get('production_benefit', 1.0)
            
            # Priority benefit
            priority_benefit = (4 - indent.priority.value) * config.PENALTY_WEIGHTS.get('priority_benefit', 10.0)
            
            # Due date urgency
            due_date_benefit = 0.0
            if indent.due_date:
                slot_time = self.run_start_time + timedelta(minutes=var['time_slot'])
                hours_to_due = (indent.due_date - slot_time).total_seconds() / 3600
                if hours_to_due > 0:
                    due_date_benefit = min(24, hours_to_due) * config.PENALTY_WEIGHTS.get('due_date_benefit', 1.0)
            
            total_benefit = volume_benefit + priority_benefit + due_date_benefit
            
            self.objective_terms.append({
                'variable': var,
                'coefficient': total_benefit
            })
    
    def _solve_with_backtracking(self) -> Dict[str, int]:
        """Solve CP model using backtracking with constraint propagation"""
        logger.info("Solving CP model with backtracking")
        
        # Initialize domains
        domains = {var_name: var['domain'].copy() for var_name, var in self.variables.items()}
        
        # Apply initial constraint propagation
        self._propagate_constraints(domains)
        
        # Solve using backtracking
        solution = {}
        if self._backtrack(domains, solution, 0):
            return solution
        else:
            logger.warning("No feasible solution found with CP")
            return {}
    
    def _propagate_constraints(self, domains: Dict[str, List[int]]) -> bool:
        """Apply constraint propagation to reduce domains"""
        changed = True
        
        while changed:
            changed = False
            
            for constraint in self.constraints:
                if constraint['type'] == 'sum_leq':
                    # Arc consistency for sum constraints
                    variables = constraint['variables']
                    limit = constraint['limit']
                    
                    # Count minimum and maximum possible values
                    min_sum = sum(min(domains[f"{var['indent_id']}_{var['line_id']}_{var['tank_id']}_{var['time_slot']}"]) 
                                for var in variables)
                    max_sum = sum(max(domains[f"{var['indent_id']}_{var['line_id']}_{var['tank_id']}_{var['time_slot']}"]) 
                                for var in variables)
                    
                    if min_sum > limit:
                        return False  # Infeasible
                    
                    # If max_sum equals limit, all variables must take maximum value
                    if max_sum == limit:
                        for var in variables:
                            var_name = f"{var['indent_id']}_{var['line_id']}_{var['tank_id']}_{var['time_slot']}"
                            if len(domains[var_name]) > 1:
                                domains[var_name] = [max(domains[var_name])]
                                changed = True
        
        return True
    
    def _backtrack(self, domains: Dict[str, List[int]], solution: Dict[str, int], var_index: int) -> bool:
        """Backtracking search with constraint checking"""
        var_names = list(self.variables.keys())
        
        if var_index == len(var_names):
            return True  # All variables assigned
        
        var_name = var_names[var_index]
        
        # Try each value in the domain
        for value in domains[var_name]:
            solution[var_name] = value
            
            # Check if assignment satisfies constraints
            if self._is_consistent(solution, var_name):
                # Create new domains for forward checking
                new_domains = self._forward_check(domains, var_name, value)
                
                if new_domains is not None:
                    # Recurse
                    if self._backtrack(new_domains, solution, var_index + 1):
                        return True
            
            # Backtrack
            del solution[var_name]
        
        return False
    
    def _is_consistent(self, solution: Dict[str, int], var_name: str) -> bool:
        """Check if current partial solution is consistent with constraints"""
        for constraint in self.constraints:
            if constraint['type'] == 'sum_leq':
                variables = constraint['variables']
                limit = constraint['limit']
                
                assigned_sum = 0
                unassigned_count = 0
                
                for var in variables:
                    var_key = f"{var['indent_id']}_{var['line_id']}_{var['tank_id']}_{var['time_slot']}"
                    if var_key in solution:
                        assigned_sum += solution[var_key]
                    else:
                        unassigned_count += 1
                
                # Check if constraint can still be satisfied
                if assigned_sum > limit:
                    return False
                
                # If all variables are assigned, check exact constraint
                if unassigned_count == 0 and assigned_sum > limit:
                    return False
        
        return True
    
    def _forward_check(self, domains: Dict[str, List[int]], assigned_var: str, assigned_value: int) -> Optional[Dict[str, List[int]]]:
        """Forward checking to prune domains of unassigned variables"""
        new_domains = {k: v.copy() for k, v in domains.items()}
        new_domains[assigned_var] = [assigned_value]
        
        # Apply constraint propagation
        if self._propagate_constraints(new_domains):
            return new_domains
        else:
            return None
    
    def _convert_cp_solution(self, solution: Dict[str, int]) -> SchedulingResult:
        """Convert CP solution to SchedulingResult"""
        scheduled_items = []
        unfulfilled_indents = []
        
        assigned_indents = set()
        
        for var_name, value in solution.items():
            if value == 1:  # Variable is assigned
                var = self.variables[var_name]
                indent = config.USER_INDENTS[var['indent_id']]
                sku = config.SKUS[indent.sku_id]
                line = config.LINES[var['line_id']]
                tank = config.TANKS[var['tank_id']]
                shift = self._get_shift_for_time(var['time_slot'])
                
                # Calculate timing
                start_time = self.run_start_time + timedelta(minutes=var['time_slot'])
                production_rate = line.compatible_skus_max_production[sku.sku_id]
                production_time = indent.qty_required / production_rate
                end_time = start_time + timedelta(minutes=production_time)
                
                # Create schedule item
                schedule_item = ScheduleItem(
                    sku=sku,
                    line=line,
                    tank=tank,
                    shift=shift,
                    start_time=start_time,
                    end_time=end_time,
                    quantity=indent.qty_required,
                    produced_quantity=indent.qty_required,
                    setup_time_minutes=self._calculate_setup_time_cp(line, sku),
                    cip_time_minutes=self._calculate_cip_time_cp(line, sku)
                )
                
                scheduled_items.append(schedule_item)
                assigned_indents.add(var['indent_id'])
        
        # Find unfulfilled indents
        for indent_id, indent in config.USER_INDENTS.items():
            if indent_id not in assigned_indents:
                unfulfilled_indents.append(indent)
        
        total_production = sum(item.quantity for item in scheduled_items)
        efficiency_score = self._calculate_efficiency_score_cp(scheduled_items)
        
        return SchedulingResult(
            schedule_items=scheduled_items,
            unfulfilled_indents=unfulfilled_indents,
            total_production=total_production,
            efficiency_score=efficiency_score,
            warnings=[]
        )
    
    def _get_shift_for_time(self, time_slot: int) -> Shift:
        """Get appropriate shift for a given time slot"""
        slot_time = (self.run_start_time + timedelta(minutes=time_slot)).time()
        
        for shift in config.SHIFTS.values():
            if shift.start_time.time() <= slot_time <= shift.end_time.time():
                return shift
        
        # Default to first available shift
        return list(config.SHIFTS.values())[0]
    
    def _calculate_setup_time_cp(self, line: Line, sku: SKU) -> int:
        """Calculate setup time for CP solution"""
        if not line.current_sku or line.current_sku.sku_id == sku.sku_id:
            return 0
        
        current_sku = config.SKUS.get(line.current_sku.sku_id)
        if not current_sku or current_sku.product_category != sku.product_category:
            return 180
        
        return sku.setup_time
    
    def _calculate_cip_time_cp(self, line: Line, sku: SKU) -> int:
        """Calculate CIP time for CP solution"""
        return 180 if line.needs_cip(sku) else 0
    
    def _calculate_efficiency_score_cp(self, scheduled_items: List[ScheduleItem]) -> float:
        """Calculate efficiency score for CP solution"""
        if not scheduled_items:
            return 0.0
        
        total_production_time = 0.0
        total_overhead_time = 0.0
        
        for item in scheduled_items:
            production_time = (item.end_time - item.start_time).total_seconds() / 60
            total_production_time += production_time
            total_overhead_time += item.setup_time_minutes + item.cip_time_minutes
        
        total_time = total_production_time + total_overhead_time
        
        if total_time == 0:
            return 0.0
        
        return (total_production_time / total_time) * 100
