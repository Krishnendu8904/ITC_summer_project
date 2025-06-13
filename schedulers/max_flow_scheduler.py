import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
from collections import defaultdict
import heapq
from itertools import combinations

import config
from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductTypeRegistry, FlowEdge, ProductionSlot
)
logger = logging.getLogger(__name__)

class MaxFlowProductionScheduler:
    """Enhanced Production Scheduler using Maximum Flow and Linear Programming concepts"""
    
    def __init__(self):
        self.scheduled_items: List[ScheduleItem] = []
        self.unfulfilled_indents: List[UserIndent] = []
        self.warnings: List[str] = []
        self.run_start_time: datetime = datetime.now()
        
        # Flow network components
        self.edges: List[FlowEdge] = []
        self.nodes: Set[str] = set()
        self.adjacency: Dict[str, List[FlowEdge]] = defaultdict(list)
        
        # Production slots for time-based scheduling
        self.production_slots: List[ProductionSlot] = []
        
        # Optimization parameters
        self.time_slot_duration = 60  # minutes per time slot
        self.max_lookahead_days = 7   # Planning horizon

    def schedule_production(self) -> SchedulingResult:
        """Main scheduling method using OR techniques"""
        logger.info("Starting enhanced production scheduling with OR techniques")
        
        # Reset state
        self._reset_state()
        
        # Step 1: Generate all feasible production slots
        self._generate_production_slots()
        
        # Step 2: Build flow network
        self._build_flow_network()
        
        # Step 3: Solve maximum flow problem
        max_flow_result = self._solve_max_flow()
        
        # Step 4: Convert flow solution to schedule
        self._convert_flow_to_schedule(max_flow_result)
        
        # Step 5: Apply post-optimization techniques
        self._apply_local_optimization()
        
        return self._generate_result()

    def _reset_state(self):
        """Reset internal state for new scheduling run"""
        self.scheduled_items = []
        self.unfulfilled_indents = []
        self.warnings = []
        self.run_start_time = datetime.now()
        self.edges = []
        self.nodes = set()
        self.adjacency = defaultdict(list)
        self.production_slots = []

    def _generate_production_slots(self):
        """Generate all feasible production time slots"""
        logger.info("Generating production slots")
        
        end_horizon = self.run_start_time + timedelta(days=self.max_lookahead_days)
        
        for line_id, line in config.LINES.items():
            if not line.is_available():
                continue
                
            for tank_id, tank in config.TANKS.items():
                for shift_id, shift in config.SHIFTS.items():
                    if not shift.is_active():
                        continue
                    
                    # Generate time slots within each shift
                    current_date = self.run_start_time.date()
                    
                    for day_offset in range(self.max_lookahead_days):
                        slot_date = current_date + timedelta(days=day_offset)
                        
                        shift_start = datetime.combine(slot_date, shift.start_time.time())
                        shift_end = datetime.combine(slot_date, shift.end_time.time())
                        
                        # Handle overnight shifts
                        if shift_end <= shift_start:
                            shift_end += timedelta(days=1)
                        
                        # Skip if shift is in the past
                        if shift_end <= self.run_start_time:
                            continue
                        
                        # Adjust start time if in the past
                        if shift_start < self.run_start_time:
                            shift_start = self.run_start_time
                        
                        # Generate slots within this shift
                        current_slot_start = shift_start
                        
                        while current_slot_start + timedelta(minutes=self.time_slot_duration) <= shift_end:
                            slot_end = current_slot_start + timedelta(minutes=self.time_slot_duration)
                            
                            # Calculate maximum production capacity for this slot
                            max_capacity = self._calculate_slot_capacity(line_id, tank_id, self.time_slot_duration)
                            
                            if max_capacity > 0:
                                slot = ProductionSlot(
                                    line_id=line_id,
                                    tank_id=tank_id,
                                    shift_id=shift_id,
                                    start_time=current_slot_start,
                                    end_time=slot_end,
                                    capacity=max_capacity
                                )
                                self.production_slots.append(slot)
                            
                            current_slot_start = slot_end
        
        logger.info(f"Generated {len(self.production_slots)} production slots")

    def _calculate_slot_capacity(self, line_id: str, tank_id: str, duration_minutes: int) -> float:
        """Calculate maximum production capacity for a slot"""
        line = config.LINES[line_id]
        tank = config.TANKS[tank_id]
        
        # Find the maximum production rate across all compatible SKUs
        max_rate = 0.0
        for sku_id, rate in line.compatible_skus_max_production.items():
            if rate > max_rate:
                # Check if tank can store this product type
                sku = config.SKUS.get(sku_id)
                if sku and tank.can_store(rate * duration_minutes, sku):
                    max_rate = rate
        
        return max_rate * duration_minutes

    def _build_flow_network(self):
        """Build the flow network for production scheduling"""
        logger.info("Building flow network")
        
        # Nodes: source, demands, production_slots, lines, tanks, sink
        source = "SOURCE"
        sink = "SINK"
        self.nodes.add(source)
        self.nodes.add(sink)
        
        # Add demand nodes
        demand_nodes = []
        for indent_id, indent in config.USER_INDENTS.items():
            demand_node = f"DEMAND_{indent_id}"
            demand_nodes.append(demand_node)
            self.nodes.add(demand_node)
            
            # Edge from source to demand (capacity = demand quantity)
            self._add_edge(source, demand_node, indent.qty_required, self._get_demand_priority_cost(indent))
        
        # Add production slot nodes
        slot_nodes = []
        for i, slot in enumerate(self.production_slots):
            slot_node = f"SLOT_{i}"
            slot_nodes.append(slot_node)
            self.nodes.add(slot_node)
            
            # Edge from slot to sink (capacity = slot capacity)
            self._add_edge(slot_node, sink, slot.capacity, 0)
        
        # Connect demands to compatible production slots
        for demand_node in demand_nodes:
            indent_id = demand_node.split("_")[1]
            indent = config.USER_INDENTS[indent_id]
            sku = config.SKUS.get(indent.sku_id)
            
            if not sku:
                continue
            
            for i, slot in enumerate(self.production_slots):
                slot_node = f"SLOT_{i}"
                
                # Check compatibility
                if self._is_slot_compatible(slot, sku, indent):
                    # Calculate cost based on timing, setup, efficiency
                    cost = self._calculate_assignment_cost_enhanced(slot, sku, indent)
                    
                    # Capacity is minimum of remaining demand and slot capacity
                    capacity = min(indent.qty_required, slot.capacity)
                    
                    self._add_edge(demand_node, slot_node, capacity, cost)

    def _add_edge(self, from_node: str, to_node: str, capacity: float, cost: float):
        """Add an edge to the flow network"""
        edge = FlowEdge(from_node, to_node, capacity, cost)
        self.edges.append(edge)
        self.adjacency[from_node].append(edge)
        
        # Add reverse edge with 0 capacity for flow algorithms
        reverse_edge = FlowEdge(to_node, from_node, 0, -cost)
        self.edges.append(reverse_edge)
        self.adjacency[to_node].append(reverse_edge)

    def _get_demand_priority_cost(self, indent: UserIndent) -> float:
        """Get cost based on demand priority"""
        priority_costs = {
            Priority.HIGH: 1.0,
            Priority.MEDIUM: 2.0,
            Priority.LOW: 3.0
        }
        return priority_costs.get(indent.priority, 2.0)

    def _is_slot_compatible(self, slot: ProductionSlot, sku: SKU, indent: UserIndent) -> bool:
        """Check if a production slot is compatible with a demand"""
        line = config.LINES[slot.line_id]
        tank = config.TANKS[slot.tank_id]
        
        # Check line compatibility
        if sku.sku_id not in line.compatible_skus_max_production:
            return False
        
        production_rate = line.compatible_skus_max_production[sku.sku_id]
        if production_rate <= 0:
            return False
        
        # Check tank compatibility
        if not tank.can_store(indent.qty_required, sku):
            return False
        
        # Check if slot can produce minimum required quantity
        slot_duration = (slot.end_time - slot.start_time).total_seconds() / 60
        max_production = production_rate * slot_duration
        
        return max_production >= min(indent.qty_required * 0.1, 100)  # At least 10% or 100L

    def _calculate_assignment_cost_enhanced(self, slot: ProductionSlot, sku: SKU, indent: UserIndent) -> float:
        """Calculate enhanced assignment cost for OR optimization"""
        cost = 0.0
        
        # Time-based costs
        if indent.due_date:
            time_diff = (indent.due_date - slot.end_time).total_seconds() / 3600
            if time_diff < 0:  # Late
                cost += abs(time_diff) * config.PENALTY_WEIGHTS.get('lateness_cost', 10.0)
            elif time_diff < 24:  # Within 24 hours
                cost += (24 - time_diff) * config.PENALTY_WEIGHTS.get('urgency_bonus', -2.0)
        
        # Setup and changeover costs
        line = config.LINES[slot.line_id]
        setup_cost = self._estimate_setup_cost(line, sku) * config.PENALTY_WEIGHTS.get('line_setup_cost', 0.5)
        cost += setup_cost
        
        # Efficiency costs (prefer high utilization)
        production_rate = line.compatible_skus_max_production[sku.sku_id]
        slot_duration = (slot.end_time - slot.start_time).total_seconds() / 60
        utilization = min(indent.qty_required / (production_rate * slot_duration), 1.0)
        cost += (1.0 - utilization) * config.PENALTY_WEIGHTS.get('inefficiency_cost', 5.0)
        
        # Tank utilization costs
        tank = config.TANKS[slot.tank_id]
        tank_utilization = indent.qty_required / tank.capacity_liters
        if tank_utilization < 0.3:
            cost += config.PENALTY_WEIGHTS.get('tank_underutilization_cost', 3.0)
        
        return cost

    def _estimate_setup_cost(self, line: Line, sku: SKU) -> float:
        """Estimate setup cost for line changeover"""
        if not line.current_sku or line.current_sku.sku_id == sku.sku_id:
            return 0.0
        
        current_sku = config.SKUS.get(line.current_sku.sku_id)
        if not current_sku or current_sku.product_category != sku.product_category:
            return 180.0
        
        return sku.setup_time

    def _solve_max_flow(self) -> Dict[str, float]:
        """Solve maximum flow problem using Ford-Fulkerson with Edmonds-Karp"""
        logger.info("Solving maximum flow problem")
        
        # Implementation of Edmonds-Karp algorithm for maximum flow
        max_flow = 0.0
        flow_dict = {}
        
        # Initialize flow on all edges to 0
        for edge in self.edges:
            flow_dict[f"{edge.from_node}->{edge.to_node}"] = 0.0
        
        while True:
            # Find augmenting path using BFS
            path, path_flow = self._find_augmenting_path("SOURCE", "SINK", flow_dict)
            
            if not path or path_flow == 0:
                break
            
            # Update flow along the path
            for i in range(len(path) - 1):
                edge_key = f"{path[i]}->{path[i+1]}"
                reverse_key = f"{path[i+1]}->{path[i]}"
                
                if edge_key in flow_dict:
                    flow_dict[edge_key] += path_flow
                else:
                    flow_dict[reverse_key] -= path_flow
            
            max_flow += path_flow
        
        logger.info(f"Maximum flow achieved: {max_flow}")
        return flow_dict

    def _find_augmenting_path(self, source: str, sink: str, flow_dict: Dict[str, float]) -> Tuple[List[str], float]:
        """Find augmenting path using BFS"""
        from collections import deque
        
        queue = deque([(source, [source], float('inf'))])
        visited = {source}
        
        while queue:
            node, path, flow = queue.popleft()
            
            if node == sink:
                return path, flow
            
            for edge in self.adjacency[node]:
                if edge.to_node not in visited:
                    edge_key = f"{edge.from_node}->{edge.to_node}"
                    current_flow = flow_dict.get(edge_key, 0.0)
                    residual_capacity = edge.capacity - current_flow
                    
                    if residual_capacity > 0:
                        visited.add(edge.to_node)
                        new_flow = min(flow, residual_capacity)
                        queue.append((edge.to_node, path + [edge.to_node], new_flow))
        
        return [], 0.0

    def _convert_flow_to_schedule(self, flow_result: Dict[str, float]):
        """Convert flow solution to actual schedule"""
        logger.info("Converting flow solution to schedule")
        
        for flow_key, flow_amount in flow_result.items():
            if flow_amount <= 0 or "->" not in flow_key:
                continue
            
            from_node, to_node = flow_key.split("->")
            
            # Find demand to slot assignments
            if from_node.startswith("DEMAND_") and to_node.startswith("SLOT_"):
                indent_id = from_node.split("_")[1]
                slot_index = int(to_node.split("_")[1])
                
                if indent_id in config.USER_INDENTS and slot_index < len(self.production_slots):
                    indent = config.USER_INDENTS[indent_id]
                    slot = self.production_slots[slot_index]
                    sku = config.SKUS.get(indent.sku_id)
                    
                    if sku:
                        # Create schedule item
                        schedule_item = self._create_schedule_item(indent, sku, slot, flow_amount)
                        if schedule_item:
                            self.scheduled_items.append(schedule_item)

    def _create_schedule_item(self, indent: UserIndent, sku: SKU, slot: ProductionSlot, quantity: float) -> Optional[ScheduleItem]:
        """Create a schedule item from flow assignment"""
        try:
            line = config.LINES[slot.line_id]
            tank = config.TANKS[slot.tank_id]
            shift = config.SHIFTS[slot.shift_id]
            
            # Calculate actual production time needed
            production_rate = line.compatible_skus_max_production[sku.sku_id]
            production_time_minutes = quantity / production_rate
            
            # Adjust end time based on actual production time
            actual_end_time = slot.start_time + timedelta(minutes=production_time_minutes)
            if actual_end_time > slot.end_time:
                actual_end_time = slot.end_time
                # Adjust quantity if needed
                max_possible = production_rate * ((slot.end_time - slot.start_time).total_seconds() / 60)
                quantity = min(quantity, max_possible)
            
            # Calculate setup and CIP times
            setup_time = self._calculate_setup_time(slot.line_id, sku)
            cip_time = self._calculate_cip_time(slot.line_id, sku)
            
            schedule_item = ScheduleItem(
                sku=sku,
                line=line,
                tank=tank,
                shift=shift,
                start_time=slot.start_time,
                end_time=actual_end_time,
                quantity=quantity,
                produced_quantity=quantity,
                setup_time_minutes=setup_time,
                cip_time_minutes=cip_time
            )
            
            return schedule_item
            
        except Exception as e:
            logger.error(f"Error creating schedule item: {e}")
            return None

    def _calculate_setup_time(self, line_id: str, sku: SKU) -> int:
        """Calculate setup time for line changeover"""
        line = config.LINES[line_id]
        
        if not line.current_sku:
            return sku.setup_time
        
        if line.current_sku.sku_id == sku.sku_id:
            return 0
        
        current_sku = config.SKUS.get(line.current_sku.sku_id)
        if current_sku is None or current_sku.product_category != sku.product_category:
            return 180
        
        return sku.setup_time

    def _calculate_cip_time(self, line_id: str, sku: SKU) -> int:
        """Calculate CIP time if needed"""
        line = config.LINES[line_id]
        return 180 if line.needs_cip(sku) else 0

    def _apply_local_optimization(self):
        """Apply local optimization techniques to improve the schedule"""
        logger.info("Applying local optimization")
        
        # Sort scheduled items by start time
        self.scheduled_items.sort(key=lambda x: x.start_time)
        
        # Apply 2-opt improvements
        self._apply_2opt_improvement()
        
        # Consolidate adjacent operations on same line
        self._consolidate_adjacent_operations()
        
        # Update system state
        for item in self.scheduled_items:
            self._update_system_state(item)

    def _apply_2opt_improvement(self):
        """Apply 2-opt local search improvement"""
        improved = True
        while improved:
            improved = False
            
            for i in range(len(self.scheduled_items)):
                for j in range(i + 2, len(self.scheduled_items)):
                    if self._can_swap_operations(i, j):
                        original_cost = self._calculate_schedule_cost()
                        
                        # Try swapping
                        self.scheduled_items[i], self.scheduled_items[j] = self.scheduled_items[j], self.scheduled_items[i]
                        
                        new_cost = self._calculate_schedule_cost()
                        
                        if new_cost < original_cost:
                            improved = True
                        else:
                            # Revert swap
                            self.scheduled_items[i], self.scheduled_items[j] = self.scheduled_items[j], self.scheduled_items[i]

    def _can_swap_operations(self, i: int, j: int) -> bool:
        """Check if two operations can be swapped"""
        item_i = self.scheduled_items[i]
        item_j = self.scheduled_items[j]
        
        # Don't swap if they're on the same line and would overlap
        if (item_i.line.line_id == item_j.line.line_id and 
            abs((item_i.start_time - item_j.start_time).total_seconds()) < 
            (item_i.end_time - item_i.start_time).total_seconds()):
            return False
        
        return True

    def _calculate_schedule_cost(self) -> float:
        """Calculate total cost of current schedule"""
        total_cost = 0.0
        
        for item in self.scheduled_items:
            # Setup costs
            total_cost += item.setup_time_minutes * config.PENALTY_WEIGHTS.get('line_setup_cost', 0.5)
            
            # CIP costs  
            total_cost += item.cip_time_minutes * config.PENALTY_WEIGHTS.get('tank_cip_cost', 1.0)
            
            # Utilization costs
            line = config.LINES[item.line.line_id]
            production_rate = line.compatible_skus_max_production.get(item.sku.sku_id, 0)
            if production_rate > 0:
                duration_minutes = (item.end_time - item.start_time).total_seconds() / 60
                utilization = item.quantity / (production_rate * duration_minutes)
                total_cost += (1.0 - min(utilization, 1.0)) * config.PENALTY_WEIGHTS.get('inefficiency_cost', 5.0)
        
        return total_cost

    def _consolidate_adjacent_operations(self):
        """Consolidate adjacent operations on the same line for the same SKU"""
        consolidated = []
        i = 0
        
        while i < len(self.scheduled_items):
            current = self.scheduled_items[i]
            
            # Look for adjacent operations that can be consolidated
            j = i + 1
            while (j < len(self.scheduled_items) and 
                   self.scheduled_items[j].line.line_id == current.line.line_id and
                   self.scheduled_items[j].sku.sku_id == current.sku.sku_id and
                   abs((self.scheduled_items[j].start_time - current.end_time).total_seconds()) < 300):  # Within 5 minutes
                
                # Consolidate operations
                current.end_time = self.scheduled_items[j].end_time
                current.quantity += self.scheduled_items[j].quantity
                current.produced_quantity += self.scheduled_items[j].produced_quantity
                j += 1
            
            consolidated.append(current)
            i = j
        
        self.scheduled_items = consolidated

    def _update_system_state(self, schedule_item: ScheduleItem):
        """Update system state after scheduling"""
        line = config.LINES[schedule_item.line.line_id]
        tank = config.TANKS[schedule_item.tank.tank_id]
        
        line.current_sku = schedule_item.sku
        line.status = LineStatus.ACTIVE
        
        tank.current_product = schedule_item.sku.product_category
        tank.current_volume += schedule_item.quantity

    def _generate_result(self) -> SchedulingResult:
        """Generate final scheduling result"""
        # Find unfulfilled indents
        scheduled_quantities = defaultdict(float)
        for item in self.scheduled_items:
            # Find matching indent
            for indent_id, indent in config.USER_INDENTS.items():
                if indent.sku_id == item.sku.sku_id:
                    scheduled_quantities[indent_id] += item.quantity
        
        # Identify unfulfilled indents
        for indent_id, indent in config.USER_INDENTS.items():
            if scheduled_quantities[indent_id] < indent.qty_required * 0.95:  # 95% fulfillment threshold
                self.unfulfilled_indents.append(indent)
        
        total_production = sum(item.quantity for item in self.scheduled_items)
        efficiency_score = self._calculate_efficiency_score()
        
        result = SchedulingResult(
            schedule_items=self.scheduled_items,
            unfulfilled_indents=self.unfulfilled_indents,
            total_production=total_production,
            efficiency_score=efficiency_score,
            warnings=self.warnings
        )
        
        logger.info(f"Enhanced scheduling complete. {len(self.scheduled_items)} items scheduled, "
                   f"{len(self.unfulfilled_indents)} unfulfilled, efficiency: {efficiency_score:.2f}%")
        
        return result

    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score"""
        if not self.scheduled_items:
            return 0.0
        
        total_production_time = 0.0
        total_setup_time = 0.0
        total_cip_time = 0.0
        
        for item in self.scheduled_items:
            line = config.LINES[item.line.line_id]
            production_rate = line.compatible_skus_max_production.get(item.sku.sku_id, 0)
            
            if production_rate > 0:
                production_time = item.quantity / production_rate
                total_production_time += production_time
            
            total_setup_time += item.setup_time_minutes
            total_cip_time += item.cip_time_minutes
        
        total_time = total_production_time + total_setup_time + total_cip_time
        
        if total_time == 0:
            return 0.0
        
        return (total_production_time / total_time) * 100

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get detailed optimization metrics"""
        if not self.scheduled_items:
            return {"message": "No items scheduled"}
        
        # Line utilization metrics
        line_utilization = {}
        for line_id, line in config.LINES.items():
            total_time = 0.0
            busy_time = 0.0
            
            for item in self.scheduled_items:
                if item.line.line_id == line_id:
                    duration = (item.end_time - item.start_time).total_seconds() / 60
                    busy_time += duration + item.setup_time_minutes + item.cip_time_minutes
            
            # Calculate against available shift time
            total_shift_time = sum(shift.duration_minutes() for shift in config.SHIFTS.values() if shift.is_active())
            total_time = total_shift_time * self.max_lookahead_days
            
            line_utilization[line_id] = (busy_time / total_time * 100) if total_time > 0 else 0
        
        # Tank utilization metrics
        tank_utilization = {}
        for tank_id, tank in config.TANKS.items():
            max_usage = 0.0
            for item in self.scheduled_items:
                if item.tank.tank_id == tank_id:
                    usage = item.quantity / tank.capacity_liters
                    max_usage = max(max_usage, usage)
            tank_utilization[tank_id] = max_usage * 100
        
        # Flow efficiency
        total_demand = sum(indent.qty_required for indent in config.USER_INDENTS.values())
        total_scheduled = sum(item.quantity for item in self.scheduled_items)
        flow_efficiency = (total_scheduled / total_demand * 100) if total_demand > 0 else 0
        
        return {
            "total_items_scheduled": len(self.scheduled_items),
            "total_production_liters": total_scheduled,
            "total_demand_liters": total_demand,
            "demand_fulfillment_rate": flow_efficiency,
            "overall_efficiency_score": self._calculate_efficiency_score(),
            "line_utilization_percent": line_utilization,
            "tank_utilization_percent": tank_utilization,
            "average_line_utilization": sum(line_utilization.values()) / len(line_utilization) if line_utilization else 0,
            "average_tank_utilization": sum(tank_utilization.values()) / len(tank_utilization) if tank_utilization else 0,
            "total_production_slots_generated": len(self.production_slots),
            "unfulfilled_indents": len(self.unfulfilled_indents),
            "setup_efficiency": self._calculate_setup_efficiency(),
            "schedule_compactness": self._calculate_schedule_compactness()
        }

    def _calculate_setup_efficiency(self) -> float:
        """Calculate setup efficiency (minimize setups)"""
        if not self.scheduled_items:
            return 100.0
        
        total_setups = sum(1 for item in self.scheduled_items if item.setup_time_minutes > 0)
        return max(0, 100 - (total_setups / len(self.scheduled_items) * 100))

    def _calculate_schedule_compactness(self) -> float:
        """Calculate how compact the schedule is (minimize gaps)"""
        if len(self.scheduled_items) < 2:
            return 100.0
        
        # Group by line
        line_schedules = defaultdict(list)
        for item in self.scheduled_items:
            line_schedules[item.line.line_id].append(item)
        
        total_gaps = 0.0
        total_production_time = 0.0
        
        for line_id, items in line_schedules.items():
            items.sort(key=lambda x: x.start_time)
            
            for i in range(len(items) - 1):
                gap = (items[i+1].start_time - items[i].end_time).total_seconds() / 60
                total_gaps += max(0, gap)
            
            for item in items:
                total_production_time += (item.end_time - item.start_time).total_seconds() / 60
        
        if total_production_time == 0:
            return 100.0
        
        compactness = max(0, 100 - (total_gaps / total_production_time * 100))
        return compactness
