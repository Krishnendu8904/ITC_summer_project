"""
Core Scheduler - Main Production Allocation Engine
Implements heuristic-based scheduling with constraint satisfaction
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import config 

from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductType
)
from config import (
    PENALTY_WEIGHTS, SETUP_TIME_SAME_VARIANT_MINUTES, 
    SETUP_TIME_DIFFERENT_VARIANT_MINUTES, DEFAULT_CIP_TIME_MINUTES
)

logger = logging.getLogger(__name__)

@dataclass
class SchedulingContext:
    """Context for scheduling operations"""
    skus: Dict[str, SKU]
    lines: Dict[str, Line]
    tanks: Dict[str, Tank]
    shifts: Dict[str, Shift]
    line_compatibility: Dict[str, List[str]]
    current_time: datetime
    
class ProductionScheduler:
    """Main scheduling engine using heuristic approaches"""
    
    def __init__(self):
        self.context: Optional[SchedulingContext] = None
        self.scheduled_items: List[ScheduleItem] = []
        self.unfulfilled_indents: List[UserIndent] = []
        self.warnings: List[str] = []
        
    def schedule_production(self, data: Dict[str, Any]) -> SchedulingResult:
        """Main scheduling method"""
        logger.info("Starting production scheduling")
        
        # Initialize context
        self.context = self._create_context(data)
        self.scheduled_items = []
        self.unfulfilled_indents = []
        self.warnings = []
        
        # Sort indents by priority and due date
        sorted_indents = self._prioritize_indents(data['user_indents'])
        
        # Schedule each indent
        for indent in sorted_indents:
            success = self._schedule_indent(indent)
            if not success:
                self.unfulfilled_indents.append(indent)
                logger.warning(f"Failed to schedule indent: {indent.sku_id} - {indent.qty_required}L")
        
        # Calculate efficiency metrics
        total_production = sum(item.quantity for item in self.scheduled_items)
        efficiency_score = self._calculate_efficiency_score()
        
        result = SchedulingResult(
            schedule_items=self.scheduled_items,
            unfulfilled_indents=self.unfulfilled_indents,
            total_production=total_production,
            efficiency_score=efficiency_score,
            warnings=self.warnings
        )
        
        logger.info(f"Scheduling complete. {len(self.scheduled_items)} items scheduled, "
                   f"{len(self.unfulfilled_indents)} unfulfilled")
        
        return result
    
    def _create_context(self, data: Dict[str, Any]) -> SchedulingContext:
        """Create scheduling context from loaded data"""
        skus = config.SKUS
        lines = config.LINES
        tanks = config.TANKS
        shifts = config.SHIFTS
        
        return SchedulingContext(
            skus=skus,
            lines=lines,
            tanks=tanks,
            shifts=shifts,
            line_compatibility=data['line_sku_compatibility'],
            current_time=datetime.now()
        )
    
    def _prioritize_indents(self, indents: List[UserIndent]) -> List[UserIndent]:
        """Sort indents by priority and due date"""
        return sorted(indents, key=lambda x: (
            x.priority.value,  # Lower number = higher priority
            x.due_date or datetime.max,  # Earlier due date first
            -x.qty_required  # Larger quantities first (for efficiency)
        ))
    
    def _schedule_indent(self, indent: UserIndent) -> bool:
        """Attempt to schedule a single indent"""
        if indent.sku_id not in self.context.skus:
            self.warnings.append(f"SKU {indent.sku_id} not found in configuration")
            return False
        
        sku = self.context.skus[indent.sku_id]
        
        # Find best line-tank-shift combination
        best_assignment = self._find_best_assignment(indent, sku)
        
        if not best_assignment:
            return False
        
        line_id, tank_id, shift_id, start_time, production_time = best_assignment
        
        # Calculate setup and CIP times
        setup_time = self._calculate_setup_time(line_id, sku)
        cip_time = self._calculate_cip_time(line_id, sku)
        
        # Create schedule item
        end_time = start_time + timedelta(minutes=production_time)
        
        schedule_item = ScheduleItem(
            sku_id=indent.sku_id,
            line_id=line_id,
            tank_id=tank_id,
            shift_id=shift_id,
            start_time=start_time,
            end_time=end_time,
            quantity=indent.qty_required,
            setup_time_minutes=setup_time,
            cip_time_minutes=cip_time
        )
        
        # Update system state
        self._update_system_state(schedule_item)
        self.scheduled_items.append(schedule_item)
        
        logger.info(f"Scheduled {indent.sku_id}: {indent.qty_required}L on {line_id} "
                   f"from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")
        
        return True
    
    def _find_best_assignment(self, indent: UserIndent, sku: SKU) -> Optional[Tuple[str, str, str, datetime, int]]:
        """Find the best line-tank-shift combination for an indent"""
        best_score = float('inf')
        best_assignment = None
        
        # Get compatible lines
        compatible_lines = []
        for line_id, line in self.context.lines.items():
            if (line_id in self.context.line_compatibility and 
                indent.sku_id in self.context.line_compatibility[line_id] and
                line.is_available()):
                compatible_lines.append(line_id)
        
        if not compatible_lines:
            self.warnings.append(f"No compatible lines for SKU {indent.sku_id}")
            return None
        
        # Try each line-tank-shift combination
        for line_id in compatible_lines:
            for tank_id, tank in self.context.tanks.items():
                if not tank.can_store(indent.qty_required, indent.sku_id):
                    continue
                
                for shift_id, shift in self.context.shifts.items():
                    if not shift.is_active():
                        continue
                    
                    # Calculate production time
                    production_rate = min(sku.base_production_rate, 
                                        self.context.lines[line_id].max_capacity)
                    production_time = int((indent.qty_required / production_rate) * 60)  # minutes
                    
                    # Find earliest available start time
                    start_time = self._find_earliest_start_time(line_id, shift, production_time)
                    
                    if not start_time:
                        continue
                    
                    # Calculate assignment score
                    score = self._calculate_assignment_score(
                        indent, sku, line_id, tank_id, shift_id, start_time, production_time
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_assignment = (line_id, tank_id, shift_id, start_time, production_time)
        
        return best_assignment
    
    def _find_earliest_start_time(self, line_id: str, shift: Shift, required_minutes: int) -> Optional[datetime]:
        """Find earliest available start time for a line in a shift"""
        line = self.context.lines[line_id]
        
        # Check if line has enough time in shift
        if shift.duration_minutes() < required_minutes:
            return None
        
        # Start with shift start time
        candidate_start = shift.start_time
        
        # Check for conflicts with existing scheduled items
        for scheduled in self.scheduled_items:
            if scheduled.line_id == line_id:
                # If there's overlap, move start time after this item
                if (candidate_start < scheduled.end_time and 
                    candidate_start + timedelta(minutes=required_minutes) > scheduled.start_time):
                    candidate_start = scheduled.end_time
        
        # Check if we still fit in the shift
        if candidate_start + timedelta(minutes=required_minutes) > shift.end_time:
            return None
        
        return candidate_start
    
    def _calculate_setup_time(self, line_id: str, sku: SKU) -> int:
        """Calculate setup time for line changeover"""
        line = self.context.lines[line_id]
        
        if not line.current_product:
            return 0  # No setup needed for empty line
        
        if line.current_product == sku.sku_id:
            return SETUP_TIME_SAME_VARIANT_MINUTES
        
        # Check if it's the same product type
        current_sku = self.context.skus.get(line.current_product)
        if current_sku and current_sku.product_type == sku.product_type:
            return SETUP_TIME_SAME_VARIANT_MINUTES
        
        return SETUP_TIME_DIFFERENT_VARIANT_MINUTES
    
    def _calculate_cip_time(self, line_id: str, sku: SKU) -> int:
        """Calculate CIP time if needed"""
        line = self.context.lines[line_id]
        
        if line.needs_cip(sku):
            return DEFAULT_CIP_TIME_MINUTES
        
        return 0
    
    def _calculate_assignment_score(self, indent: UserIndent, sku: SKU, line_id: str, 
                                  tank_id: str, shift_id: str, start_time: datetime, 
                                  production_time: int) -> float:
        """Calculate score for a potential assignment (lower is better)"""
        score = 0.0
        
        # Priority penalty (higher priority = lower penalty)
        score += indent.priority.value * PENALTY_WEIGHTS['unfulfilled_demand'] * 0.1
        
        # Due date penalty
        if indent.due_date:
            days_until_due = (indent.due_date - start_time).days
            if days_until_due < 0:
                score += abs(days_until_due) * PENALTY_WEIGHTS['unfulfilled_demand'] * 0.5
        
        # Setup cost
        setup_time = self._calculate_setup_time(line_id, sku)
        score += setup_time * PENALTY_WEIGHTS['line_setup_cost']
        
        # CIP cost
        cip_time = self._calculate_cip_time(line_id, sku)
        score += cip_time * PENALTY_WEIGHTS['tank_cip_cost']
        
        # Efficiency bonus for good line utilization
        line_capacity = self.context.lines[line_id].max_capacity
        utilization = min(sku.base_production_rate / line_capacity, 1.0)
        score += (1.0 - utilization) * PENALTY_WEIGHTS['efficiency_bonus']
        
        # Tank capacity utilization
        tank = self.context.tanks[tank_id]
        tank_utilization = indent.qty_required / tank.capacity_liters
        if tank_utilization > 0.8:  # Bonus for efficient tank usage
            score += PENALTY_WEIGHTS['efficiency_bonus'] * 0.5
        
        return score
    
    def _update_system_state(self, schedule_item: ScheduleItem):
        """Update line and tank states after scheduling"""
        # Update line state
        line = self.context.lines[schedule_item.line_id]
        line.current_product = schedule_item.sku_id
        line.status = LineStatus.ACTIVE
        
        # Update tank state
        tank = self.context.tanks[schedule_item.tank_id]
        tank.current_product = schedule_item.sku_id
        tank.current_volume += schedule_item.quantity
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall scheduling efficiency score"""
        if not self.scheduled_items:
            return 0.0
        
        total_production_time = sum(item.duration_minutes() for item in self.scheduled_items)
        total_setup_time = sum(item.setup_time_minutes for item in self.scheduled_items)
        total_cip_time = sum(item.cip_time_minutes for item in self.scheduled_items)
        
        if total_production_time + total_setup_time + total_cip_time == 0:
            return 0.0
        
        # Efficiency = production time / (production + setup + cip time)
        efficiency = total_production_time / (total_production_time + total_setup_time + total_cip_time)
        
        return efficiency * 100  # Return as percentage
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the schedule"""
        if not self.scheduled_items:
            return {"message": "No items scheduled"}
        
        summary = {
            "total_items": len(self.scheduled_items),
            "total_production": sum(item.quantity for item in self.scheduled_items),
            "total_duration_hours": sum(item.duration_minutes() for item in self.scheduled_items) / 60,
            "average_setup_time": sum(item.setup_time_minutes for item in self.scheduled_items) / len(self.scheduled_items),
            "efficiency_score": self._calculate_efficiency_score(),
            "unfulfilled_count": len(self.unfulfilled_indents),
            "success_rate": len(self.scheduled_items) / (len(self.scheduled_items) + len(self.unfulfilled_indents)) * 100
        }
        
        return summary