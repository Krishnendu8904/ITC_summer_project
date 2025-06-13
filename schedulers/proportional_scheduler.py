import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

import config
from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductTypeRegistry
)

logger = logging.getLogger(__name__)


class ProportionalProductionScheduler:
    """
    Production Scheduler that allocates extra production time proportional to indent quantities.
    This scheduler produces additional quantities beyond the required amount based on the 
    proportion of each indent relative to total demand.
    """
    
    def __init__(self, extra_production_factor: float = 0.2):
        """
        Initialize scheduler with proportional extra production.
        
        Args:
            extra_production_factor: Factor to determine extra production (e.g., 0.2 = 20% extra)
        """
        self.scheduled_items: List[ScheduleItem] = []
        self.unfulfilled_indents: List[UserIndent] = []
        self.warnings: List[str] = []
        self.run_start_time: datetime = datetime.now()
        self.extra_production_factor = extra_production_factor

    def schedule_production(self) -> SchedulingResult:
        """Main scheduling method with proportional extra production allocation."""
        logger.info("Starting proportional production scheduling")

        # Reset internal state
        self.scheduled_items = []
        self.unfulfilled_indents = []
        self.warnings = []
        self.run_start_time = datetime.now()

        # Calculate total demand for proportional allocation
        total_demand = sum(indent.qty_required for indent in config.USER_INDENTS.values())
        
        if total_demand == 0:
            logger.warning("No indents found or total demand is zero")
            return SchedulingResult([], [], 0, 0.0, ["No indents to schedule"])

        # Calculate proportional extra production for each indent
        enhanced_indents = self._calculate_proportional_quantities(config.USER_INDENTS, total_demand)
        
        # Sort indents by priority and due date
        sorted_indents = self._prioritize_indents(enhanced_indents)

        # Schedule each indent with enhanced quantities
        for indent_data in sorted_indents:
            indent, enhanced_quantity = indent_data
            success = self._schedule_indent_with_extra(indent, enhanced_quantity)
            if not success:
                self.unfulfilled_indents.append(indent)
                logger.warning(f"Failed to schedule indent: {indent.sku_id} - {enhanced_quantity}L (original: {indent.qty_required}L)")

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

        logger.info(f"Proportional scheduling complete. {len(self.scheduled_items)} items scheduled, "
                   f"{len(self.unfulfilled_indents)} unfulfilled. Extra production factor: {self.extra_production_factor}")

        return result

    def _calculate_proportional_quantities(self, indents: Dict[str, UserIndent], total_demand: float) -> List[Tuple[UserIndent, float]]:
        """
        Calculate enhanced quantities with proportional extra production.
        
        Returns list of tuples: (indent, enhanced_quantity)
        """
        enhanced_indents = []
        
        for indent in indents.values():
            # Calculate proportion of this indent relative to total demand
            proportion = indent.qty_required / total_demand
            
            # Calculate extra production for this indent based on its proportion
            extra_quantity = (total_demand * self.extra_production_factor) * proportion
            
            # Enhanced quantity = original + proportional extra
            enhanced_quantity = indent.qty_required + extra_quantity
            
            enhanced_indents.append((indent, enhanced_quantity))
            
            logger.info(f"SKU {indent.sku_id}: Original={indent.qty_required}L, "
                       f"Proportion={proportion:.3f}, Extra={extra_quantity:.1f}L, "
                       f"Enhanced={enhanced_quantity:.1f}L")
        
        return enhanced_indents

    def _prioritize_indents(self, enhanced_indents: List[Tuple[UserIndent, float]]) -> List[Tuple[UserIndent, float]]:
        """Sort enhanced indents by priority, due date, and enhanced quantity."""
        return sorted(enhanced_indents, key=lambda x: (
            x[0].priority.value if x[0].priority else Priority.MEDIUM.value,
            x[0].due_date or (datetime.today() + timedelta(days=3)),
            -x[1]  # Sort by enhanced quantity (higher first)
        ))

    def _schedule_indent_with_extra(self, indent: UserIndent, enhanced_quantity: float) -> bool:
        """Schedule an indent with enhanced (proportional extra) quantity."""
        if indent.sku_id not in config.SKUS:
            self.warnings.append(f"SKU {indent.sku_id} from indent not found in configuration.")
            return False

        sku = config.SKUS[indent.sku_id]

        # Find best assignment using enhanced quantity
        best_assignment = self._find_best_assignment_for_quantity(indent, sku, enhanced_quantity)

        if not best_assignment:
            return False

        line_id, tank_id, shift_id, start_time, production_time = best_assignment

        # Calculate setup and CIP times
        setup_time = self._calculate_setup_time(line_id, sku)
        cip_time = self._calculate_cip_time(line_id, sku)

        # Create schedule item with enhanced quantity
        end_time = start_time + timedelta(minutes=production_time)

        schedule_item = ScheduleItem(
            sku=config.SKUS[indent.sku_id],
            line=config.LINES[line_id],
            tank=config.TANKS[tank_id],
            shift=config.SHIFTS[shift_id],
            start_time=start_time,
            end_time=end_time,
            quantity=enhanced_quantity,  # Use enhanced quantity
            produced_quantity=enhanced_quantity,  # Assume full production
            setup_time_minutes=setup_time,
            cip_time_minutes=cip_time
        )

        # Update system state
        self._update_system_state(schedule_item)
        self.scheduled_items.append(schedule_item)

        logger.info(f"Scheduled {indent.sku_id}: {enhanced_quantity:.1f}L (original: {indent.qty_required}L) "
                   f"on {line_id} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")

        return True

    def _find_best_assignment_for_quantity(self, indent: UserIndent, sku: SKU, quantity: float) -> Optional[Tuple[str, str, str, datetime, int]]:
        """Find best assignment for a specific quantity (enhanced)."""
        best_score = float('inf')
        best_assignment = None

        # Find compatible lines
        compatible_lines = []
        for line_id, line in config.LINES.items():
            if (line.is_available() and
                sku.sku_id in line.compatible_skus_max_production and
                line.compatible_skus_max_production[sku.sku_id] > 0):
                compatible_lines.append(line_id)

        if not compatible_lines:
            self.warnings.append(f"No compatible lines found for SKU {indent.sku_id}")
            return None

        # Try each combination
        for line_id in compatible_lines:
            line = config.LINES[line_id]
            for tank_id, tank in config.TANKS.items():
                
                # Check if tank can store the enhanced quantity
                if not tank.can_store(quantity, sku):
                    continue

                for shift_id, shift in config.SHIFTS.items():
                    if not shift.is_active():
                        continue

                    # Calculate production time for enhanced quantity
                    production_rate = line.compatible_skus_max_production.get(sku.sku_id)
                    if production_rate is None or production_rate <= 0:
                        continue
                    
                    production_time = int(quantity / production_rate)

                    # Find earliest start time
                    start_time = self._find_earliest_start_time(line, shift, production_time)
                    if not start_time:
                        continue

                    # Calculate assignment score for enhanced quantity
                    score = self._calculate_assignment_score_for_quantity(
                        indent, sku, line_id, tank_id, shift_id, start_time, production_time, quantity
                    )

                    if score < best_score:
                        best_score = score
                        best_assignment = (line_id, tank_id, shift_id, start_time, production_time)

        return best_assignment

    def _calculate_assignment_score_for_quantity(self, indent: UserIndent, sku: SKU, line_id: str,
                                               tank_id: str, shift_id: str, start_time: datetime,
                                               production_time: int, quantity: float) -> float:
        """Calculate assignment score for enhanced quantity."""
        score = 0.0

        # Priority penalty
        score += indent.priority.value * config.PENALTY_WEIGHTS.get('priority_cost', 0.1)

        # Due date penalty (based on original due date)
        if indent.due_date:
            time_to_due = (indent.due_date - (start_time + timedelta(minutes=production_time)))
            if time_to_due < timedelta(minutes=0):
                score += abs(time_to_due.total_seconds() / 3600) * config.PENALTY_WEIGHTS.get('lateness_cost', 10.0)
            elif time_to_due < timedelta(days=1):
                score -= abs(time_to_due.total_seconds() / 3600) * config.PENALTY_WEIGHTS.get('due_date_proximity_cost', 5.0)

        # Setup and CIP costs
        setup_time = self._calculate_setup_time(line_id, sku)
        cip_time = self._calculate_cip_time(line_id, sku)
        score += setup_time * config.PENALTY_WEIGHTS.get('line_setup_cost', 0.5)
        score += cip_time * config.PENALTY_WEIGHTS.get('tank_cip_cost', 1.0)

        # Efficiency calculation for enhanced quantity
        line = config.LINES[line_id]
        actual_production_rate = line.compatible_skus_max_production.get(sku.sku_id, 0.0)
        
        if actual_production_rate > 0:
            cost_per_liter_minute = (production_time + setup_time + cip_time) / quantity
            score += cost_per_liter_minute * config.PENALTY_WEIGHTS.get('inefficiency_cost', 0.01)

        # Tank utilization for enhanced quantity
        tank = config.TANKS[tank_id]
        if tank.capacity_liters > 0:
            tank_utilization = quantity / tank.capacity_liters
            if tank_utilization < config.PENALTY_WEIGHTS.get('min_tank_fill_percentage', 0.2):
                score += config.PENALTY_WEIGHTS.get('tank_underutilization_cost', 5.0)
            elif tank_utilization > config.PENALTY_WEIGHTS.get('max_tank_fill_percentage', 0.8):
                score -= config.PENALTY_WEIGHTS.get('tank_high_utilization_bonus', 2.0)

        # Bonus for producing extra (incentivize extra production)
        extra_production_bonus = (quantity - indent.qty_required) * 0.1
        score -= extra_production_bonus

        return score

    # Reuse helper methods from original scheduler
    def _find_earliest_start_time(self, line: Line, shift: Shift, required_minutes: int) -> Optional[datetime]:
        """Find earliest available start time for a line in a shift."""
        if shift.duration_minutes() < required_minutes:
            return None
        
        candidate_start = datetime.combine(self.run_start_time.date(), shift.start_time.time())
        candidate_end = datetime.combine(self.run_start_time.date(), shift.end_time.time())
        if candidate_end < candidate_start:
            candidate_end += timedelta(1)

        if candidate_start < self.run_start_time:
            candidate_start = candidate_start + timedelta(1)
            candidate_end = candidate_end + timedelta(1)

        # Check for conflicts with existing scheduled items
        for scheduled in self.scheduled_items:
            if scheduled.line.line_id == line.line_id:
                if (candidate_start < scheduled.end_time and
                    candidate_start + timedelta(minutes=required_minutes) > scheduled.start_time):
                    candidate_start = scheduled.end_time

        if candidate_start + timedelta(minutes=required_minutes) > candidate_end:
            return None

        return candidate_start

    def _calculate_setup_time(self, line_id: str, sku: SKU) -> int:
        """Calculate setup time for line changeover."""
        line = config.LINES[line_id]
        if not line.current_sku:
            return sku.setup_time  
        if line.current_sku == sku.sku_id:
            return 0
        current_sku_on_line = config.SKUS.get(line.current_sku.sku_id)
        if current_sku_on_line is None or current_sku_on_line.product_category != sku.product_category:
            return 180
        return sku.setup_time

    def _calculate_cip_time(self, line_id: str, sku: SKU) -> int:
        """Calculate CIP time if needed."""
        line = config.LINES[line_id]
        if line.needs_cip(sku):
            return 180
        return 0

    def _update_system_state(self, schedule_item: ScheduleItem):
        """Update line and tank states after scheduling."""
        line = config.LINES[schedule_item.line.line_id]
        tank = config.TANKS[schedule_item.tank.tank_id]
        
        line.current_sku = schedule_item.sku
        line.status = LineStatus.ACTIVE
        tank.current_product = schedule_item.sku.product_category
        tank.current_volume += schedule_item.quantity

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall scheduling efficiency score."""
        if not self.scheduled_items:
            return 0.0

        total_production_time = sum(
            item.quantity / config.LINES[item.line.line_id].compatible_skus_max_production[item.sku.sku_id]
            for item in self.scheduled_items 
            if config.LINES[item.line.line_id].compatible_skus_max_production.get(item.sku.sku_id, 0) > 0
        )
        total_setup_time = sum(item.setup_time_minutes for item in self.scheduled_items)
        total_cip_time = sum(item.cip_time_minutes for item in self.scheduled_items)
        total_scheduled_time = total_production_time + total_setup_time + total_cip_time

        if total_scheduled_time == 0:
            return 0.0

        return (total_production_time / total_scheduled_time) * 100

    def get_proportional_summary(self) -> Dict[str, Any]:
        """Get summary with proportional production details."""
        if not self.scheduled_items:
            return {"message": "No items scheduled"}

        original_demand = sum(indent.qty_required for indent in config.USER_INDENTS.values())
        total_scheduled = sum(item.quantity for item in self.scheduled_items)
        extra_production = total_scheduled - original_demand

        summary = {
            "total_items_scheduled": len(self.scheduled_items),
            "original_demand_liters": original_demand,
            "total_scheduled_liters": total_scheduled,
            "extra_production_liters": extra_production,
            "extra_production_percentage": (extra_production / original_demand * 100) if original_demand > 0 else 0,
            "proportional_factor_used": self.extra_production_factor,
            "total_scheduled_duration_hours": sum(item.duration_minutes() for item in self.scheduled_items) / 60,
            "efficiency_score": self._calculate_efficiency_score(),
            "unfulfilled_indent_count": len(self.unfulfilled_indents),
            "fulfillment_rate_percentage": (total_scheduled / original_demand * 100) if original_demand > 0 else 0
        }

        return summary