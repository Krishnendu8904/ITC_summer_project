import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

import config # Import the entire config module to access its populated attributes

from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductTypeRegistry
)

logger = logging.getLogger(__name__)


class ProductionScheduler:
    """Main Scheduling and Allocation Engine using heuristic approaches."""
    def __init__(self):
        # These are instance variables for the current scheduling run's results
        self.scheduled_items: List[ScheduleItem] = []
        self.unfulfilled_indents: List[UserIndent] = []
        self.warnings: List[str] = []
        # Store the current time for this scheduling run's context
        self.run_start_time: datetime = datetime.now()

    def schedule_production(self) -> SchedulingResult:
        """Main scheduling method.
        This method assumes `config` module has been populated by DataLoader.
        """
        logger.info("Starting production scheduling")

        # Reset internal state for a new scheduling run
        self.scheduled_items = []
        self.unfulfilled_indents = []
        self.warnings = []
        self.run_start_time = datetime.now() # Update start time for this specific run

        # Sort indents by priority and due date.
        # Ensure config.USER_INDENTS is a Dict[str, UserIndent] as expected by _prioritize_indents.
        sorted_indents = self._prioritize_indents(config.USER_INDENTS)

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
                    f"{len(self.unfulfilled_indents)} unfulfilled.")

        return result
    
    def _prioritize_indents(self, indents: Dict[str, UserIndent]) -> List[UserIndent]:
        """
        Sort user indents by:
        1. Priority (HIGH=1, LOW=3)
        2. Due date (earlier is better)
        3. Quantity required (higher is better)
        """
        return sorted(indents.values(), key=lambda indent: (
            indent.priority.value if indent.priority else Priority.MEDIUM.value,
            indent.due_date or (datetime.today() +timedelta(days=3)), # Use datetime.max if due_date is None
            -indent.qty_required if indent.qty_required is not None else 0
        ))
    
    def _schedule_indent(self, indent: UserIndent) -> bool:
        """Attempt to schedule a single indent."""
        if indent.sku_id not in config.SKUS:
            self.warnings.append(f"SKU {indent.sku_id} from indent not found in configuration.")
            return False

        sku = config.SKUS[indent.sku_id]

        # Find best line-tank-shift combination
        best_assignment = self._find_best_assignment(indent, sku)
        print(best_assignment)

        if not best_assignment:
            return False

        line_id, tank_id, shift_id, start_time, production_time = best_assignment

        # Calculate setup and CIP times
        setup_time = self._calculate_setup_time(line_id, sku)
        cip_time = self._calculate_cip_time(line_id, sku)

        # Create schedule item
        end_time = start_time + timedelta(minutes=production_time)

        schedule_item = ScheduleItem(
            sku=config.SKUS[indent.sku_id],
            line=config.LINES[line_id],
            tank=config.TANKS[tank_id],
            shift=config.SHIFTS[shift_id],
            start_time=start_time,
            end_time=end_time,
            quantity=indent.qty_required,
            produced_quantity = 100, 
            setup_time_minutes=setup_time,
            cip_time_minutes=cip_time
        )

        # Update system state (i.e., the state of the line and tank objects in config)
        self._update_system_state(schedule_item)
        self.scheduled_items.append(schedule_item)

        logger.info(f"Scheduled {indent.sku_id}: {indent.qty_required}L on {line_id} "
                   f"from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")

        return True
    
    def _find_best_assignment(self, indent: UserIndent, sku: SKU) -> Optional[Tuple[str, str, str, datetime, int]]:
        """Find the best line-tank-shift combination for an indent."""
        best_score = float('inf')
        best_assignment = None

        compatible_lines = []
        # Iterate directly through config.LINES
        for line_id, line in config.LINES.items():
            # Check line availability and specific SKU compatibility and its production rate
            if (line.is_available() and
                sku.sku_id in line.compatible_skus_max_production and
                line.compatible_skus_max_production[sku.sku_id] > 0): # Ensure a non-zero production rate
                compatible_lines.append(line_id)

        if not compatible_lines:
            self.warnings.append(f"No compatible lines found for SKU {indent.sku_id} that are available or have a valid production rate.")
            return None

        # Try each line-tank-shift combination
        for line_id in compatible_lines:
            line = config.LINES[line_id] # Get the line object once
            # Iterate directly through config.TANKS
            for tank_id, tank in config.TANKS.items():
                # Pass the SKU object to tank.can_store
                if not tank.can_store(indent.qty_required, sku):
                    continue

                # Iterate directly through config.SHIFTS
                for shift_id, shift in config.SHIFTS.items():
                    if not shift.is_active():
                        continue

                    # Calculate production time using the specific production rate for this SKU on this line
                    production_rate = line.compatible_skus_max_production.get(sku.sku_id)
                    if production_rate is None or production_rate <= 0: # Should be caught by compatible_lines filter, but defensive check
                        continue
                    
                    production_time = int((indent.qty_required / production_rate)) # minutes
                    print(production_rate)

                    # Find earliest available start time for this line within this shift
                    start_time = self._find_earliest_start_time(line, shift, production_time)

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
    
    def _find_earliest_start_time(self, line: Line, shift: Shift, required_minutes: int) -> Optional[datetime]:
        """Find earliest available start time for a line in a shift."""

        # Check if line has enough time in shift
        # Consider total available time in shift (shift duration - any CIP/setup needed at shift start)
        # For simplicity, let's just use shift.duration_minutes() for now, as setup/CIP are calculated separately
        if shift.duration_minutes() < required_minutes:
            return None

        # Start with shift start time (adjusted to run_start_time's date if shift is defined relative)
        # Assuming shift.start_time is relative to current date, combine it
        candidate_start = datetime.combine(self.run_start_time.date(), shift.start_time.time())

        # If the candidate start time is in the past, move it to now (if scheduling begins from now)
        if candidate_start < self.run_start_time:
            candidate_start = self.run_start_time

        # Check for conflicts with existing scheduled items on this line
        # This is a crucial part of sequential scheduling
        for scheduled in self.scheduled_items:
            if scheduled.line.line_id == line.line_id:
                # If there's overlap, move start time after this item's end time
                # Consider if new item would finish before current item starts or starts after current finishes
                # This logic ensures no overlap on the same line
                if (candidate_start < scheduled.end_time and
                    candidate_start + timedelta(minutes=required_minutes) > scheduled.start_time):
                    candidate_start = scheduled.end_time # Move start to after previous task ends

        # Check if we still fit in the shift's end time
        # Ensure shift.end_time is also combined with the run_start_time's date
        shift_end_datetime = datetime.combine(self.run_start_time.date(), shift.end_time.time())

        if candidate_start + timedelta(minutes=required_minutes) > shift_end_datetime:
            return None

        return candidate_start
    
    def _calculate_setup_time(self, line_id: str, sku: SKU) -> int:
        """Calculate setup time for line changeover."""
        line = config.LINES[line_id] # Access line from config

        if not line.current_sku:
            return 0  # No setup needed for an idle line

        # If the line is already producing this exact SKU, no setup needed (or minimal for variant change)
        if line.current_sku == sku.sku_id:
            return config.SETUP_TIME_SAME_VARIANT_MINUTES # Using a general "same variant" setup

        # Get the SKU object of the current product on the line
        current_sku_on_line = config.SKUS.get(line.current_sku)

        # If current product is unknown or different product type, consider full setup
        if current_sku_on_line is None or current_sku_on_line.product_category != sku.product_category:
            return config.SETUP_TIME_DIFFERENT_VARIANT_MINUTES
        
        # If same product category but different SKU (variant)
        return config.SETUP_TIME_SAME_VARIANT_MINUTES


    def _calculate_cip_time(self, line_id: str, sku: SKU) -> int:
        """Calculate CIP time if needed for a line changeover."""
        line = config.LINES[line_id] # Access line from config

        if line.needs_cip(sku): # Use the Line's method to check CIP needs
            return config.DEFAULT_CIP_TIME_MINUTES

        return 0


    def _calculate_assignment_score(self, indent: UserIndent, sku: SKU, line_id: str,
                                   tank_id: str, shift_id: str, start_time: datetime,
                                   production_time: int) -> float:
        """Calculate score for a potential assignment (lower is better)."""
        score = 0.0

        # Prioritize indents already handled by sorting, so penalty here is more about
        # how "bad" this particular assignment is for the system.
        # Priority penalty (higher priority = lower resulting penalty if scheduled)
        # Using a direct factor of priority value, higher value (lower priority) gives higher penalty
        score += indent.priority.value * config.PENALTY_WEIGHTS.get('priority_cost', 0.1) # Ensure key exists

        # Due date penalty (penalize if past due or very close to due date)
        if indent.due_date:
            time_to_due = (indent.due_date - (start_time + timedelta(minutes=production_time)))
            if time_to_due < timedelta(minutes=0): # Past due
                score += abs(time_to_due.total_seconds() / 3600) * config.PENALTY_WEIGHTS.get('lateness_cost', 10.0) # Penalty per hour late
            elif time_to_due < timedelta(days=1): # Within 24 hours of due date
                score += config.PENALTY_WEIGHTS.get('due_date_proximity_cost', 1.0) # Smaller penalty for being close

        # Setup cost
        setup_time = self._calculate_setup_time(line_id, sku)
        score += setup_time * config.PENALTY_WEIGHTS.get('line_setup_cost', 0.5)

        # CIP cost
        cip_time = self._calculate_cip_time(line_id, sku)
        score += cip_time * config.PENALTY_WEIGHTS.get('tank_cip_cost', 1.0)

        # Efficiency bonus for good line utilization
        line = config.LINES[line_id] # Access line from config
        # Use the specific production rate for this SKU on this line
        actual_production_rate = line.compatible_skus_max_production.get(sku.sku_id, 0.0)
        
        # If for some reason actual_production_rate is 0, avoid division by zero or negative utilization
        if actual_production_rate > 0:
            # We want to minimize (1 - utilization), so a higher utilization gives a lower score
            # Utilization should be relative to the line's inherent capability for this SKU
            # Not just sku.base_production_rate / line.max_capacity, but how much of the *possible*
            # production rate for this SKU on this line is being used by the indent's quantity.
            # Assuming production_rate is the actual production rate (L/min)
            # A simpler approach: penalize for longer total time on line relative to quantity.
            cost_per_liter_minute = (production_time + setup_time + cip_time) / indent.qty_required
            score += cost_per_liter_minute * config.PENALTY_WEIGHTS.get('inefficiency_cost', 0.01)

        # Tank capacity utilization
        tank = config.TANKS[tank_id] # Access tank from config
        if tank.capacity_liters > 0: # Avoid division by zero
            tank_utilization = indent.qty_required / tank.capacity_liters
            # Penalize if tank is used very inefficiently (e.g., small quantity in huge tank)
            # Or give bonus for filling it up well. Let's penalize under-utilization for score minimization
            if tank_utilization < config.PENALTY_WEIGHTS.get('min_tank_fill_percentage', 0.2):
                score += config.PENALTY_WEIGHTS.get('tank_underutilization_cost', 5.0)
            elif tank_utilization > config.PENALTY_WEIGHTS.get('max_tank_fill_percentage', 0.8): # Bonus for efficient usage near full
                score -= config.PENALTY_WEIGHTS.get('tank_high_utilization_bonus', 2.0) # Subtract from score

        return score


    def _update_system_state(self, schedule_item: ScheduleItem):
        """Update line and tank states in the config module after scheduling."""
        line = config.LINES[schedule_item.line.line_id]
        tank = config.TANKS[schedule_item.tank.tank_id]

        # Update line state
        line.current_sku = schedule_item.sku
        line.status = LineStatus.ACTIVE # Mark line as active/busy

        # Update tank state
        tank.current_product = schedule_item.sku.product_category # Assuming tank holds only one product type at a time
        tank.current_volume += schedule_item.quantity
        # Potentially update tank status here (e.g., TankStatus.FULL) if needed


    def _calculate_efficiency_score(self) -> float:
        """Calculate overall scheduling efficiency score."""
        if not self.scheduled_items:
            return 0.0

        total_production_time = sum(item.quantity / config.LINES[item.line.line_id].compatible_skus_max_production[item.sku.sku_id]
                                   for item in self.scheduled_items if config.LINES[item.line.line_id].compatible_skus_max_production.get(item.sku.sku_id, 0) > 0)
        total_setup_time = sum(item.setup_time_minutes for item in self.scheduled_items)
        total_cip_time = sum(item.cip_time_minutes for item in self.scheduled_items)

        total_scheduled_time = total_production_time + total_setup_time + total_cip_time

        if total_scheduled_time == 0:
            return 0.0

        efficiency = total_production_time / total_scheduled_time
        return efficiency * 100  # Return as percentage


    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the schedule."""
        if not self.scheduled_items:
            return {"message": "No items scheduled"}

        summary = {
            "total_items_scheduled": len(self.scheduled_items),
            "total_production_liters": sum(item.quantity for item in self.scheduled_items),
            "total_scheduled_duration_hours": sum(item.duration_minutes() for item in self.scheduled_items) / 60,
            "total_setup_time_hours": sum(item.setup_time_minutes for item in self.scheduled_items) / 60,
            "total_cip_time_hours": sum(item.cip_time_minutes for item in self.scheduled_items) / 60,
            "efficiency_score": self._calculate_efficiency_score(),
            "unfulfilled_indent_count": len(self.unfulfilled_indents),
            "total_demand_liters": sum(indent.qty_required for indent in config.USER_INDENTS.values()),
            "fulfillment_rate_percentage": (sum(item.quantity for item in self.scheduled_items) / sum(indent.qty_required for indent in config.USER_INDENTS.values()) * 100) if sum(indent.qty_required for indent in config.USER_INDENTS.values()) > 0 else 0
        }

        return summary
    