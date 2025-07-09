import math
import json
import logging
from typing import Dict, Any, List, Tuple
from utils.data_models import *
from utils.data_loader import DataLoader
import config

OEE_FACTOR = 0.8

# --- 1. Logging Configuration ---
# This sets up a logger to write detailed calculation steps to a file,
# making the process transparent and easy to debug.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='capacity_analysis.log',
    filemode='w'  # 'w' overwrites the file each time, 'a' appends
)
# Also, add a handler to print logs to the console.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
# Avoid adding duplicate handlers
if not logging.getLogger('').hasHandlers():
    logging.getLogger('').addHandler(console_handler)


# --- 2. Project Imports ---
# This script is designed to be a module within your existing project.

class CapacityAnalyzer:
    """
    Performs capacity mapping and feasibility checks for a dairy plant.
    Implements the final, agreed-upon methodology for maximum accuracy.
    """

    def __init__(self, products: Dict[str, Product], equipment: Dict[str, Equipment],
                 lines: Dict[str, Line], tanks: Dict[str, Tank],
                 skus: Dict[str, SKU], rooms: Dict[str, Room]):
        """
        Initializes the analyzer with the pre-loaded data from your DataLoader.
        """
        self.products = products
        self.skus = skus
        self.rooms = rooms
        self.lines = lines
        self.equipment = equipment
        self.tanks = tanks

        self.time_based_resources = {**equipment, **lines, **tanks}
        self.all_resources = {**self.time_based_resources, **rooms}
        
        self.DAILY_BUDGET_MINUTES = 1300
        self.DEFAULT_BATCH_SIZE_LITERS = 8000
        logging.info("CapacityAnalyzer initialized with plant configuration.")

    # --- MAIN PUBLIC METHODS ---

    def map_maximum_capacity(self, sku_ratio: Dict[str, float]) -> Dict[str, Any]:
            """
            Calculates the maximum theoretical plant capacity for a given SKU mix ratio.
            This version correctly allocates time for shared resources.
            """
            logging.info(f"--- Starting `map_maximum_capacity` for SKU ratio: {sku_ratio} ---")
            if not math.isclose(sum(sku_ratio.values()), 1.0):
                logging.error("SKU ratios do not sum to 1.0.")
                return {"error": "SKU ratios must sum to 1.0"}

            categorized_skus = self._group_skus_by_category(sku_ratio)
            
            # --- NEW: Pre-computation step to map resource sharing ---
            # This map will store which categories use a resource and their production ratios.
            # Format: {resource_id: {category_name: category_ratio, ...}}
            resource_to_category_ratios = {}
            for category, data in categorized_skus.items():
                product_def = self.products.get(category)
                if not product_def: continue
                
                category_ratio = data.get('total_ratio', 0)
                if category_ratio == 0: continue

                for step in product_def.processing_steps:
                    # We only consider time-based resources for this calculation
                    if step.process_type == ProcessType.POST_PACKAGING: continue
                    if step.requirements and step.requirements[0].compatible_ids:
                        for resource_id in step.requirements[0].compatible_ids:
                            if resource_id not in resource_to_category_ratios:
                                resource_to_category_ratios[resource_id] = {}
                            resource_to_category_ratios[resource_id][category] = category_ratio
            
            logging.info("Built map of shared resource usage.")
            # --- End of New Step ---

            stage_capacities_kg = {}
            logging.info("--- Calculating Capacities for Each Stage ---")
            for category, data in categorized_skus.items():
                product_def = self.products.get(category)
                if not product_def: continue

                logging.info(f"Analyzing Product Category: {category}")
                
                flow_map = self._build_flow_map(product_def, data['skus_in_category'])

                for step in product_def.processing_steps:
                    engagement_time, batch_size_used = self._get_engagement_time(step, product_def, flow_map)
                    
                    if engagement_time > 0:
                        num_parallel_resources = 1
                        if step.requirements and step.requirements[0].compatible_ids:
                            compatible_ids = step.requirements[0].compatible_ids
                            num_parallel_resources = len(compatible_ids)
                            
                            # --- MODIFIED: Adjust time budget for shared resources ---
                            adjusted_daily_budget = self.DAILY_BUDGET_MINUTES
                            # Use the first resource to check for sharing, assuming all parallel resources are shared the same way
                            first_res_id = compatible_ids[0] 
                            
                            if first_res_id in resource_to_category_ratios:
                                sharing_info = resource_to_category_ratios[first_res_id]
                                if len(sharing_info) > 1: # Resource is shared
                                    total_ratio_on_resource = sum(sharing_info.values())
                                    current_category_ratio = sharing_info.get(category, 0)
                                    
                                    if total_ratio_on_resource > 0:
                                        time_allocation_factor = current_category_ratio / total_ratio_on_resource
                                        adjusted_daily_budget *= time_allocation_factor
                                        logging.info(f"  - Resource '{first_res_id}' is shared. Time for '{category}' adjusted by factor {time_allocation_factor:.2f}")

                            capacity_per_resource = (adjusted_daily_budget / engagement_time) * batch_size_used
                            capacity_kg = capacity_per_resource * num_parallel_resources
                            
                            logging.info(f"  - Stage '{step.step_id}': Engagement Time={engagement_time:.2f} mins/res, Budget={adjusted_daily_budget:.2f} mins -> Max Capacity={capacity_kg:.2f} kg/day")
                            
                            if step.step_id in stage_capacities_kg:
                                stage_capacities_kg[step.step_id] = min(stage_capacities_kg[step.step_id], capacity_kg)
                            else:
                                stage_capacities_kg[step.step_id] = capacity_kg

            logging.info("--- Solving System Constraints ---")
            max_total_kg, bottleneck_stage = self._solve_capacity_constraints(stage_capacities_kg, categorized_skus)
            logging.info(f"System Bottleneck identified: '{bottleneck_stage}' with an implied max total capacity of {max_total_kg:.2f} kg.")

            final_sku_distribution = {sku_id: math.floor((max_total_kg * ratio) / 100) * 100 for sku_id, ratio in sku_ratio.items()}
            logging.info(f"Final achievable production (rounded): {final_sku_distribution}")

            return {
            "input_sku_ratio": sku_ratio,
            "system_bottleneck_stage": bottleneck_stage,
            "bottleneck_stage_actual_capacity_kg": round(stage_capacities_kg.get(bottleneck_stage, 0), 2),
            "implied_max_total_capacity_kg": round(max_total_kg, 2),
            "final_achievable_production_kg": final_sku_distribution,
            "details_stage_capacities_kg_per_day": {k: round(v, 2) for k, v in stage_capacities_kg.items()}
        }

    def check_feasibility(self, production_plan: List[Dict[str, Any]], optimize_hard_constraints: bool = False) -> Dict[str, Any]:
        """
        Checks if a specific production plan is feasible using the 'Capacity Map' approach.
        Includes an optional optimization loop for hard constraints.
        """
        logging.info(f"--- Starting `check_feasibility` for plan with {len(production_plan)} items. ---")
        if not production_plan:
            return {"overall_status": "FEASIBLE", "reason": "No production items in the plan."}

        # === Phase 1: Calculate the Plan's Specific Ratio ===
        total_requested_kg = sum(item['quantity_kg'] for item in production_plan)
        if total_requested_kg == 0:
            return {"overall_status": "FEASIBLE", "reason": "Total requested quantity is zero."}

        plan_ratio = {item['sku_id']: item['quantity_kg'] / total_requested_kg for item in production_plan}
        logging.info(f"Calculated plan-specific ratio: {plan_ratio}")

        # === Phase 2: Run Capacity Map for this Specific Ratio ===
        capacity_report = self.map_maximum_capacity(plan_ratio)
        max_capacity_kg = capacity_report.get("implied_max_total_capacity_kg", 0)
        
        # === Phase 3: Analyze the Results and Build Report ===
        is_fully_achievable = max_capacity_kg >= total_requested_kg
        final_analysis = []
        hard_req_failure = False
        
        for item in production_plan:
            sku_id = item['sku_id']
            requested = item['quantity_kg']
            
            if is_fully_achievable:
                achieved = requested
            else:
                achieved_production_map = capacity_report.get("final_achievable_production_kg", {})
                achieved = achieved_production_map.get(sku_id, 0)

            final_analysis.append({
                "sku_id": sku_id,
                "type": item['type'],
                "requested_kg": requested,
                "achievable_kg": achieved
            })
            
            if achieved < requested and item['type'] == 'hard':
                hard_req_failure = True

        # === Phase 4: Determine Final Status ===
        if is_fully_achievable:
            final_status = "FEASIBLE"
            report_analysis = final_analysis
            bottleneck = capacity_report.get("system_bottleneck_stage")
        elif hard_req_failure:
            initial_infeasible_report = {
                "overall_status": "INFEASIBLE",
                "analysis": final_analysis,
                "system_bottleneck_for_this_plan": capacity_report.get("system_bottleneck_stage")
            }
            if optimize_hard_constraints:
                # If optimization is enabled, run the loop
                return self._run_optimization_loop(final_analysis, initial_infeasible_report)
            else:
                # Otherwise, just return the infeasible report
                return initial_infeasible_report
        else: # Only soft requirements were impacted
            final_status = "FEASIBLE_WITH_ADJUSTMENTS"
            report_analysis = final_analysis
            bottleneck = capacity_report.get("system_bottleneck_stage")
            
        return {
            "overall_status": final_status,
            "analysis": report_analysis,
            "system_bottleneck_for_this_plan": bottleneck
        }

    def _run_optimization_loop(self, production_plan: List[Dict[str, Any]], initial_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to find a feasible production mix by iteratively adjusting ratios
        to favor failing hard constraints.
        """
        logging.info("--- Entering Hard Constraint Optimization Loop ---")
        MAX_ATTEMPTS = 3
        
        current_plan_ratio = {item['sku_id']: item['quantity_kg'] for item in production_plan}
        
        for attempt in range(MAX_ATTEMPTS):
            logging.info(f"Optimization Attempt {attempt + 1}/{MAX_ATTEMPTS}")
            
            # Find the hard requirement with the largest percentage shortfall
            worst_shortfall = 1.0
            target_sku = None
            for item in production_plan:
                if item['type'] == 'hard':
                    requested = item['requested_kg']
                    achieved = item['achievable_kg']
                    if requested > 0 and (achieved / requested) < worst_shortfall:
                        worst_shortfall = achieved / requested
                        target_sku = item['sku_id']
            
            if not target_sku:
                # This shouldn't happen if we entered the loop, but as a safeguard
                logging.warning("Optimization loop entered but no hard requirement shortfall found.")
                return initial_report

            # Adjust the ratio: increase the weight of the failing SKU
            # We use a factor to "boost" its importance in the mix.
            # If it achieved 80% (0.8), the factor is 1/0.8 = 1.25
            adjustment_factor = 1.25 # Default boost
            if worst_shortfall > 0 and worst_shortfall < 1:
                adjustment_factor = 1 / worst_shortfall
            
            # Cap the adjustment to prevent extreme swings
            adjustment_factor = min(adjustment_factor, 2.0)
            
            logging.info(f"Boosting ratio for '{target_sku}' by a factor of {adjustment_factor:.2f}")
            current_plan_ratio[target_sku] *= adjustment_factor
            
            # Re-normalize the ratios so they sum to 1.0
            total_ratio_weight = sum(current_plan_ratio.values())
            new_plan_ratio = {sku: weight / total_ratio_weight for sku, weight in current_plan_ratio.items()}

            # Re-run the capacity map with the new, adjusted ratio
            new_capacity_report = self.map_maximum_capacity(new_plan_ratio)
            
            # Check if the new plan is successful
            new_achieved_production = new_capacity_report.get("final_achievable_production_kg", {})
            hard_req_failure = False
            for item in production_plan:
                if item['type'] == 'hard':
                    if new_achieved_production.get(item['sku_id'], 0) < item['quantity_kg']:
                        hard_req_failure = True
                        break # A hard req still failed, continue the loop
            
            # If all hard requirements are now met, we have found a solution
            if not hard_req_failure:
                logging.info("Optimization successful. Found a feasible plan.")
                # Rebuild the analysis with the successful results
                final_analysis = []
                for item in production_plan:
                    final_analysis.append({
                        "sku_id": item['sku_id'],
                        "type": item['type'],
                        "requested_kg": item['quantity_kg'],
                        "achievable_kg": new_achieved_production.get(item['sku_id'], 0)
                    })
                return {
                    "overall_status": "FEASIBLE_WITH_ADJUSTMENTS",
                    "analysis": final_analysis,
                    "system_bottleneck_for_this_plan": new_capacity_report.get("system_bottleneck_stage")
                }

        # If the loop finishes without a solution, return the original infeasible report
        logging.warning("Optimization loop finished without finding a feasible solution.")
        return initial_report

    def _build_flow_map(self, product_def: Product, skus_in_category: Dict[str, float]) -> Dict[str, Dict]:
        """Pass 1: Analyzes a recipe to create a map of speeds and durations."""
        flow_map = {}
        recipe = product_def.processing_steps
        logging.info(f"[{product_def.product_category}] Building Flow Map...")

        for step in recipe:
            if step.process_type == ProcessType.PACKAGING:
                effective_speed = self._get_effective_packing_speed(product_def, skus_in_category)
                flow_map[step.step_id] = {'type': 'Flow', 'speed_kg_per_min': effective_speed}
                logging.info(f"  - Mapped PACKAGING step '{step.step_id}' with effective speed: {effective_speed:.2f} kg/min")
            elif getattr(step, 'scheduling_rule', None) == SchedulingRule.ZERO_STAGNATION:
                total_speed_lpm = 0
                if step.requirements and step.requirements[0].compatible_ids:
                    for res_id in step.requirements[0].compatible_ids:
                        res = self.equipment.get(res_id)
                        # --- BUG FIX #1: Assume equipment speed is in Liters per MINUTE ---
                        if res and isinstance(res, Equipment): 
                            total_speed_lpm += res.processing_speed
                if total_speed_lpm > 0:
                    # Assuming 1L = 1kg
                    speed_kg_per_min = total_speed_lpm
                    flow_map[step.step_id] = {'type': 'Flow', 'speed_kg_per_min': speed_kg_per_min}
                    logging.info(f"  - Mapped FLOW step '{step.step_id}' with speed: {speed_kg_per_min:.2f} kg/min")
            else:
                flow_map[step.step_id] = {'type': 'Static', 'duration_mins': step.duration_minutes}
                logging.info(f"  - Mapped STATIC step '{step.step_id}' with duration: {step.duration_minutes} mins")
        return flow_map

    def _get_effective_packing_speed(self, product_def: Product, skus_in_category: Dict[str, float]) -> float:
        """Calculates the final 'pull' speed from the end of the line, adjusted for bottlenecks."""
        packing_step = next((s for s in product_def.processing_steps if s.process_type == ProcessType.PACKAGING), None)
        if not packing_step: return 0

        unconstrained_packing_speed, unconstrained_storing_speed = self._calculate_unconstrained_rates(packing_step, skus_in_category)
        logging.info(f"[{product_def.product_category}] Unconstrained Packing Speed: {unconstrained_packing_speed:.2f} kg/min")
        
        post_packing_steps = [s for s in product_def.processing_steps if s.process_type == ProcessType.POST_PACKAGING]
        reduction_factor = 1.0
        for step in post_packing_steps:
            if not step.requirements or not step.requirements[0].compatible_ids: continue
            
            total_room_capacity = 0
            compatible_room_ids = step.requirements[0].compatible_ids
            for room_id in compatible_room_ids:
                room = self.rooms.get(room_id)
                if room:
                    total_room_capacity += room.capacity_units
            
            if total_room_capacity == 0: continue

            total_space_required = unconstrained_storing_speed * step.duration_minutes
            logging.info(f"  - Room Stage '{step.step_id}': Requires {total_space_required:.2f} units (Total Capacity: {total_room_capacity}).")
            if total_space_required > total_room_capacity:
                factor = total_room_capacity / total_space_required
                logging.warning(f"    - Room stage '{step.step_id}' is a bottleneck! Reduction factor: {factor:.3f}")
                reduction_factor = min(reduction_factor, factor)

        effective_speed = unconstrained_packing_speed * reduction_factor
        logging.info(f"[{product_def.product_category}] Final Effective Packing Speed: {effective_speed:.2f} kg/min")
        return effective_speed

    def _calculate_unconstrained_rates(self, packing_step: ProcessingStep, skus_in_category: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculates packing/storing speed. It correctly calculates the equivalent speed
        for SKUs sharing a line using a properly weighted harmonic mean, and then sums
        the speeds of the independent, parallel lines.
        """
        total_packing_speed_kg_per_min = 0
        total_storing_speed_units_per_min = 0
        line_usage = {}
        for sku_id in skus_in_category:
            for line_id, line_def in self.lines.items():
                if sku_id in line_def.compatible_skus_max_production:
                    if line_id not in line_usage: line_usage[line_id] = []
                    line_usage[line_id].append(sku_id)

        for line_id, skus_on_line in line_usage.items():
            line_def = self.lines[line_id]
            line_packing_speed = 0

            # --- FIX: Calculate total_ratio_on_line before the if/else block ---
            # This ensures the variable always exists.
            total_ratio_on_line = sum(skus_in_category.get(sku_id, 0) for sku_id in skus_on_line)

            if len(skus_on_line) == 1:
                line_packing_speed = line_def.compatible_skus_max_production.get(skus_on_line[0], 0)
            else:
                inverse_speed_sum = 0
                if total_ratio_on_line > 0:
                    for sku_id in skus_on_line:
                        normalized_line_ratio = skus_in_category.get(sku_id, 0) / total_ratio_on_line
                        speed = line_def.compatible_skus_max_production.get(sku_id, 0) * OEE_FACTOR
                        if speed > 0:
                            inverse_speed_sum += normalized_line_ratio / speed
                
                if inverse_speed_sum > 0:
                    line_packing_speed = 1 / inverse_speed_sum

            total_packing_speed_kg_per_min += line_packing_speed
            
            if total_ratio_on_line > 0:
                for sku_id in skus_on_line:
                    normalized_line_ratio = skus_in_category.get(sku_id, 0) / total_ratio_on_line
                    speed_for_sku_on_line = line_packing_speed * normalized_line_ratio
                    total_storing_speed_units_per_min += speed_for_sku_on_line * self.skus[sku_id].inventory_size

        return total_packing_speed_kg_per_min, total_storing_speed_units_per_min

    def _get_engagement_time(self, current_step: ProcessingStep, product_def: Product, flow_map: Dict[str, Dict]) -> Tuple[float, float]:
        """
        Pass 2: Uses the flow_map to calculate the true engagement time and batch size for a step.
        This version correctly handles POST_PACKAGING steps.
        """
        step_info = flow_map.get(current_step.step_id)
        if not step_info: return 0, 0
        if not current_step.requirements or not current_step.requirements[0].compatible_ids: return 0, 0
        
        resource_id = current_step.requirements[0].compatible_ids[0]
        resource = self.all_resources.get(resource_id)
        if not resource: return 0, 0

        batch_size_liters = self.DEFAULT_BATCH_SIZE_LITERS
        if isinstance(resource, Tank) and hasattr(resource, 'capacity_liters') and resource.capacity_liters > 0:
            batch_size_liters = resource.capacity_liters * OEE_FACTOR
        elif hasattr(product_def, 'max_batch_size') and product_def.max_batch_size > 0:
            batch_size_liters = product_def.max_batch_size

        cip_time = getattr(resource, 'CIP_duration_minutes', 45)
        
        # --- NEW LOGIC: Handle Post-Packaging Steps Separately ---
        if current_step.process_type == ProcessType.POST_PACKAGING:
            # The engagement time for a room is just its fixed duration.
            return step_info.get('duration_mins', 0), batch_size_liters

        if step_info['type'] == 'Flow':
            speed_kg_per_min = step_info.get('speed_kg_per_min', 0)
            if speed_kg_per_min > 0:
                time_to_process = (batch_size_liters / speed_kg_per_min)
                return time_to_process + cip_time, batch_size_liters
            return 0, batch_size_liters
            
        elif step_info['type'] == 'Static':
            process_time = step_info.get('duration_mins', 0)
            recipe = product_def.processing_steps
            current_index = recipe.index(current_step)
            upstream_speed = next((flow_map.get(recipe[i].step_id, {}).get('speed_kg_per_min', 0) for i in range(current_index - 1, -1, -1) if flow_map.get(recipe[i].step_id, {}).get('type') == 'Flow'), 0)
            downstream_speed = next((flow_map.get(recipe[i].step_id, {}).get('speed_kg_per_min', 0) for i in range(current_index + 1, len(recipe)) if flow_map.get(recipe[i].step_id, {}).get('type') == 'Flow'), 0)
            
            fill_time = (batch_size_liters / upstream_speed) if upstream_speed > 0 else 0
            drain_time = (batch_size_liters / downstream_speed) if downstream_speed > 0 else 0
            
            return fill_time + process_time + drain_time + cip_time, batch_size_liters

        return 0, batch_size_liters

    def _calculate_plan_cost(self, plan: List[Dict[str, Any]]) -> Tuple[Dict, Dict]:
        """
        Helper for check_feasibility to calculate resource cost for a plan.
        This version correctly groups items by category to build a single, consistent
        flow map for each category in the plan.
        """
        time_usage = {res_id: 0.0 for res_id in self.time_based_resources.keys()}
        space_usage = {res_id: 0.0 for res_id in self.rooms.keys()}

        # --- NEW LOGIC: Group plan items by product category ---
        plan_by_category = {}
        for item in plan:
            sku_def = self.skus.get(item['sku_id'])
            if not sku_def: continue
            category = sku_def.product_category
            if category not in plan_by_category:
                plan_by_category[category] = []
            plan_by_category[category].append(item)

        # --- Process each category with its own consistent flow map ---
        for category, items_in_category in plan_by_category.items():
            product_def = self.products.get(category)
            if not product_def: continue

            # Create a representative SKU ratio for this specific plan
            total_kg_in_category = sum(item['quantity_kg'] for item in items_in_category)
            if total_kg_in_category == 0: continue
            
            plan_sku_ratio = {item['sku_id']: item['quantity_kg'] / total_kg_in_category for item in items_in_category}

            # Build ONE flow map for the entire category based on the plan's mix
            flow_map = self._build_flow_map(product_def, plan_sku_ratio)

            # Calculate cost for each item using the consistent flow map
            for item in items_in_category:
                sku_id, kg_required = item['sku_id'], item['quantity_kg']
                sku_def = self.skus.get(sku_id)
                if not sku_def: continue

                # Determine batch size, using a default if necessary
                # This could be enhanced to use the resource's actual capacity
                batch_size = self.DEFAULT_BATCH_SIZE_LITERS
                num_batches = math.ceil(kg_required / batch_size) if batch_size > 0 else 0

                for step in product_def.processing_steps:
                    engagement_time_per_batch, _ = self._get_engagement_time(step, product_def, flow_map)
                    
                    if engagement_time_per_batch > 0 and step.requirements and step.requirements[0].compatible_ids:
                        compatible_ids = step.requirements[0].compatible_ids
                        num_parallel_resources = len(compatible_ids)
                        if num_parallel_resources > 0:
                            total_time_for_step = engagement_time_per_batch * num_batches
                            time_per_resource = total_time_for_step / num_parallel_resources
                            for res_id in compatible_ids:
                                if res_id in time_usage:
                                    time_usage[res_id] += time_per_resource
                
                # Calculate space usage for the item
                for step in product_def.processing_steps:
                    if step.process_type == ProcessType.POST_PACKAGING and step.requirements and step.requirements[0].compatible_ids:
                        compatible_ids = step.requirements[0].compatible_ids
                        space_per_room = (kg_required * sku_def.inventory_size) / len(compatible_ids) if compatible_ids else 0
                        for room_id in compatible_ids:
                            if room_id in space_usage: space_usage[room_id] += space_per_room
                            
        return time_usage, space_usage

    def _group_skus_by_category(self, sku_ratio: Dict[str, float]) -> Dict[str, Dict]:
        """Groups SKUs by product category and normalizes their ratios."""
        categorized = {}
        for sku_id, ratio in sku_ratio.items():
            sku_def = self.skus.get(sku_id)
            if not sku_def: continue
            category = sku_def.product_category
            if category not in categorized:
                categorized[category] = {'total_ratio': 0, 'skus_in_category': {}}
            categorized[category]['total_ratio'] += ratio
            categorized[category]['skus_in_category'][sku_id] = ratio
        for category, data in categorized.items():
            if data['total_ratio'] > 0:
                for sku_id in data['skus_in_category']: data['skus_in_category'][sku_id] /= data['total_ratio']
        return categorized

    def _solve_capacity_constraints(self, stage_capacities: Dict[str, float], categorized_skus: Dict) -> Tuple[float, str]:
        """Finds the maximum total production possible given all stage bottlenecks."""
        max_possible_kg = float('inf')
        bottleneck_stage = "None"
        for stage_name, capacity_kg in stage_capacities.items():
            owner_category = next((cat for cat, prod_def in self.products.items() if any(s.step_id == stage_name for s in prod_def.processing_steps)), None)
            if owner_category:
                category_ratio = categorized_skus.get(owner_category, {}).get('total_ratio', 1.0)
                implied_total_kg = capacity_kg / category_ratio if category_ratio > 0 else float('inf')
                if implied_total_kg < max_possible_kg:
                    max_possible_kg = implied_total_kg
                    bottleneck_stage = stage_name
            elif capacity_kg < max_possible_kg: # Fallback for truly shared resources
                max_possible_kg = capacity_kg
                bottleneck_stage = stage_name
        return max_possible_kg, bottleneck_stage

# --- Standalone Execution Example ---
if __name__ == "__main__":
    logging.info("--- Running Standalone Capacity Analysis ---")
    data_loader = DataLoader()
    data_loader.clear_all_data()
    try:
        data_loader.load_sample_data()
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.fatal(f"Could not load data. Error: {e}")
        exit()

    analyzer = CapacityAnalyzer(
        products=config.PRODUCTS, equipment=config.EQUIPMENTS, lines=config.LINES,
        tanks=config.TANKS, skus=config.SKUS, rooms=config.ROOMS
    )

    # --- Test 1: Capacity Mapping ---
    sku_production_ratio = {
    "ROS-LAS-170G": 0.25,
    "MNG-LAS-170G": 0.25,
    "PLN-PCH-CRD-1KG": 0.115,
    "PLN-PCH-CRD-400G": 0,
    "PLN-PCH-CRD-200G": 0.115,
    "SEL-BKT-15KG": 0.09,
    "SEL-BKT-5KG": 0.18,
    "SEL-BKT-2KG": 0,
    "SEL-BKT-1KG": 0,
    "SEL-CUP-200G": 0,
    "SEL-CUP-400G": 0
}
    logging.info(f"\n--- 1. Mapping Maximum Capacity for Ratio: {sku_production_ratio} ---")
    capacity_report = analyzer.map_maximum_capacity(sku_production_ratio)
    print("\n--- Capacity Mapping Report ---")
    print(json.dumps(capacity_report, indent=2))

    # --- Test 2: Feasibility Check ---
    production_plan = [
        {"sku_id": "SEL-BKT-5KG", "quantity_kg": 5000, "type": "hard"},
        #{"sku_id": "SEL-BKT-1KG", "quantity_kg": 1200, "type": "hard"},
        #{"sku_id": "SEL-BKT-2KG", "quantity_kg": 1500, "type": "soft"},
        {"sku_id": "SEL-BKT-15KG", "quantity_kg": 3000, "type": "soft"},
        {"sku_id": "PLN-PCH-CRD-1KG", "quantity_kg": 4000, "type": "soft"},
        {"sku_id": "PLN-PCH-CRD-200G", "quantity_kg": 3600, "type": "soft"}

    ]
    logging.info(f"\n--- 2. Checking Feasibility for Plan ---")
    feasibility_report = analyzer.check_feasibility(production_plan, optimize_hard_constraints=True)
    print("\n--- Feasibility Check Report ---")
    print(json.dumps(feasibility_report, indent=2))