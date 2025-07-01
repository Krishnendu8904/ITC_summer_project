import os
import json
import copy
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

# --- Imports from the main scheduler project ---
from heuristic_scheduler import HeuristicScheduler, ScheduleStatus
from utils.data_models import UserIndent, Priority
from utils.data_loader import DataLoader
import config


def setup_optimizer_logger():
    """Configures a logger for the optimizer, directing output to console and a file."""
    logger = logging.getLogger('OptimizerLogger')
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if this function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a unique log file name for each run
    log_filename = f"optimizer_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # File Handler - to save logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console Handler - to print logs to the console
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename


class ProductionOptimizer:
    """
    An engine to find the maximum feasible production mix by iteratively
    running mock schedules and adjusting order quantities.
    """

    def __init__(self, base_config: Dict, reduction_step=500, max_iterations=20, category_cap=10000):
        """
        Initializes the optimizer.

        Args:
            base_config (Dict): A dictionary containing the base data for skus,
                                products, lines, tanks, etc.
            reduction_step (int): The volume (in Liters) to reduce an order by
                                  in each iteration of the reduction phase.
            max_iterations (int): A safeguard to prevent infinite loops.
            category_cap (int): The maximum total volume allowed per product category.
        """
        self.base_config = base_config
        self.reduction_step = reduction_step
        self.max_iterations = max_iterations
        self.category_cap = category_cap
        self.logger = logging.getLogger('OptimizerLogger')

    def _apply_category_caps(self, indents: Dict[str, UserIndent]) -> Dict[str, UserIndent]:
        """
        Enforces a maximum total volume for each product category.
        """
        self.logger.info(f"Applying product category cap of {self.category_cap} L.")
        
        # Group indents by product category
        indents_by_category = defaultdict(list)
        for indent in indents.values():
            sku = self.base_config['skus'].get(indent.sku_id)
            if sku:
                indents_by_category[sku.product_category].append(indent)

        for category, indent_list in indents_by_category.items():
            total_volume = sum(i.qty_required_liters for i in indent_list)
            
            if total_volume > self.category_cap:
                self.logger.warning(f"Category '{category}' exceeds cap. Total: {total_volume} L. Reducing...")
                
                volume_to_reduce = total_volume - self.category_cap
                
                # Sort orders by lowest priority first to reduce them first
                sorted_indents = sorted(indent_list, key=lambda x: x.priority.value, reverse=True)
                
                for indent_to_reduce in sorted_indents:
                    if volume_to_reduce <= 0:
                        break
                    
                    reduction_amount = min(indent_to_reduce.qty_required_liters, volume_to_reduce)
                    
                    self.logger.info(f"  - Reducing order {indent_to_reduce.order_no} by {reduction_amount} L.")
                    indent_to_reduce.qty_required_liters -= reduction_amount
                    volume_to_reduce -= reduction_amount

        return indents

    def _run_mock_schedule(self, indents_to_check: Dict[str, UserIndent]) -> bool:
        """
        Runs a full, in-memory scheduling simulation with a given set of indents.
        """
        self.logger.info("--- Running Mock Schedule ---")
        self.logger.info(f"Testing with {len(indents_to_check)} orders...")

        # Create a fresh scheduler instance for the simulation
        scheduler = HeuristicScheduler(
            indents=indents_to_check,
            skus=self.base_config['skus'],
            products=self.base_config['products'],
            lines=self.base_config['lines'],
            tanks=self.base_config['tanks'],
            equipments=self.base_config['equipments'],
            shifts=self.base_config['shifts']
        )

        # Run the full scheduling logic
        scheduler.run_heuristic_scheduler()

        # Check if all generated tasks were successfully booked
        is_feasible = not any(
            task.status == ScheduleStatus.FAILED for task in scheduler.master_task_list
        )
        
        if is_feasible:
            self.logger.info("  -> Result: FEASIBLE")
        else:
            self.logger.info("  -> Result: INFEASIBLE")

        return is_feasible

    def find_feasible_baseline(self, initial_indents: Dict[str, UserIndent]) -> Dict:
        """
        Takes a set of desired orders and iteratively reduces quantities until
        a feasible schedule is found.
        """
        self.logger.info("--- Starting Feasibility Baseline Search ---")
        
        current_indents = copy.deepcopy(initial_indents)

        # --- NEW STEP: Apply category caps before starting iterations ---
        current_indents = self._apply_category_caps(current_indents)
        # ---

        for i in range(self.max_iterations):
            self.logger.info(f"--- Iteration {i+1} ---")
            
            if self._run_mock_schedule(current_indents):
                self.logger.info("Feasible baseline found!")
                return {
                    "status": "Success",
                    "feasible_indents": current_indents,
                }

            orders_to_reduce = sorted(
                current_indents.values(), 
                key=lambda x: (x.priority.value, x.qty_required_liters), 
                reverse=True
            )

            if not orders_to_reduce:
                self.logger.error("No orders left to reduce. Cannot find feasible schedule.")
                break

            order_to_modify = orders_to_reduce[0]
            
            self.logger.warning(f"Reducing lowest priority order: {order_to_modify.order_no}")
            self.logger.info(f"  Old Quantity: {order_to_modify.qty_required_liters} L")

            order_to_modify.qty_required_liters -= self.reduction_step
            
            self.logger.info(f"  New Quantity: {order_to_modify.qty_required_liters} L")

            if order_to_modify.qty_required_liters <= 0:
                self.logger.warning(f"Order {order_to_modify.order_no} removed as quantity reached zero.")
                del current_indents[order_to_modify.order_no]

        self.logger.error("Failed to find a feasible baseline within max iterations.")
        return {
            "status": "Failed",
            "message": "Could not find a feasible schedule within the iteration limit.",
        }


if __name__ == "__main__":
    # 1. Set up the logger
    logger, log_file = setup_optimizer_logger()
    logger.info("Running Production Optimizer in standalone mode...")

    # 2. Load all base configuration data once
    loader = DataLoader()
    loader.load_sample_data()
    base_config = {
        "skus": config.SKUS, "products": config.PRODUCTS, "lines": config.LINES,
        "tanks": config.TANKS, "equipments": config.EQUIPMENTS, "shifts": config.SHIFTS
    }

    # 3. Define the initial set of orders we want to make
    start_date = datetime.now()
    # This mix now has over 10,000L for Product-C to test the cap
    initial_orders = {
        "ORD_101": UserIndent(order_no="ORD_101", sku_id="BKT-SEL", qty_required_liters=7000, priority=Priority.MEDIUM, due_date=start_date + timedelta(days=3)),
        "ORD_102": UserIndent(order_no="ORD_102", sku_id="PCH-CRD", qty_required_liters=8000, priority=Priority.MEDIUM, due_date=start_date + timedelta(days=3)),
        "ORD_201": UserIndent(order_no="ORD_201", sku_id="CUP-SEL", qty_required_liters=7500, priority=Priority.HIGH, due_date=start_date + timedelta(days=3))
    }

    # 4. Initialize and run the optimizer
    optimizer = ProductionOptimizer(base_config, category_cap=10000)
    result = optimizer.find_feasible_baseline(initial_orders)

    # 5. Print the results
    logger.info("="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Final Status: {result['status']}")
    logger.info("="*50)

    if result['status'] == 'Success':
        # Using print here for final, clean output summary
        print("\n--- Initial Orders ---")
        for key, indent in initial_orders.items():
            print(f"  {key}: {indent.qty_required_liters} L")
        
        print("\n--- Feasible Baseline Orders (after capping and optimization) ---")
        for key, indent in result['feasible_indents'].items():
            print(f"  {key}: {indent.qty_required_liters} L")

    logger.info(f"Detailed logs have been saved to: {log_file}")