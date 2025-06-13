import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
from collections import defaultdict
import heapq
from itertools import combinations
from schedulers import max_flow_scheduler, proportional_scheduler, pulp_scheduler, constrains_scheduler
from scheduler import ProductionScheduler
import config
from models.data_models import (
    SKU, Line, Tank, Shift, UserIndent, ScheduleItem, SchedulingResult,
    LineStatus, Priority, ProductTypeRegistry, FlowEdge, ProductionSlot
)

logger = logging.getLogger(__name__)

# Hybrid Scheduler combining multiple OR techniques
class HybridORScheduler:
    """Hybrid scheduler combining Max Flow, Constraint Programming, and Heuristics"""
    
    def __init__(self):
        self.max_flow_scheduler = max_flow_scheduler.MaxFlowProductionScheduler()
        self.cp_scheduler = constrains_scheduler.ConstraintProgrammingScheduler()
        self.original_scheduler = ProductionScheduler()
        
    def schedule_production_hybrid(self) -> SchedulingResult:
        """Schedule using hybrid approach"""
        logger.info("Starting hybrid OR scheduling")
        
        # Try all three approaches
        results = {}
        
        try:
            results['max_flow'] = self.max_flow_scheduler.schedule_production()
            logger.info(f"Max Flow result: {results['max_flow'].efficiency_score:.2f}% efficiency")
        except Exception as e:
            logger.error(f"Max Flow scheduling failed: {e}")
            results['max_flow'] = None
        
        try:
            results['constraint_programming'] = self.cp_scheduler.schedule_production_cp()
            logger.info(f"CP result: {results['constraint_programming'].efficiency_score:.2f}% efficiency")
        except Exception as e:
            logger.error(f"CP scheduling failed: {e}")
            results['constraint_programming'] = None
        
        try:
            results['heuristic'] = self.original_scheduler.schedule_production()
            logger.info(f"Heuristic result: {results['heuristic'].efficiency_score:.2f}% efficiency")
        except Exception as e:
            logger.error(f"Heuristic scheduling failed: {e}")
            results['heuristic'] = None
        
        # Select best result based on multiple criteria
        best_result = self._select_best_result(results)
        
        if best_result:
            logger.info(f"Selected best result with {best_result.efficiency_score:.2f}% efficiency")
            return best_result
        else:
            logger.error("All scheduling approaches failed")
            return SchedulingResult([], [], 0, 0, ["All scheduling approaches failed"])
    
    def _select_best_result(self, results: Dict[str, Optional[SchedulingResult]]) -> Optional[SchedulingResult]:
        """Select the best result from multiple approaches"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None
        
        # Multi-criteria scoring
        scored_results = []
        
        for approach, result in valid_results.items():
            score = 0.0
            
            # Production volume (40% weight)
            total_demand = sum(indent.qty_required for indent in config.USER_INDENTS.values())
            if total_demand > 0:
                fulfillment_rate = result.total_production / total_demand
                score += fulfillment_rate * 0.4
            
            # Efficiency score (30% weight)
            score += (result.efficiency_score / 100) * 0.3
            
            # Number of scheduled items (20% weight)
            total_indents = len(config.USER_INDENTS)
            if total_indents > 0:
                scheduled_rate = len(result.schedule_items) / total_indents
                score += scheduled_rate * 0.2
            
            # Penalty for unfulfilled indents (10% weight)
            if total_indents > 0:
                unfulfilled_penalty = len(result.unfulfilled_indents) / total_indents
                score -= unfulfilled_penalty * 0.1
            
            scored_results.append((score, approach, result))
        
        # Return result with highest score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_approach, best_result = scored_results[0]
        
        logger.info(f"Best approach: {best_approach} with score: {best_score:.3f}")
        return best_result
