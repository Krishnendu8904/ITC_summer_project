from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ResourceType(Enum):
    TANK = "tank"
    EQUIPMENT = "equipment"
    LINE = "line"

@dataclass
class Resource:
    name: str
    type: ResourceType
    available_at: datetime
    last_category: Optional[str] = None

@dataclass
class Task:
    step_name: str
    resource_name: str
    duration_hours: float
    start_time: datetime
    end_time: datetime

@dataclass
class Job:
    sku_id: str
    category: str  # "SELECT-CURD" or "LOW-FAT-CURD"
    packaging_type: str  # "BUCKET" or "CUP"

class ProcessScheduler:
    def __init__(self, start_time: datetime):
        self.current_time = start_time
        self.resources = self._initialize_resources()
        self.schedule = []
        
    def _initialize_resources(self) -> Dict[str, Resource]:
        """Initialize all available resources"""
        resources = {}
        
        # Tanks
        resources["MST-1"] = Resource("MST-1", ResourceType.TANK, self.current_time)
        resources["LT-1"] = Resource("LT-1", ResourceType.TANK, self.current_time)
        resources["LT-2"] = Resource("LT-2", ResourceType.TANK, self.current_time)
        
        # Equipment
        resources["PST-1"] = Resource("PST-1", ResourceType.EQUIPMENT, self.current_time)
        
        # Packaging lines
        resources["BUCKET-LINE-1"] = Resource("BUCKET-LINE-1", ResourceType.LINE, self.current_time)
        resources["CUP-LINE-1"] = Resource("CUP-LINE-1", ResourceType.LINE, self.current_time)
        
        return resources
    
    def _get_process_steps(self, job: Job) -> List[Tuple[str, List[str], float]]:
        """Define the process steps for each SKU"""
        packaging_line = "BUCKET-LINE-1" if job.packaging_type == "BUCKET" else "CUP-LINE-1"
        
        return [
            ("MST Standardisation", ["MST-1"], 8.0),
            ("Pasteurisation", ["PST-1"], 1.0),
            ("LT Standardisation", ["LT-1", "LT-2"], 0.5),
            ("Inoculation", ["LT-1", "LT-2"], 0.5),  # Uses same tank as LT Standardisation
            ("Packing", [packaging_line], 1.0)
        ]
    
    def _needs_cip(self, resource: Resource, job_category: str) -> bool:
        """Check if CIP is needed when switching categories"""
        return (resource.last_category is not None and 
                resource.last_category != job_category)
    
    def _find_earliest_available_resource(self, resource_names: List[str], 
                                        job_category: str, duration: float,
                                        earliest_start: datetime) -> Tuple[str, datetime]:
        """Find the earliest available resource from the list"""
        best_resource = None
        best_start_time = None
        
        for resource_name in resource_names:
            resource = self.resources[resource_name]
            
            # Calculate when this resource can start
            resource_available = max(resource.available_at, earliest_start)
            
            # Add CIP time if needed
            if self._needs_cip(resource, job_category):
                resource_available += timedelta(hours=1)
            
            # Choose the earliest available resource
            if best_start_time is None or resource_available < best_start_time:
                best_resource = resource_name
                best_start_time = resource_available
        
        return best_resource, best_start_time
    
    def _schedule_cip(self, resource_name: str, start_time: datetime, job_category: str):
        """Schedule a CIP operation"""
        resource = self.resources[resource_name]
        if self._needs_cip(resource, job_category):
            cip_task = Task(
                step_name=f"CIP (Category change from {resource.last_category} to {job_category})",
                resource_name=resource_name,
                duration_hours=1.0,
                start_time=start_time - timedelta(hours=1),
                end_time=start_time
            )
            self.schedule.append(cip_task)
    
    def schedule_job(self, job: Job) -> List[Task]:
        """Schedule all tasks for a single job"""
        job_tasks = []
        process_steps = self._get_process_steps(job)
        
        # Track the earliest time the next step can start
        earliest_next_start = self.current_time
        last_tank_used = None  # For LT Standardisation -> Inoculation continuity
        
        for i, (step_name, possible_resources, duration) in enumerate(process_steps):
            # Special handling for Inoculation - must use same tank as LT Standardisation
            if step_name == "Inoculation" and last_tank_used:
                possible_resources = [last_tank_used]
            
            # Find the earliest available resource
            resource_name, start_time = self._find_earliest_available_resource(
                possible_resources, job.category, duration, earliest_next_start
            )
            
            # Schedule CIP if needed
            if self._needs_cip(self.resources[resource_name], job.category):
                self._schedule_cip(resource_name, start_time, job.category)
            
            # Create the task
            end_time = start_time + timedelta(hours=duration)
            task = Task(
                step_name=f"{job.sku_id} - {step_name}",
                resource_name=resource_name,
                duration_hours=duration,
                start_time=start_time,
                end_time=end_time
            )
            
            job_tasks.append(task)
            self.schedule.append(task)
            
            # Update resource availability and category
            resource = self.resources[resource_name]
            resource.available_at = end_time
            resource.last_category = job.category
            
            # Track tank for LT operations
            if step_name == "LT Standardisation":
                last_tank_used = resource_name
            
            # Next step can start immediately after this one ends
            earliest_next_start = end_time
        
        return job_tasks
    
    def schedule_jobs(self, jobs: List[Job]) -> Dict:
        """Schedule multiple jobs and return complete schedule"""
        job_schedules = {}
        
        for job in jobs:
            print(f"Scheduling {job.sku_id} ({job.category}, {job.packaging_type})")
            job_tasks = self.schedule_job(job)
            job_schedules[job.sku_id] = job_tasks
        
        return {
            'job_schedules': job_schedules,
            'complete_schedule': sorted(self.schedule, key=lambda x: x.start_time),
            'resource_utilization': self._get_resource_utilization()
        }
    
    def _get_resource_utilization(self) -> Dict[str, datetime]:
        """Get final availability time for each resource"""
        return {name: resource.available_at for name, resource in self.resources.items()}
    
    def print_schedule(self, schedule_result: Dict):
        """Print a formatted schedule"""
        print("\n" + "="*80)
        print("COMPLETE PRODUCTION SCHEDULE")
        print("="*80)
        
        for task in schedule_result['complete_schedule']:
            print(f"{task.start_time.strftime('%Y-%m-%d %H:%M')} - "
                  f"{task.end_time.strftime('%H:%M')} | "
                  f"{task.resource_name:12} | {task.step_name}")
        
        print("\n" + "="*80)
        print("RESOURCE FINAL AVAILABILITY")
        print("="*80)
        for resource_name, available_at in schedule_result['resource_utilization'].items():
            print(f"{resource_name:15} | Available at: {available_at.strftime('%Y-%m-%d %H:%M')}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize scheduler
    start_time = datetime(2025, 6, 25, 8, 0, 0)
    scheduler = ProcessScheduler(start_time)
    
    # Define sample jobs
    sample_jobs = [
        Job("SKU-SC-001", "SELECT-CURD", "BUCKET"),
        Job("SKU-SC-002", "SELECT-CURD", "CUP"),
        Job("SKU-LF-001", "LOW-FAT-CURD", "BUCKET"),
        Job("SKU-LF-002", "LOW-FAT-CURD", "CUP"),
    ]
    
    # Schedule all jobs
    schedule_result = scheduler.schedule_jobs(sample_jobs)
    
    # Print the complete schedule
    scheduler.print_schedule(schedule_result)
    
    print("\n" + "="*80)
    print("JOB-WISE BREAKDOWN")
    print("="*80)
    
    for sku_id, tasks in schedule_result['job_schedules'].items():
        print(f"\n{sku_id}:")
        for task in tasks:
            print(f"  {task.start_time.strftime('%H:%M')}-{task.end_time.strftime('%H:%M')} "
                  f"| {task.resource_name:12} | {task.step_name}")