from __future__ import annotations
from datetime import datetime
import logging
from enum import Enum
import networkx as nx
from intervaltree import Interval, IntervalTree 
from copy import deepcopy

from typing import Any, Tuple, Dict, List, Optional, Union

from pydantic import BaseModel, Field, confloat, conint, PrivateAttr

# from pydantic import RootModel

from datetime import datetime, timedelta


def datetime_to_int(base: datetime, datetime_value: datetime):
    return int((datetime_value - base).total_seconds() / 60)


def int_to_datetime(base: datetime, int_value: int):
    return base + timedelta(minutes=int_value)


# class WeightVectorItem(RootModel):
#     root: conint(ge=0.0)

# Resources
class ResourceType(str, Enum):
    DISCRETE = 'DISCRETE'
    CONTINUES = 'CONTINUES'
class Resource(BaseModel):
    name: str
    measurement_unit: str
    measurement_type: ResourceType

class CostInterval(BaseModel):
    from_datetime: datetime
    to_datetime: datetime
    cost: float
    # cost levels [lb,ub,cost] It should cover the whole range [min,max] and should be sorted
    cost_levels: Optional[List[Tuple[float,float,float]]] = Field(default_factory=list)
    min: Optional[confloat(ge=0.0)] = 0.0
    max: Optional[float] = 10E10
    _start: int = PrivateAttr()
    _end: int = PrivateAttr()

    def datetime_int(self, base: datetime):
        self._start = datetime_to_int(base, self.from_datetime)
        self._end = datetime_to_int(base, self.to_datetime)
    
    def get_cost(self):
        return self.cost
    
    def get_cost_levels(self):
        if self.cost_levels == None:
            self._setup_levels()
        return self.cost_levels
    
    def _setup_levels(self):
        self.cost_levels = [self.min,self.max,self.cost]
    

class CostComponent(BaseModel):
    name: str
    resource_name: str = "Working Time"
    cost_intervals: List[CostInterval] = Field(..., min_length=0)
    _charge_per_minute: List[float] = PrivateAttr()
    _resource: Resource = PrivateAttr()

    def fix_after_load_from_json(self, base: datetime, parameters:Parameters):
        for ec in self.cost_intervals:
            ec.datetime_int(base)
        self.setup_charge_per_minute()
        self._resource = parameters.resources[self.resource_name]

    def __init__(self, name="Default Cost", cost_intervals=[], resource_name = "Electricity"):
        super().__init__(name=name, cost_intervals=cost_intervals, resource_name = resource_name)

    def add_cost_interval(self, cost_interval: CostInterval):
        self.cost_intervals.append(cost_interval)

    def setup_charge_per_minute(self):
        self._charge_per_minute = []
        # TODO should create a correct formula for cost based on given cost, unit and size of interval
        # TODO make sure it is sorted in time
        for ci in self.cost_intervals:
            start, end, cost = ci._start, ci._end, ci.cost
            self._charge_per_minute.extend([cost] * (end - start))

    def get_charge_per_minute(self):
        if self._charge_per_minute == None:
            self.setup_charge_per_minute()
        return self._charge_per_minute

    def __str__(self):
        charge_intervals_str = (
            "Charge: " + " ".join([c.__repr__() for c in self.cost_intervals]) + "\n"
        )
        return f"Cost name:{self.name} \t" + charge_intervals_str + "\n"

class Parameters(BaseModel):
    id: str
    scenario: str
    generated_datetime: Optional[datetime] = None
    energy_measurement_unit: str = "KWh"
    resources: Dict[str, Resource]
    time_granularity_in_minutes: conint(ge=1.0) = 1
    weight_vector: Optional[List[int]] = Field([0,1,0,0,0], min_length=5)
    solver: Optional[str] = 'CPSolver'
    max_solution_time_in_secs: Optional[conint(ge=20.0)] = 60

# Energy Resources
class EnergyInterval(CostInterval):
    emissions:float = 0
    # cost and emmisions levels [lb,ub,cost,emissions] It should cover the whole range [min,max] and should be sorted
    cost_levels: Optional[List[Tuple[float,float,float,float]]] = Field(default_factory=list)

    def get_emissions(self):
        return self.emissions
    
    def get_cost_emission_levels(self):
        if self.cost_levels == None:
            self._setup_levels()
        return self.cost_levels
    
    def _setup_levels(self):
        self.cost_levels = [self.min,self.max,self.cost,self.emissions]

class EnergySource(CostComponent):
    consumption_multiplier: confloat(ge=0.0) = 1000.0
    setup_time: Optional[conint(ge=0)] = 0
    shutdown_time: Optional[conint(ge=0)] = 0
    
    def __init__(self, name="Default Market", cost_intervals=[], resource_name = "Electricity", consumption_multiplier=1000.0):
        super().__init__(name=name, cost_intervals=cost_intervals, resource_name = resource_name)
        self.consumption_multiplier = consumption_multiplier

    def add_charge_interval(self, charge_interval: CostInterval):
        self.cost_intervals.append(charge_interval)

    def setup_charge_per_minute(self):
        self._charge_per_minute = []
        # TODO should create a correct formula for cost based on given cost, unit and size of interval
        # TODO make sure it is sorted in time
        for ci in self.cost_intervals:
            start, end, cost = ci._start, ci._end, ci.cost
            self._charge_per_minute.extend([cost/self.consumption_multiplier] * (end - start))

    def __str__(self):
        charge_intervals_str = (
            "Charge: " + " ".join([c.__repr__() for c in self.cost_intervals]) + "\n"
        )
        return f"Market name:{self.name} \t" + charge_intervals_str + "\n"

#Products, Material

class Product(BaseModel):
    id: int
    name: str
    volume: Optional[confloat(ge=0.0)] = 0.0
    weight: Optional[confloat(ge=0.0)] = 0.0

class ProductSequence(BaseModel):
    product_id: int
    job_ids: List[int]
    _product: Product = PrivateAttr()
    _job_sequence: List[Job] = PrivateAttr()

    def fix_after_load_from_json(self, problem:Problem):
        if self.product_id in problem.out_products:
            self._product = problem.out_products[self.product_id]
        else:
            self._product = None
        self._job_sequence = []
        for j_id in self.job_ids:
            self._job_sequence.append(problem.jobs[j_id])         

class Order(BaseModel):
    id: int
    product_id: int
    process_id: int
    quantity: conint(ge=1) = 1
    _jobs: Dict[int,Job] = PrivateAttr(default={})

    def generate_jobs(self, problem:Problem):
        id = len(problem.jobs)
        product = problem.out_products[self.product_id]
        process  = problem.processes[self.process_id]
        for k in range(self.quantity):
            j_id = id+k
            name = product.name + "_" + f'{k:d}'
            job = Job(
                id=j_id, name=name,
                process_id=self.process_id
            )
            self._jobs[j_id] = problem.jobs[j_id] = job

class ResourceUse(BaseModel):
    resource_name: str
    consumption: float

class EmissionsGenerated(BaseModel):
    emission_name: str
    generated_amount: float

class Storage(BaseModel):
    id: int
    name: str
    capacity: confloat(ge=0.0)
    min_value: confloat(ge=0.0)
    max_value: confloat(ge=0.0)
    mobile: Optional[bool] = False
    

class ResourceStorage(Storage):
    resource_name: str
    _resource: Resource = PrivateAttr(default=None)

class ProductStorage(Storage):
    eligible_products: List[str]


# Transportation
class InternalTransferItem(BaseModel):
    machine_id1: int
    machine_id2: int
    base_volume: Optional[int] = 1
    time: int
    consumptions: List[ResourceUse]

class IntraFactoryTransferItem(BaseModel):
    factory_id1: int
    factory_id2: int
    base_volume: Optional[int] = 1
    time: int
    consumptions: List[ResourceUse]

# class CostPeriod(BaseModel):
#     from_datetime: datetime
#     to_datetime: datetime
#     cost: float
#     _start: int = PrivateAttr()
#     _end: int = PrivateAttr()

#     def datetime_int(self, base: datetime):
#         self._start = datetime_to_int(base, self.from_datetime)
#         self._end = datetime_to_int(base, self.to_datetime)

# class CostComponent(BaseModel):
#     name: str
#     cost_periods: List[CostPeriod] = Field(..., min_items=1)
#     _cost_per_minute: List[float] = PrivateAttr()

#     def fix_after_load_from_json(self, base: datetime):
#         for ec in self.cost_periods:
#             ec.datetime_int(base)
#         self.setup_cost_per_minute()

#     def __init__(self, name="Default Shift", cost_periods=[]):
#         super().__init__(name=name, cost_periods=cost_periods)

#     def add_cost_period(self, cost_period: CostPeriod):
#         self.cost_periods.append(cost_period)

#     def setup_cost_per_minute(self):
#         self._cost_per_minute = []
#         # TODO should create a correct formula for cost based on given cost, unit and size of interval
#         # TODO make sure it is sorted in time
#         for ci in self.cost_periods:
#             start, end, cost = ci._start, ci._end, ci.cost
#             self._cost_per_minute.extend([cost] * (end - start))

#     def get_charge_per_minute(self):
#         if self._cost_per_minute == None:
#             self.setup_charge_per_minute()
#         return self._cost_per_minute

#     def __str__(self):
#         cost_periods_str = (
#             "Cost: " + " ".join([c.__repr__() for c in self.cost_periods]) + "\n"
#         )
#         return f"Cost name:{self.name} \t" + cost_periods_str + "\n"

# Processes, Jobs, Tasks
class Attribute(BaseModel):
    attribute_id: int
    description: Optional[str] = None
    state:Optional[Dict[str, Any]] = None

class DependencyItem(BaseModel):
    task_id1: int
    task_id2: int
    min_time: int = 0
    max_time: int = 60*24*356
    use_same_machine: bool = False
    keep_together: bool = False
    consumptions: Optional[List[ResourceUse]] = None

class ProcessingType(str, Enum):
    #When the task is performed a single machine is occupied
    SINGLE_MACHINE = 'SINGLE_MACHINE'
    #When the task is performed multiple machines are occupied in parallel
    SYNCHRONOUS_MACHINES = 'SYNCHRONOUS_MACHINES'

class AbstractOperation(BaseModel):
    name: Optional[str] = ""
    attribute_id: int
    # Bill of Material, it defines the consumed materials used by the task
    # It associates the material id with the consumed quantity
    BOM: Optional[List[Tuple[int, float]]] = Field(None, min_length=0)
    # It defines the products that are processed by the task 
    # It associates the product id with the consumed quantity
    in_products: Optional[List[Tuple[int, float]]] = Field(None, min_length=0)
    # It defines the products that are produced by the task 
    # It associates the product id with the produced quantity
    out_products: Optional[List[Tuple[int, float]]] = Field(None, min_length=0)
    product_limits: Optional[List[Tuple[int, float, float]]] = Field(None, min_length=0)
    is_optional: Optional[bool] = False
    operation_type: ProcessingType = ProcessingType.SINGLE_MACHINE

class Task(AbstractOperation):
    id: int
    job_id: int
    # TODO do we need them ???
    earliest_start_time: Optional[datetime] = None
    latest_finish_time: Optional[datetime] = None
    _est: int = PrivateAttr()
    _lft: int = PrivateAttr()
    #[(f,m)]->[opm]->(est,lft,ru)
    _eligible_machines: Dict[Tuple[int, int], Dict[int, Tuple[int,int,FixedTimeTaskResourceUseProperties]]] = PrivateAttr()
    _left_right_tasks: Tuple[List[Task], List[Task]] = PrivateAttr()
    _min_duration: int = PrivateAttr(default=-1)
    _max_duration: int = PrivateAttr(default=-1)

    def datetime_int(self, base: datetime):
        self._est = datetime_to_int(base, self.earliest_start_time)
        if self.latest_finish_time == None:
            self._lft = 1000000
        else:
            self._lft = datetime_to_int(base, self.latest_finish_time)

    def get_earliest_start_time(self) -> int:
        return self._est

    def get_latest_finish_time(self) -> int:
        return self._lft
    
    def get_min_duration(self) -> int:
        if self._min_duration == -1:
            min_dur=10E10
            for (f,m) in self._eligible_machines:
                for opm,lim in self._eligible_machines[(f,m)].items():
                    dur = lim[2].get_minimum_time()
                    if dur<min_dur:
                        min_dur=dur
            if min_dur==10E10:
                min_dur=0
            self._min_duration = min_dur
        return self._min_duration
    
    def get_max_duration(self) -> int:
        if self._max_duration == -1:
            max_dur=-1
            for (f,m) in self._eligible_machines:
                for opm,lim in self._eligible_machines[(f,m)].items():
                    dur = lim[2].get_maximum_time()
                    if dur>max_dur:
                        max_dur=dur
            self._max_duration = max_dur
        return self._max_duration

class Process(BaseModel):
    id: int
    tasks: Dict[int, AbstractOperation]  # = Field(..., min_items=1)
    task_dependencies: Optional[List[DependencyItem]] = Field(None, min_length=0)
    #Associate the task resources used to operate a task on a factory,machine,operation mode **t_id,f_id,m_id,opm => ru**
    resource_use:Dict[Tuple[int,int,int,int],Union[FixedTimeTaskResourceUseProperties,VariableTimeTaskResourceUseProperties]]

    def generate_job(self,job_id:int, name: str, product_name:str,
                     earliest_start_time: datetime,
                     problem:Problem,
                     no_penalty_finish_time: Optional[datetime] = None,
                     latest_finish_time: Optional[datetime] = None,
                     duration_limits: Optional[Tuple[int,int]] = None,
                     ):       
        job = Job(id=job_id, name=name, product_name=product_name,
                  earliest_start_time=earliest_start_time,
                  tasks=deepcopy(self.tasks),
                  task_dependencies=deepcopy(self.task_dependencies)
                  )
        if no_penalty_finish_time:
            job.no_penalty_finish_time = no_penalty_finish_time
        if latest_finish_time:
            job.latest_finish_time = latest_finish_time
        if duration_limits:
            job.duration_limits = duration_limits
        problem.jobs[job_id] = job

        for (t_id,f_id,m_id,opm),_ru in self.resource_use:
            ru = deepcopy(_ru)
            ru.job_id = job_id
            ru.task_id = t_id
            problem.factories[f_id].machines[m_id].add_machine_use(ru,opm)

class Job(BaseModel):
    id: int
    name: Optional[str] = ""
    product_name: Optional[str] = ""
    process_id: Optional[int] = -1
    tasks: Optional[Dict[int, Task]] = Field(None) # = Field(..., min_items=1)
    task_dependencies: Optional[List[DependencyItem]] = Field(None, min_length=0)
    earliest_start_time: datetime
    no_penalty_finish_time: Optional[datetime] = None
    latest_finish_time: Optional[datetime] = None
    #impose constraints on the maximum and minimum duration of the job
    duration_limits: Optional[Tuple[int,int]] = None
    #impose constraints on the maximum and minimum products that this job should produce product_id,min,max
    product_limits: Optional[List[Tuple[int,float,float]]] = None
    _est: int = PrivateAttr()
    _npft: int = PrivateAttr()
    _lft: int = PrivateAttr()
    _process: Process = PrivateAttr()
    _tasks: Dict[int, Task] = PrivateAttr()
    _graph: TaskGraph = PrivateAttr()

    def fix_after_load_from_json(self, base: datetime, problem:Problem):
        self.datetime_int(base)
        self.create_ru(problem)
        self._graph = TaskGraph(self, f"Graph_j{self.id}")
        for t, task in self.tasks.items():
            self._graph.addTask(task)
        if self.task_dependencies:
            for d in self.task_dependencies:
                task1=self.tasks[d.task_id1]
                task2=self.tasks[d.task_id2]
                self._graph.addDependency(
                    task1=task1,
                    task2=task2,
                    min_time=d.min_time,
                    max_time=d.max_time,
                    use_same_machine = d.use_same_machine,
                    keep_together = d.keep_together,
                    consumptions=d.consumptions,
                )
                if d.keep_together:
                  task1._left_right_tasks[1].append(task2)
                  task2._left_right_tasks[0].append(task1)
        # TODO we must make tighter est,lft per task given min-max duration and dependencies
        self.update_task_est_lft(problem) 

    def create_ru(self,problem:Problem):
        for t,task in self.tasks.items():
            task._eligible_machines={}
            task._left_right_tasks=([],[])
            for f,factory in problem.factories.items():
                for machine in factory.eligible_machines_for_task(self.id,t):
                    m = machine.id
                    task._eligible_machines[f,m]={}
                    opmList = factory.get_eligible_op_modes(m,self.id,t)
                    for opm,res_use in opmList:
                        task._eligible_machines[f,m][opm]=(self._est,self._lft,res_use)

    def get_Tasks(self):
        return self._graph.get_tasks()

    def datetime_int(self, base: datetime):
        self._est = datetime_to_int(base, self.earliest_start_time)
        if self.latest_finish_time:
            self._lft = datetime_to_int(base, self.latest_finish_time)
        else:
            self._lft = 100000
        if self.no_penalty_finish_time:
            self._npft = datetime_to_int(base, self.no_penalty_finish_time)
        else:
            self._npft = self._lft
        for t in self.tasks.values():
            t.datetime_int(base)
    
    def update_task_est_lft(self, problem:Problem):
        #source -1, sync -2
        self._graph.graph.nodes[-1]['est'] = self._est
        # self._graph.graph.nodes[-1]['lft'] = self._lft
        # self._graph.graph.nodes[-2]['est'] = self._est
        self._graph.graph.nodes[-2]['lft'] = self._lft
        #calculate est
        topological_order = list(nx.topological_sort(self._graph.graph))
        for ct in topological_order:
            if ct==-1:
                continue
            pred_ct = self._graph.get_prev_tasks(ct)
            est=0
            for p in pred_ct:
                if p<0:
                    pdur=0
                else:
                    pdur = self.tasks[p].get_min_duration()
                    min_setup=10E10
                    for f,factory in problem.factories.items():
                        for machine in factory.eligible_machines_for_task(self.id,p):
                            m = machine.id
                            m_tasks =  factory.tasks_that_can_be_executed_at_machine(m)
                            if (self.id,p) in m_tasks and (self.id,ct) in m_tasks:
                                setup = problem.setup_time_consumption(
                                            machine_id=m, 
                                            previous_job_id=self.id, previous_task_id=p,
                                            current_job_id=self.id, current_task_id=ct,
                                            factory_id=f)[0]
                                if setup<min_setup:
                                    min_setup=setup
                    if min_setup<10E10:
                        pdur+=min_setup
                cand_eft = self._graph.graph.nodes[p]['est']+pdur
                if cand_eft>est:
                    est=cand_eft
            if ct!=-2 and est < self.tasks[ct]._est:
                est = self.tasks[ct]._est
            self._graph.graph.nodes[ct]['est'] = est
        #calculate lft
        for ct in reversed(topological_order):
            if ct==-2:
                continue
            succ_ct = self._graph.get_next_tasks(ct)
            lft=1000000
            for s in succ_ct:
                if s<0:
                    sdur=0
                else:
                    sdur = self.tasks[s].get_min_duration()
                    max_setup=-10000000
                    for f,factory in problem.factories.items():
                        for machine in factory.eligible_machines_for_task(self.id,p):
                            m = machine.id
                            m_tasks =  factory.tasks_that_can_be_executed_at_machine(m)
                            if (self.id,s) in m_tasks and (self.id,ct) in m_tasks:
                                setup = problem.setup_time_consumption(
                                            machine_id=m, 
                                            previous_job_id=self.id, previous_task_id=ct,
                                            current_job_id=self.id, current_task_id=s,
                                            factory_id=f)[0]
                                if max_setup<setup:
                                    max_setup=setup
                    if max_setup>-10000000:
                        sdur+=max_setup
                cand_lst = self._graph.graph.nodes[s]['lft']-sdur
                if cand_lst<lft:
                    lft=cand_lst
                if ct !=-1 and lft > self.tasks[ct]._lft:
                    lft = self.tasks[ct]._lft
            self._graph.graph.nodes[ct]['lft'] = lft
        
        for t,td in self.get_Tasks():
            task=td["task"]
            task._est = self._graph.graph.nodes[t]['est']
            task._lft = self._graph.graph.nodes[t]['lft']

#Consumption / Generation 
class TaskResourceUseProperties(BaseModel):
    job_id: int
    task_id: int
    operators: Optional[int] = None

    def fix_after_load_from_json(self):
        pass
    def get_minimum_time(self):
        return 0
    def get_maximum_time(self):
        return 0
    def get_consumptions_per_min(self):
        return []
    def get_emissions_per_minute(self):
        return []

class FixedTimeTaskResourceUseProperties(TaskResourceUseProperties):
    time: int
    consumptions: List[ResourceUse]
    emissions: List[EmissionsGenerated] = None
    _consumptions_per_minute: List[ResourceUse] = PrivateAttr()
    _emissions_per_minute: List[EmissionsGenerated] = PrivateAttr()
    # time to produce product quantity [product_id,time,quantity]
    production: Optional[List[Tuple[int,float,float]]] = None

    def fix_after_load_from_json(self):
        self._consumptions_per_minute = []
        self._emissions_per_minute = []
        for ru in self.consumptions:
            self._consumptions_per_minute.append(
                ResourceUse(resource_name=ru.resource_name,
                            consumption=ru.consumption/self.time)
            )
        if self.emissions:
            for em in self.emissions:
                self._emissions_per_minute.append(
                    EmissionsGenerated(emission_name=em.emission_name,
                                       generated_amount=em.generated_amount/self.time)
                )
    
    def get_minimum_time(self):
        return self.time
    def get_maximum_time(self):
        return self.time
    def get_consumptions_per_min(self):
        return self._consumptions_per_minute
    def get_emissions_per_minute(self):
        return self._emissions_per_minute

class VariableTimeTaskResourceUseProperties(TaskResourceUseProperties):
    min_time: Optional[conint(ge=0)] = 0
    max_time: Optional[conint(ge=0)] = 0
    consumptions_per_min: List[ResourceUse]
    emissions_per_minute: Optional[List[EmissionsGenerated]] = None
    # time to produce product quantity [product_id,time,quantity]
    production: Optional[List[Tuple[int,float,float]]] = None

    def get_minimum_time(self):
        return self.min_time
    def get_maximum_time(self):
        return self.max_time
    def get_consumptions_per_min(self):
        return self.consumptions_per_min
    def get_emissions_per_minute(self):
        return self.emissions_per_minute

# Machine

class MachineCombination(BaseModel):
    id:int
    name:str
    machine_ids:List[int]
    processing_type:ProcessingType = ProcessingType.SYNCHRONOUS_MACHINES
    _machines:List[Machine]=None

    def fix_after_load_from_json(self, base: datetime, parameters:Parameters, factory:Factory):
        self._machines = []
        for m_id in self.machine_ids:
            self._machines.append(factory.machines[m_id])

class UnavailablePeriod(BaseModel):
    from_datetime: datetime
    to_datetime: datetime
    description: Optional[str] = None
    _start: int = PrivateAttr()
    _end: int = PrivateAttr()

    def datetime_int(self, base: datetime):
        self._start = datetime_to_int(base, self.from_datetime)
        self._end = datetime_to_int(base, self.to_datetime)

class SetupProperties(BaseModel):
    task_attribute1: int
    task_attribute2: int
    time: int
    consumptions: List[ResourceUse]

class MachineOperationalMode(BaseModel):
    id: int
    description: Optional[str] = None
    task_operation: Optional[
        List[Union[FixedTimeTaskResourceUseProperties,VariableTimeTaskResourceUseProperties]]
    ]  # = Field(None, min_items=0)
    _task_operation: Dict[Tuple[int, int], TaskResourceUseProperties] = PrivateAttr()

    def __init__(self, id: int, description: str = "", task_operation=[]):
        super().__init__(id=id, description=description, task_operation=task_operation)
        self._task_operation = {}

    def fix_after_load_from_json(self, base: datetime):
        if self.task_operation:
            for to in self.task_operation:
                to.fix_after_load_from_json()
                self._task_operation[(to.job_id, to.task_id)] = to

    def add(self, to: TaskResourceUseProperties):
        self._task_operation[(to.job_id, to.task_id)] = to
        self.task_operation.append(to)

    def get_resource_use(self, job_id: int, task_id: int):
        return self._task_operation.get((job_id, task_id))

    def __repr__(self):
        return (
            "Mode:"
            + str(self.id)
            + " ".join([m.__repr__() for m in self.task_operation])
            + "\n"
        )

class Stage(BaseModel):
    name: str
    machines: List[Machine]

class MachineProcessingType(str, Enum):
    SEQUENTIAL = 'SEQUENTIAL'
    CASCADING = 'CASCADING'
    BATCH = 'BATCH'
# It is used for calculating the consumption of a machine if more than one task is processed in parallel
class MachineConsumptionAggregation(str, Enum):
    NONE = 'NONE'
    ADDITIVE = 'ADDITIVE'
    PROPORTIONAL = 'PROPORTIONAL'
    MAXIMUM = 'MAXIMUM'

class Machine(BaseModel):
    id: int
    name: str
    idle_consumptions: Optional[List[ResourceUse]] = None
    unavailable: Optional[List[UnavailablePeriod]] = None
    can_process: Optional[List[int]] = None  # = Field(..., min_items=0)
    setup: Optional[List[SetupProperties]] = None  # = Field(..., min_items=0)
    operational_modes: Dict[int, MachineOperationalMode]  # = Field(..., min_items=1)
    processing_type: Optional[MachineProcessingType] = MachineProcessingType.SEQUENTIAL
    consumption_aggregation: Optional[MachineConsumptionAggregation] = MachineConsumptionAggregation.ADDITIVE
    capacity: Optional[conint(ge=1)] = 1
    stage: Optional[str] = "NO_STAGE"
    _eligible_tasks: Dict[
        Tuple[int, int], List[Tuple[int, TaskResourceUseProperties]]
    ] = PrivateAttr()
    _setup: Dict[Tuple[int, int], SetupProperties] = PrivateAttr()
    _internal_id: int = PrivateAttr()

    def __init__(
        self,
        id: int,
        name: str = "",
        operational_modes: dict = {},
        idle_consumption=0,
        setup=[],
        unavailable: list = None,
        capacity:int = 1,
        can_process: list = [],
        processing_type: MachineProcessingType = MachineProcessingType.SEQUENTIAL,
        consumption_aggregation: MachineConsumptionAggregation = MachineConsumptionAggregation.ADDITIVE,
        stage = "NO_STAGE"
    ):
        super().__init__(
            id=id,
            name=name,
            idle_consumption=idle_consumption,
            setup=setup,
            unavailable=unavailable,
            operational_modes=operational_modes,
            capacity = capacity,
            can_process = can_process,
            processing_type = processing_type,
            consumption_aggregation = consumption_aggregation,
            stage = stage
        )
        self._eligible_tasks = {}
        self._setup = {}

    def fix_after_load_from_json(self, base: datetime):
        if self.unavailable:
            for u in self.unavailable:
                u.datetime_int(base)
        for om, mom in self.operational_modes.items():
            mom.fix_after_load_from_json(base)
            for job_id, task_id in mom._task_operation:
                if not (job_id, task_id) in self._eligible_tasks:
                    self._eligible_tasks[(job_id, task_id)] = []
                self._eligible_tasks[(job_id, task_id)].append(
                    (om, mom._task_operation[(job_id, task_id)])
                )
        if self.setup:
            for s in self.setup:
                self._setup[(s.task_attribute1, s.task_attribute2)] = s
                if s.task_attribute1 not in self.can_process:
                    self.can_process.append(s.task_attribute1)
                if s.task_attribute2 not in self.can_process:
                    self.can_process.append(s.task_attribute2)

    def add_machine_unavailability(self, ump: UnavailablePeriod):
        if self.unavailable:
            self.unavailable.append(ump)
        else:
            self.unavailable = [ump]

    def add_machine_use(self, usage: Union[FixedTimeTaskResourceUseProperties,VariableTimeTaskResourceUseProperties], op_mode: int = 0):
        if op_mode not in self.operational_modes:
            self.operational_modes[op_mode] = MachineOperationalMode(id=op_mode)
        self.operational_modes[op_mode].add(usage)
        if not (usage.job_id, usage.task_id) in self._eligible_tasks:
            self._eligible_tasks[(usage.job_id, usage.task_id)] = []
        self._eligible_tasks[(usage.job_id, usage.task_id)].append(
                    (op_mode, usage)
                )

    def add_machine_setup(self, setup: SetupProperties):
        self.setup.append(setup)
        self._setup[(setup.task_attribute1, setup.task_attribute2)] = setup
        if setup.task_attribute1 not in self.can_process:
            self.can_process.append(setup.task_attribute1)
        if setup.task_attribute2 not in self.can_process:
            self.can_process.append(setup.task_attribute2)

    def eligible_for_task(self, job_id, task_id):
        return (job_id, task_id) in self._eligible_tasks

    def setup_time_consumption_from_attr1_to_attr2(self, attr1, attr2):
        if (attr1, attr2) not in self._setup:
            logging.info(f"{(attr1, attr2)}")
        value = self._setup.get((attr1, attr2))
        if not value:
            return 0, 0.0
        else:
            return value.time, value.consumptions[0].consumption # TODO: temp fix, we should support multiple resources
    
    def setup_time_consumption_from_attr1_to_attr2(self, attr1, attr2, resource_name="Electricity"):
        if (attr1, attr2) not in self._setup:
            logging.info(f"Machine {self.id} failed to access setup{(attr1, attr2)}")
        value = self._setup.get((attr1, attr2))
        if not value:
            return 0, 0.0
        else:
            for ru in value.consumptions:
                if ru.resource_name == resource_name:
                    return value.time, ru.consumption
            return 0, 0.0

    def __repr__(self):
        op_mode_str = (
            "Operational modes:\n"
            + "\t".join([m.__repr__() for m in self.operational_modes.values()])
            + "\n"
        )
        return (
            f"Machine(id={self.id}, name={self.name}, idle_consumption={self.idle_consumption}, setup_times={self.setup}), unavailable={self.unavailable})\n"
            + op_mode_str
        )

# Human resources 

class Operator(BaseModel):
    id: int
    name: str
    unavailable: Optional[List[UnavailablePeriod]] = None
    machines: List[int] = Field(..., min_length=1)

    def fix_after_load_from_json(self, base: datetime):
        for u in self.unavailable:
            u.datetime_int(base)

# Factory

class Factory(BaseModel):
    id: int
    name: str
    energy_source_names: List[str] = Field(..., min_length=1)
    machines: Dict[int, Machine]  # = Field(..., min_items=1)
    operators: Optional[Dict[int, Operator]]  = None # = Field(None, min_length=0)
    m2m_transportation: Optional[List[InternalTransferItem]] = None
    cost_components: Optional[Dict[str, CostComponent]] = None #Field(...)
    machine_combinations: Optional[Dict[int, MachineCombination]] = None
    storages: Optional[Dict[int, ProductStorage]] = None
    _energy_sources: Dict[str, EnergySource] = PrivateAttr()

    def fix_after_load_from_json(
        self, base: datetime, energy_sources: Dict[str, EnergySource], parameters: Parameters
    ):
        for esn in self.energy_source_names:
            if esn not in self._energy_sources:
                self._energy_sources[esn] = energy_sources.get(esn)
        self.create_energy_cost_component(parameters)
        for m in self.machines.values():
            m.fix_after_load_from_json(base)
        if self.operators:
            for o in self.operators.values():
                o.fix_after_load_from_json(base)
        if self.m2m_transportation:
            for t in self.m2m_transportation:
                t.fix_after_load_from_json(base)
        if self.cost_components:
            for cc in self.cost_components.values():
                cc.fix_after_load_from_json(base,parameters)
        if self.machine_combinations:
            for mc in self.machine_combinations.values():
                mc.fix_after_load_from_json(base,parameters,self)
    
    def change_energy_sources(
        self, base: datetime, energy_sources: Dict[str, EnergySource], parameters: Parameters
    ):
        self._energy_sources = {}
        for esn in self.energy_source_names:
            if esn not in self._energy_sources:
                self._energy_sources[esn] = energy_sources.get(esn)
        self.create_energy_cost_component(parameters)
        if self.cost_components:
            for cc in self.cost_components.values():
                cc.fix_after_load_from_json(base,parameters)    

    def __init__(
        self,
        id: int,
        name: str = "Default Factory",
        energy_market: EnergySource = None,
        energy_source_names: list = None,
        machines: dict = {},
        cost_components: dict = {},
        storages: list ={}
    ):
        if energy_source_names:
            super().__init__(
                id=id,
                name=name,
                energy_source_names=energy_source_names,
                machines=machines,
                cost_components=cost_components,
                storages = storages
            )
        elif energy_market:
            super().__init__(
                id=id,
                name=name,
                energy_source_names=[energy_market.name],
                machines=machines,
                cost_components=cost_components,
                storages = storages
            )
        self._energy_sources = {}
        if energy_market:
            self._energy_sources[energy_market.name] = energy_market

    def add_machine(self, machine: Machine):
        self.machines[machine.id] = machine

    def get_machines(self):
        return self.machines
    
    def add_storage(self, storage: Storage):
        if self.storages is None:
            self.storages = {}
        self.storages[storage.id] = storage

    def eligible_machines_for_task(self, job_id: int, task_id: int) -> list[Machine]:
        ret = []
        for m in self.machines.values():
            if m.eligible_for_task(job_id, task_id):
                ret.append(m)
        return ret

    def tasks_that_can_be_executed_at_machine(self, machine_id: int):
        return self.machines[machine_id]._eligible_tasks.keys()

    def get_eligible_op_modes(self, machine_id: int, job_id: int, task_id: int):
        return self.machines[machine_id]._eligible_tasks.get((job_id, task_id))

    def add_cost_component(self, cc: CostComponent):
        self.cost_components[cc.name] = cc

    def get_charge_per_minute(self):
        # TODO support multiple energy market combined cost and cost ranges
        for em in self._energy_sources.values():
            return em.get_charge_per_minute()

    def setup_time_consumption(self, machine_id: int, attr1: int, attr2: int):
        return self.machines[machine_id].setup_time_consumption_from_attr1_to_attr2(
            attr1, attr2
        )

    def get_cost_for_task_machine_at_time(
        self, task: Task, machine_id: int, op_mode: int, time: int
    ):
        est = task._est
        lft = task._lft
        machine = self.machines[machine_id]
        ru = machine.operational_modes[op_mode].get_resource_use(task.job_id, task.id)
        execution_time = ru.get_minimum_time()
        consumption = ru.get_consumptions_per_min()[0].consumption*execution_time # TODO temp fix, we should support multiple resources
        if time < est:  # or time > lft - execution_time:
            return int(10e10)
        tmp = self.get_charge_per_minute()
        cost = 0
        for charge_per_minute in tmp[time : time + execution_time]:
            cost += charge_per_minute * consumption / execution_time
        return cost
    
    def create_energy_cost_component(self, parameters: Parameters):
        # If we have a single energy market, we can use it directly
        if len(self._energy_sources)==1:
            cost_intervals = deepcopy(list(self._energy_sources.values())[0].cost_intervals)
            multiplier = list(self._energy_sources.values())[0].consumption_multiplier
            for ci in cost_intervals:
                ci.cost /= multiplier 
            cc = CostComponent("Electricity Cost",cost_intervals=cost_intervals)
            self.cost_components["Electricity"] = cc
        else:
            # If we have multiple energy markets, we need to create a combined cost component per energy resource
            res_trees = {}
            for res_name,resource in parameters.resources.items():
                for es in self._energy_sources.values():
                    if es.resource_name not in res_trees:
                        res_trees[es.resource_name] = IntervalTree()
                    for interval in es.cost_intervals:
                        res_trees[es.resource_name][interval._start:interval._end] = interval

                for res_name,tree in res_trees.items():
                    cost_intervals = []
                    for interval in tree:
                        pass
                    cc = CostComponent(res_name+" Cost",cost_intervals=cost_intervals)
                    self.cost_components[res_name+" Cost"] = cc

    def __str__(self):
        machines_str = (
            "Machines: "
            + " ".join([m.__repr__() for m in self.machines.values()])
            + "\n"
        )

        return (
            f"Factory #{self.id} name:{self.name}, energy sources:{self.energy_source_names}\n"
            + machines_str
            + "\n"
        )

# Problem 

class Problem(BaseModel):
    start_time: datetime
    finish_time: datetime
    task_attributes: Dict[int, Attribute]  # = Field(..., min_items=1)
    jobs: Dict[int, Job]  # = Field(..., min_items=1)
    factories: Dict[int, Factory]  # = Field(..., min_items=1)
    in_products:Optional[Dict[int,Product]] = None
    out_products:Optional[Dict[int,Product]] = None
    productSequences:Optional[Dict[int,ProductSequence]] = None
    orders:Optional[List[Order]] = None
    _horizon_finish: int = PrivateAttr()

    def datetime_int(self):
        self._horizon_finish = datetime_to_int(self.start_time, self.finish_time)

    def fix_after_load_from_json(self, energy_sources: Dict[str, EnergySource], parameters: Parameters):
        self.datetime_int()
        if self.orders:
            for o in self.orders:
                o.generate_jobs(self)
        for f in self.factories.values():
            f.fix_after_load_from_json(self.start_time, energy_sources, parameters)
        #Give a unique integer number for each machine
        m_counter = 0
        for f in self.factories.values():
            for m in f.machines.values():
                m._internal_id = m_counter
                m_counter += 1

        for j in self.jobs.values():
            j.fix_after_load_from_json(self.start_time,self)
        if self.productSequences:
            for ps in self.productSequences.values():
                ps.fix_after_load_from_json(self)
    
    def change_energy_sources(self, energy_sources: Dict[str, EnergySource], parameters: Parameters):
        for f in self.factories.values():
            f.change_energy_sources(self.start_time, energy_sources, parameters)

    # def __init__(self):
    #     super().__init__(id=1,start_time=datetime.now(),finish_time=datetime.now(),job_attributes={},jobs={},factories={})
    #     self.factories=dict[int,Factory]()

    def add_job(self, job: Job):
        self.jobs[job.id] = job

    def add_factory(self, factory: Factory):
        self.factories[factory.id] = factory

    def eligible_machines_for_task(self, job_id: int, task_id: int)->List[int,int]:
        factory_machines = []
        for factory_id in self.factories:
            fm = self.factories[factory_id].eligible_machines_for_task(job_id, task_id)
            for m in fm:
                factory_machines.append((m.id, factory_id))
        return factory_machines

    def job_tasks_that_can_be_executed_at_machine(
        self, machine_id: int, factory_id: int
    ):
        return self.factories[factory_id].jobs_that_can_be_executed_at_machine(
            machine_id
        )

    def setup_time_consumption(
        self,
        machine_id,
        previous_job_id,
        previous_task_id,
        current_job_id,
        current_task_id,
        factory_id: int,
    ):
        if factory_id in self.factories:
            attr1 = self.jobs[previous_job_id].tasks[previous_task_id].attribute_id
            attr2 = self.jobs[current_job_id].tasks[current_task_id].attribute_id
            return self.factories[factory_id].setup_time_consumption(
                machine_id, attr1, attr2
            )
        else:
            raise Exception("Error in Factory id")

    def setup_charge_per_minute(self):
        for f in self.factories.values():
            f.setup_charge_per_minute()

    def get_charge_per_minute(self, factory_id: int):
        return self.factories[factory_id].get_charge_per_minute()

    def get_cost_for_job_task_machine_at_time(
        self, job_id, task_id, machine_id, op_mode, time, factory_id
    ):
        task = self.jobs[job_id].tasks[task_id]
        return self.factories[factory_id].get_cost_for_task_machine_at_time(
            task, machine_id, op_mode, time
        )

    def __str__(self):
        job_attrs_str = (
            "Job Attributes: "
            + " ".join([str(a) for a in self.task_attributes.keys()])
            + "\n"
        )
        jobs_str = (
            "Jobs: " + " ".join([j.__repr__() for j in self.jobs.values()]) + "\n"
        )
        factories_str = (
            "####Factories:#####\n\t*"
            + "\t*".join([f.__str__() for f in self.factories.values()])
            + "\n"
        )
        # energy_markets_str = (
        #     "####Energy Markets:####\n\t*" + "\t*".join([m.__str__() for m in self.energy_markets.values()]) + "\n"
        # )

        return (
            # f"id:{self.id}, scenario:{self.scenario}, generated at:{self.generated_datetime}\n"
            # +
            f"start time:{self.start_time}, finish_time:{self.finish_time}, horizon:{self._horizon_finish}\n"  # , granularity:{self.time_granularity_in_minutes} minutes\n"
            + job_attrs_str
            + jobs_str
            + factories_str
            # + energy_markets_str
            + "\n"
        )

# Solution 

class ScheduleItem(BaseModel):
    job_id: int
    task_id: int
    factory_id: int
    machine_id: int
    machine_mode: int = 0
    start_time: datetime
    finish_time: datetime
    total_consumption: float
    consumptions: List[ResourceUse]
    cost: float
    
    def __str__(self):
        return f"[{self.job_id}:{self.task_id}]=>[{self.factory_id}:{self.machine_id}:{self.machine_mode}] at [{self.start_time},{self.finish_time}] using {self.total_consumption} and costing {self.cost}" 

class StatisticsType(str, Enum):
    MACHINE_STATS = 'MACHINE_STATS'
    FACTORY_STATS = 'FACTORY_STATS'
    PROBLEM_STATS = 'PROBLEM_STATS'
    COST_STATS = 'COST_STATS'
    PRODUCT_STATS = 'PRODUCT_STATS'

class StatisticsItem(BaseModel):
    type : StatisticsType
    id: List[int]
    start_time: datetime
    finish_time: datetime
    total_consumption: float
    consumptions: List[ResourceUse]
    cost: float
    
    def __str__(self):
        return f"[{self.type}:{self.id}]=>[{self.start_time},{self.finish_time}]=>consumes={self.total_consumption},costing={self.cost}"


class Solution(BaseModel):
    id: int
    problem_id: str
    generated_date: datetime
    schedule: List[ScheduleItem]
    statistics: Optional[List[StatisticsItem]] = None
    _schedule: Dict[Tuple[int, int], ScheduleItem] = PrivateAttr()
    _problem: Problem = PrivateAttr()

    def __init__(self,
                id: int,
                problem_id: str,
                generated_date: datetime,
                schedule: List[ScheduleItem],
                statistics: List[StatisticsItem] = None
            ):
        super().__init__(id=id,problem_id=problem_id,generated_date=generated_date,schedule=schedule, statistics=statistics)
        self._schedule = {}

    def fix_after_load_from_json(self, problem:Problem):
        self._problem = problem
        for sc in self.schedule:
            self._schedule[sc.job_id,sc.task_id] = sc

    def schedule_task_to_machine(
        self,
        job_id: int,
        task_id: int,
        machine_mode: int,
        machine_id: int,
        factory_id: int,     
        start_time: int,
        duration: int,
        cost: float,
        total_consumption: float,
        consumptions=[]
    ):
        si =  ScheduleItem(
            job_id=job_id,
            task_id=task_id,
            machine_mode=machine_mode,
            machine_id=machine_id,
            factory_id=factory_id,            
            start_time=int_to_datetime(self._problem.start_time,start_time),
            finish_time=int_to_datetime(self._problem.start_time,start_time + duration),
            duration=duration,
            total_consumption = total_consumption,
            consumptions=consumptions,
            cost=cost,
        )
        self._schedule[(job_id, task_id)] = si
        self.schedule.append(si)

    
    def validate_solution(self):
        # check that all tasks are scheduled
        is_solution_valid = True
        for job_id,job in self._problem.jobs.items():
            for t_id,task in job.tasks.items():
                if (job_id,t_id) not in self._schedule and not task.is_optional:
                    logging.error(f"Task ({job_id},{t_id}) is not scheduled")
                    is_solution_valid = False

        # check that all tasks are scheduled at [EST,LFT]
        for (job_id,task_id),sc in self._schedule.items():
            a_job = self._problem.jobs[job_id]
            a_task = a_job.tasks[task_id]
            if sc.start_time < a_job.earliest_start_time:
                logging.error(f"Job {job_id} is scheduled before earliest start time")
                is_solution_valid = False
            elif sc.finish_time > a_job.latest_finish_time:
                logging.error(f"Job {job_id} is scheduled after latest finish time")
                is_solution_valid = False
        
        # TODO check that all tasks dependencies are satisfied

        # check that no machine is used when not available
        for f,factory in self._problem.factories.items():
            for m,machine in factory.get_machines().items():
                if machine.unavailable:
                    tl = factory.tasks_that_can_be_executed_at_machine(m)
                    for djob in machine.unavailable:
                        for j_id,t_id in tl:
                            if (j_id,t_id) in self._schedule:
                                if self._schedule[j_id,t_id].factory_id == f and self._schedule[j_id,t_id].machine_id==m:
                                    if max(self._schedule[j_id,t_id].start_time, djob.from_datetime) < min(self._schedule[j_id,t_id].finish_time, djob.to_datetime):
                                        logging.error(
                                            f"Machine ({f},{m}) is used when not available {djob.from_datetime}-{djob.to_datetime} by task ({j_id},{t_id})"
                                        )
                                        is_solution_valid = False

        # check that no jobs overlap at the same machine
        for f,factory in self._problem.factories.items():
            for m,machine in factory.get_machines().items():
                sc_of_machine = []
                for sc in self._schedule.values():
                    if sc.factory_id==f and sc.machine_id == m:
                        sc_of_machine.append(sc)
                # logging.info(sc_of_machine)
                sc_of_machine.sort(key=lambda sc: sc.start_time)
                if len(sc_of_machine)>machine.capacity:
                    for i in range(len(sc_of_machine) - machine.capacity):
                        if sc_of_machine[i].finish_time > sc_of_machine[i + machine.capacity].start_time:
                            logging.error(
                                f"Machine (On machine {f},{m}) task {sc_of_machine[i]} overlap (start,end) with task {sc_of_machine[i+machine.capacity]} exceeding machine capacity"
                            )
                            is_solution_valid = False


        # check that machine setup times are satisfied
        for f,factory in self._problem.factories.items():
            for m,machine in factory.get_machines().items():
                sc_of_machine = []
                for sc in self._schedule.values():
                    if sc.factory_id==f and sc.machine_id == m:
                        sc_of_machine.append(sc)
                # logging.info(sc_of_machine)
                sc_of_machine.sort(key=lambda sc: sc.start_time)
                for i in range(len(sc_of_machine) - 1):
                    job_id1 = sc_of_machine[i].job_id
                    task_id1 = sc_of_machine[i].task_id
                    job_id2 = sc_of_machine[i + 1].job_id
                    task_id2 = sc_of_machine[i+1].task_id
                    time = self._problem.setup_time_consumption(
                            machine_id=m, 
                            previous_job_id=job_id1, previous_task_id=task_id1,
                            current_job_id=job_id2, current_task_id=task_id2,
                            factory_id=f)[0]
                    if time > 0:
                        if ((sc_of_machine[i + 1].start_time - sc_of_machine[i].finish_time).total_seconds() / 60 ) < time:
                            logging.error(
                                f"setup time violation {sc_of_machine[i]}, {sc_of_machine[i+1]}"
                            )
                            is_solution_valid = False

        return is_solution_valid

    def cost(self):
        if self._problem == None:
            return -1
        consumption_cost = 0
        setups_cost = 0
        start = self._problem.start_time
        for sc in self.schedule:
            t_int = datetime_to_int(start,sc.start_time)
            ecost = self._problem.get_cost_for_job_task_machine_at_time(
                sc.job_id, sc.task_id, sc.machine_id, sc.machine_mode, t_int, sc.factory_id
            )
            consumption_cost += ecost
            # logging.info(f"job{job_id} ecost={ecost:.2f}")

        # for f_id, factory in self.problem.factories.items():
        #     tmp = factory.get_charge_per_minute()
        #     setups_cost = 0
        #     for machine_id in factory.machines:
        #         sc_of_machine = []
        #         for sc in self.schedule.values():
        #             if sc.factory_id == f_id and sc.machine_id == machine_id:
        #                 sc_of_machine.append(sc)
        #         sc_of_machine.sort(key=lambda sc: sc.start_time)
        #         for i in range(len(sc_of_machine) - 1):
        #             job_id1 = sc_of_machine[i].job_id
        #             job_id2 = sc_of_machine[i + 1].job_id
        #             time, consumption = self.problem.setup_time_consumption(
        #                 machine_id, job_id1, job_id2, f_id
        #             )
        #             setup_cost = 0
        #             job_id2_start_time = self.schedule[job_id2].start_time
        #             for charge_per_minute in tmp[
        #                 job_id2_start_time - time : job_id2_start_time
        #             ]:
        #                 setup_cost += charge_per_minute * consumption / time
        #             logging.info(
        #                 f"job{job_id1} -> job{job_id2} setup_cost={setup_cost:.2f}"
        #             )
        #             setups_cost += setup_cost
        return consumption_cost, setups_cost

    def finish_time(self):
        return max([sc.finish_time for sc in self._schedule.values()])
    
    def start_time(self):
        return min([sc.start_time for sc in self._schedule.values()])
    
    def horizon_finish(self):
        start = self.start_time()
        end = self.finish_time()
        return datetime_to_int(start,end)

    def get_consumption(self):
        consumption = 0
        for job_id, sc in self._schedule.items():
            consumption += sc.consumption
        return consumption
    
    def generate_statistics(self, granularity_in_min=15):
        def daterange(start_date:datetime, end_date:datetime, step_in_min:int=1):
            diff = int((end_date - start_date).seconds/60)
            for n in range(0,diff,step_in_min):
                yield start_date + timedelta(minutes=n)
        
        if not self.statistics:
            self.statistics = []

        sc_of_factory_machine = {}
        
        for f,factory in self._problem.factories.items():
            for m in factory.get_machines():
                sc_of_factory_machine[f,m] = []
        for sc in self._schedule.values():
            sc_of_factory_machine[(sc.factory_id,sc.machine_id)].append(sc)
        for f,factory in self._problem.factories.items():
            for m in factory.get_machines():
                sc_of_factory_machine[f,m].sort(key=lambda sc: sc.start_time)
        
        for f,factory in self._problem.factories.items():    
            #Generate statistics for all periods per machine and for the factory
            #Assume the energy and cost are linear and overlapping time is used to calculate the percentage
            cc = -1
            cp = 0
            for start_time in daterange(self._problem.start_time,self._problem.finish_time,granularity_in_min):
                end_time = start_time + timedelta(minutes=granularity_in_min)
                total_consumption_factory = 0
                total_cost_factory = 0           
                for m,machine in factory.get_machines().items():
                    if len(sc_of_factory_machine[f,m]) == 0:
                        continue
                    total_consumption_machine = 0
                    total_cost_machine = 0
                    consumptions_machine = deepcopy(sc_of_factory_machine[f,m][0].consumptions)
                    for cons in consumptions_machine:
                        cons.consumption = 0.0
                    if total_consumption_factory == 0 :
                        consumptions_factory = deepcopy(sc_of_factory_machine[f,m][0].consumptions)
                        for cons in consumptions_factory:
                            cons.consumption = 0.0
                    for sc in sc_of_factory_machine[f,m]:
                        # No overlap between the stat timewindow and the task
                        if sc.finish_time < start_time or sc.start_time > end_time:
                            continue
                        overlap = max( 0, (min(sc.finish_time,end_time) - max(sc.start_time,start_time)).seconds/60)
                        duration = (sc.finish_time-sc.start_time).seconds/60
                        percentage = overlap/duration
                        total_consumption_machine += percentage*sc.total_consumption
                        total_cost_machine += percentage*sc.cost
                        for m_cons,cons in zip(consumptions_machine,sc.consumptions):
                            m_cons.consumption += cons.consumption*percentage
                    self.statistics.append(StatisticsItem(
                            type=StatisticsType.MACHINE_STATS,
                            id=[f,cc,cp,m],
                            start_time=start_time,
                            finish_time=end_time,
                            total_consumption=total_consumption_machine,
                            consumptions=consumptions_machine,
                            cost=total_cost_machine         
                        )
                    )

                    total_consumption_factory += total_consumption_machine
                    total_cost_factory += total_cost_machine
                    for f_cons,cons in zip(consumptions_factory,consumptions_machine):
                        f_cons.consumption += cons.consumption
                self.statistics.append(StatisticsItem(
                            type=StatisticsType.FACTORY_STATS,
                            id=[f,cc,cp],
                            start_time=start_time,
                            finish_time=end_time,
                            total_consumption=total_consumption_factory,
                            consumptions=consumptions_factory,
                            cost=total_cost_factory         
                        )
                    )
                cp+=1


    def __str__(self):
        out = "Schedule: "
        for (job_id,task_id) in self._schedule:
            out += f"\n{self._schedule[(job_id,task_id)]}"
        return out

class SolverMode(str, Enum):
    GENERATE_BEST = 'GENERATE_BEST'
    GENERATE_FIRST = 'GENERATE_FIRST'
    INCREMENTAL = 'INCREMENTAL'
    ONLY_STATS = 'ONLY_STATS'

class Scenario(BaseModel):
    parameters: Parameters
    problem: Problem
    energy_sources: Dict[str, EnergySource]  # = Field(..., min_items=1)
    initial_solution: Optional[Solution] = None
    solver_mode : SolverMode = SolverMode.GENERATE_BEST

    def fix_after_load_from_json(self):
        for es in self.energy_sources.values():
            es.fix_after_load_from_json(self.problem.start_time, self.parameters)
        self.problem.fix_after_load_from_json(self.energy_sources, self.parameters)
    
    def change_energy_sources(self, new_energy_sources:Dict[str, EnergySource]):
        self.energy_sources = new_energy_sources
        for es in self.energy_sources.values():
            es.fix_after_load_from_json(self.problem.start_time, self.parameters)
        self.problem.change_energy_sources(self.energy_sources, self.parameters)       

class TaskGraph:
    def __init__(self, job: Job, name=""):
        self.job = job
        self.name = name
        self.graph = nx.DiGraph()
        self.start_task = Task(
            id=-1,
            job_id=self.job.id,
            attribute_id=-1,
            earliest_start_time=self.job.earliest_start_time,
            latest_finish_time=self.job.latest_finish_time,
        )
        self.graph.add_node(self.start_task.id, task=self.start_task)
        self.sync_task = Task(
            id=-2,
            job_id=self.job.id,
            attribute_id=-1,
            earliest_start_time=self.job.earliest_start_time,
            latest_finish_time=self.job.latest_finish_time,
        )
        self.graph.add_node(self.sync_task.id, task=self.sync_task)

    def get_tasks(self):
        return self.graph.nodes.items()

    def addTask(self, task: Task):
        self.graph.add_node(task.id, task=task)
        self.graph.add_edge(self.start_task.id, task.id)
        self.graph.add_edge(task.id, self.sync_task.id)

    def addDependency(self, task1: Task, task2: Task, **params_dict):
        self.graph.add_edge(task1.id, task2.id, **params_dict)
        if self.graph.has_edge(self.start_task.id, task2.id):
            self.graph.remove_edge(self.start_task.id, task2.id)
        if self.graph.has_edge(task1.id, self.sync_task.id):
            self.graph.remove_edge(task1.id, self.sync_task.id)

    def get_prev_tasks(self, t_id: int):
        return self.graph.predecessors(t_id)
    
    def get_descendant_tasks(self, t_id: int):
        return nx.descendants(self.graph, t_id)

    def get_next_tasks(self, t_id: int):
        return self.graph.successors(t_id)
    
    def get_dependency(self, s_id:int, e_id:int):
        return self.graph.edges[s_id,e_id]


if __name__ == "__main__":
    with open("alefrag/schemas/V5/scenario.json", "w", encoding="utf-8") as f:
        txt = Scenario.schema_json(indent=2)
        f.write(txt)
