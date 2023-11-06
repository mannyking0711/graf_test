import random
import logging

from datetime import datetime

from .scenario import CostInterval, CostComponent, SetupProperties, UnavailablePeriod, DependencyItem, Attribute, ResourceUse, Resource, ResourceType
from .scenario import Scenario, Parameters, Problem, Factory, EnergySource, MachineOperationalMode 
from .scenario import Machine, Job, Task, CostInterval, FixedTimeTaskResourceUseProperties
from .scenario import ProductStorage
from .scenario import datetime_to_int, int_to_datetime

class ProblemGeneratorV5():
    def __init__(self):
        self.parameters_specs = False
        self.core_specs = False
        self.attributes_specs = False
        self.machines_specs = False
        self.jobs_specs = False
        self.energy_source_specs = False
        self.energy_sources_specs = False
        self.factory_specs = False
    
    def find_unavailable_intervals(unavailable_intervals, horizon, percentage):
        intervals = []
        sum_ = 0
        while True:
            size = random.randint(
                int(horizon * percentage / unavailable_intervals) // 2,
                int(horizon * percentage / unavailable_intervals) * 2,
            )
            sum_ += size
            if sum_ > horizon:
                break
            intervals.append(size)

        # logging.info(intervals)
        selected_intervals = []
        sum_ = 0
        idxs = list(range(len(intervals)))
        random.shuffle(idxs)
        for idx in idxs:
            sum_ += intervals[idx]
            selected_intervals.append(idx)
            if sum_ > horizon * percentage:
                break

        start_time = 0
        finish_time = 0
        r = []
        for idx in range(len(intervals)):
            finish_time += intervals[idx]
            if idx in selected_intervals:
                r.append((start_time, finish_time))
            start_time = finish_time

        logging.info(r)
        return r
    
    def set_parameters_specs(
        self, id, scenario, energy_measurement_unit="kWh", time_granularity_in_minutes=1
    ):
        self.parameters = Parameters(
            id=id, 
            scenario=scenario,
            generated_datetime=datetime.now(),
            resources={
                "Electricity": Resource(
                    name="Electricity",
                    measurement_unit="KWh",
                    measurement_type=ResourceType.CONTINUES
                )
            },
            energy_measurement_unit = energy_measurement_unit,
            time_granularity_in_minutes= time_granularity_in_minutes
        )
        self.parameters_specs = True
    
    def set_optional_parameters_specs(
        self, weight_vector=[10,1,1000,10,1], solver="CPSolver", max_solution_time_in_secs=60
    ):
        self.parameters.weight_vector = weight_vector
        self.parameters.solver = solver
        self.parameters.max_solution_time_in_secs = max_solution_time_in_secs
        self.optional_parameters_specs = True

    def set_core_specs(self, start_time:datetime, finish_time:datetime):
        self.start_time = start_time
        self.finish_time = finish_time
        self.start_time_int = 0
        self.finish_time_int = datetime_to_int(start_time,finish_time)
        self.horizon_finish = self.finish_time_int - self.start_time_int
        self.core_specs = True

    def set_attribute_specs(self, number_of_attributes=2):
        self.number_of_attributes = number_of_attributes
        self.attributes = {}
        for attrib_id in range(self.number_of_attributes):
            self.attributes[
                attrib_id
            ] = Attribute(
                attribute_id=attrib_id,
                description= f"Description of task attribute {attrib_id}"
            )
        self.attributes_specs = True
    
    def generate_energy_markets(
        self,
        number_of_energy_markets=1,
        cost_per_measurement_unit_low=55.0, 
        cost_per_measurement_unit_hi=325.0,
        cost_per_measurement_unit_multiplier_low=1.0,
        cost_per_measurement_unit_multiplier_hi=2.0,
        co2_per_measurement_unit_multiplier_low=1.0,
        co2_per_measurement_unit_multiplier_hi=3.0
    ):  
        base_cost_per_measurement_unit_low=cost_per_measurement_unit_low
        base_cost_per_measurement_unit_hi=cost_per_measurement_unit_hi
        base_co2_per_measurement_unit_low=1.0
        base_co2_per_measurement_unit_hi=3.0
        for m_id in range(number_of_energy_markets):
            m_name = f"Market {m_id}"
            #generate costs
            charge_intervals = []
            low_cost = random.uniform( cost_per_measurement_unit_multiplier_low,cost_per_measurement_unit_multiplier_hi)*base_cost_per_measurement_unit_low
            high_cost = random.uniform( cost_per_measurement_unit_multiplier_low,cost_per_measurement_unit_multiplier_hi)*base_cost_per_measurement_unit_hi
            low_co2 = random.uniform( co2_per_measurement_unit_multiplier_low,co2_per_measurement_unit_multiplier_hi)*base_co2_per_measurement_unit_low
            high_co2 = random.uniform( co2_per_measurement_unit_multiplier_low,co2_per_measurement_unit_multiplier_hi)*base_co2_per_measurement_unit_hi
            for t in range(self.start_time_int, self.finish_time_int+60, 60):
                cost_per_kwh = round(
                    random.uniform(low_cost, high_cost),
                    2,
                )
                co2_per_kwh = round(
                    random.uniform(low_co2, high_co2),
                    2,
                )
                start = int_to_datetime(self.start_time,t)
                end = int_to_datetime(self.start_time,t+60)
                ci = CostInterval(from_datetime=start, to_datetime=end, cost = cost_per_kwh, emissions=co2_per_kwh)
                charge_intervals.append(ci)
            a_market = EnergySource(name=m_name, cost_intervals = charge_intervals)
            self.energy_markets[m_name]=a_market
    
    def generate_local_sources(self,
        number_of_energy_sources=1,
        cost_per_measurement_unit_low=25.0,
        cost_per_measurement_unit_hi=85.0,
        cost_per_measurement_unit_multiplier_low=1.0,
        cost_per_measurement_unit_multiplier_hi=2.0,
        co2_per_measurement_unit_multiplier_low=1.0,
        co2_per_measurement_unit_multiplier_hi=3.0                   
    ):
        # The average amount of co2 emissions in Kg per 1 MWh [lifetime,production]
        co2_production_per_type ={
            "Coal": (820,770),
            "Biomass - cofiring": (740,705),
            "Natural Gas": (490,407),
            "Biomass": (230,190),
            "Solar PV - utility": (48,0),
            "Solar PV - roof": (41,0),
            "Geothermal": (38,0),
            "Solar concentrated": (27,0),
            "Hydropower": (27,0),
            "Neuclear": (12,0),
            "Wind offshore": (12,0),
            "Wind onshore": (12,0),
        }
        production_pattern_per_day ={
            "Solar PV - utility": [0,0,0,0,0,0,0.52,2.72,6.47,7.14,9.11,12.32,14.76,13.89,12.62,10.66,8.44,5.21,2.68,1.07,0,0,0,0],
            "Solar PV - roof": [0,0,0,0,0,0,0.97,3.12,6.97,7.86,10.01,13.52,15.86,14.67,13.21,11.08,8.95,5.72,2.99,1.57,0,0,0,0],
        }  

        
    
    def set_energy_sources_specs(
        self,
        number_of_energy_markets=1,
        cost_per_measurement_unit_low=55.0, 
        cost_per_measurement_unit_hi=325.0,
        cost_per_measurement_unit_multiplier_low=1.0,
        cost_per_measurement_unit_multiplier_hi=2.0,
        co2_per_measurement_unit_multiplier_low=1.0,
        co2_per_measurement_unit_multiplier_hi=3.0
    ):
        self.energy_markets={}
        self.number_of_energy_markets = number_of_energy_markets
        self.generate_energy_markets(number_of_energy_markets=number_of_energy_markets, 
                                     cost_per_measurement_unit_low=cost_per_measurement_unit_low,
                                     cost_per_measurement_unit_hi=cost_per_measurement_unit_hi,
                                     cost_per_measurement_unit_multiplier_low=cost_per_measurement_unit_multiplier_low,
                                     cost_per_measurement_unit_multiplier_hi=cost_per_measurement_unit_multiplier_hi,
                                     co2_per_measurement_unit_multiplier_low = co2_per_measurement_unit_multiplier_low,
                                     co2_per_measurement_unit_multiplier_hi=co2_per_measurement_unit_multiplier_hi
                                     )
        self.generate_local_sources( number_of_energy_sources=1, 
                                     cost_per_measurement_unit_low=cost_per_measurement_unit_low/3,
                                     cost_per_measurement_unit_hi=cost_per_measurement_unit_hi/3,
                                     cost_per_measurement_unit_multiplier_low=cost_per_measurement_unit_multiplier_low,
                                     cost_per_measurement_unit_multiplier_hi=cost_per_measurement_unit_multiplier_hi,
                                     co2_per_measurement_unit_multiplier_low = co2_per_measurement_unit_multiplier_low,
                                     co2_per_measurement_unit_multiplier_hi=co2_per_measurement_unit_multiplier_hi
                                     )        
        self.energy_sources_specs = True

    def set_cost_component_specs(
        self,
        number_of_cost_components=1,
        cost_per_measurement_unit_low=55.0, 
        cost_per_measurement_unit_hi=325.0,
        cost_per_measurement_unit_multiplier_low=1.0,
        cost_per_measurement_unit_multiplier_hi=2.0,
    ):
        self.cost_component={}
        self.number_of_cost_components = number_of_cost_components
        base_cost_per_measurement_unit_low=cost_per_measurement_unit_low
        base_cost_per_measurement_unit_hi=cost_per_measurement_unit_hi
        for c_id in range(number_of_cost_components):
            c_name = f"Cost Factor {c_id}"
            #generate costs
            charge_intervals = []
            low_cost = random.uniform(cost_per_measurement_unit_multiplier_low,cost_per_measurement_unit_multiplier_hi)*base_cost_per_measurement_unit_low
            high_cost = random.uniform(cost_per_measurement_unit_multiplier_low,cost_per_measurement_unit_multiplier_hi)*base_cost_per_measurement_unit_hi
            #TODO: Emulate cost per 8h shift 
            for t in range(self.start_time_int, self.finish_time_int+8*60, 8*60):
                cost_per_kwh = round(
                    random.uniform(low_cost, high_cost),
                    2,
                )
                start = int_to_datetime(self.start_time,t)
                end = int_to_datetime(self.start_time,t+8*60)
                ci = CostInterval(from_datetime=start, to_datetime=end, cost = cost_per_kwh)
                charge_intervals.append(ci)
            a_cost = CostComponent(name=c_name, charge_intervals = charge_intervals)
            self.cost_component[c_name]=a_cost
        self.cost_component_specs = True
    
    def set_factories_specs(
        self,
        number_of_factories=1,
        number_of_machines_low=1,
        number_of_machines_high=10,
        cost_per_measurement_unit_multiplier_low=1.0,
        cost_per_measurement_unit_multiplier_hi=2.0,
        co2_per_measurement_unit_multiplier_low=1.0,
        co2_per_measurement_unit_multiplier_hi=3.0,
        number_of_storages_low=0,
        number_of_storages_high=3,
    ):
        self.number_of_factories = number_of_factories
        self.factories = {}
        for f_id in range(number_of_factories):
            a_market = random.choice(list(self.energy_markets.values()))
            a_factory = Factory(f_id, name=f"Factory{f_id}", energy_market=a_market)
            # generate machines
            number_of_machines = random.randint(number_of_machines_low,number_of_machines_high)
            self.gen_machine_specs(a_factory, number_of_machines)
            number_of_storages = random.randint(number_of_storages_low,number_of_storages_high)
            self.gen_storages_specs(a_factory, number_of_storages)
            self.gen_job_time_consumptions(a_factory)
            self.gen_cost_component_specs(a_factory)
            self.factories[a_factory.id]=a_factory
        self.factory_specs = True
    
    def gen_cost_component_specs(self,
                                 a_factory:Factory, 
                                 number_of_cost_components=1,
                                 period_low=1,
                                 period_high=10,
                                 cost_per_measurement_unit_multiplier_low=0.0,
                                 cost_per_measurement_unit_multiplier_hi=2.0
        ):
        for c_id in range(number_of_cost_components):
            c_name = f"Shift Cost{c_id}"
            #generate costs
            charge_intervals = []
            periods_number = random.randint(period_low, period_high)
            period_duration = self.horizon_finish/periods_number
            for i in range(periods_number):
                cost = round(
                    random.uniform(cost_per_measurement_unit_multiplier_low,cost_per_measurement_unit_multiplier_hi),
                    2,
                )
                start = int_to_datetime(self.start_time,i*period_duration)
                end = int_to_datetime(self.start_time,i*period_duration+period_duration)
                ci = CostInterval(from_datetime=start, to_datetime=end, cost = cost)
                charge_intervals.append(ci)
            a_cost = CostComponent(name=c_name, cost_intervals = charge_intervals)
            a_factory.add_cost_component(a_cost)

    def gen_machine_specs(
        self,
        factory,
        number_of_machines,
        operational_modes_low=1,
        operational_modes_high=3,
        setup_time_low=1,
        setup_time_hi=20,
        setup_consumption_low=1,
        setup_consumption_hi=20,
        unavailable_intervals=1,
        unavailability_percentage=0.2,
    ):
        for m_id in range(number_of_machines):
            op_dict={}
            op_modes = random.randint(operational_modes_low, operational_modes_high)
            for opm in range(op_modes):  
                 op_dict[opm] = MachineOperationalMode(id=opm)         
            a_machine = Machine(id=m_id, name=f"machine{m_id}",operational_modes=op_dict)
            for attr1 in self.attributes.keys():
                for attr2 in self.attributes.keys():
                    setup_time = random.randint(setup_time_low, setup_time_hi)
                    setup_consumption = random.randint(
                        setup_consumption_low, setup_consumption_hi
                    )
                    setupP = SetupProperties(
                        task_attribute1=attr1, task_attribute2=attr2,
                        time=setup_time,
                        consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = setup_consumption
                            )
                            ]
                    )
                    a_machine.add_machine_setup(setupP)
            factory.add_machine(a_machine)

            # unavailable_periods = SyntheticProblemGenerator.find_unavailable_intervals(
            #     unavailable_intervals,
            #     self.problem.horizon_finish,
            #     unavailability_percentage,
            # )
            # for unavailable_start_time, unavailable_finish_time in unavailable_periods:
            #     self.problem.add_machine_unavailability(
            #         m_id, unavailable_start_time, unavailable_finish_time
            #     )

            # 1 unavailable interval for each machine
            num_of_unavailable_periods = random.randint(0,unavailable_intervals)
            for up in range(num_of_unavailable_periods):
                unavailable_start_time = random.randrange(
                    self.start_time_int, self.finish_time_int
                )
                duration = random.randint(1, int(self.horizon_finish*unavailability_percentage))
                unavailable_finish_time = unavailable_start_time + duration
                if unavailable_finish_time > self.finish_time_int:
                    unavailable_finish_time = self.finish_time_int
                ump = UnavailablePeriod(
                    from_datetime=int_to_datetime(self.start_time,unavailable_start_time),
                    to_datetime=int_to_datetime(self.start_time,unavailable_finish_time)
                    )
                a_machine.add_machine_unavailability(ump)
        self.machines_specs = True
    
    def gen_storages_specs(
        self,
        factory,
        number_of_storages,
        capacity_low=10,
        capacity_high=50,
        minimum_capacity_low=0,
        minimum_capacity_hi=20,
        maximum_capacity_low=80,
        maximum_capacity_hi=100
    ):
        for s_id in range(number_of_storages):
            capacity = random.randint(capacity_low, capacity_high)
            min_capacity = random.randint(minimum_capacity_low, minimum_capacity_hi)
            max_capacity = random.randint(maximum_capacity_low, maximum_capacity_hi)  
            a_storage = ProductStorage(id=s_id, name=f"storage{s_id}", capacity=capacity, min_value=min_capacity, max_value=max_capacity,eligible_products=[])
            factory.add_storage(a_storage)
        self.storages_specs = True

    def set_jobs_specs(
        self,
        number_of_jobs,
        min_number_of_tasks=1,
        max_number_of_tasks=5,
        incompatible_machine_probability=0.2,
        probability_of_prev_task_connecting=1.0,
        task_duration_low=15,
        task_duration_hi=100,
        task_consumption_low=20,
        task_consumption_hi=100,
        task_dependency_time_min=0,
        task_dependency_time_max=10,
        task_dependency_consumption=5,
        flexibility_multiplier=2
    ):
        self.number_of_jobs = number_of_jobs
        self.incompatible_machine_probability= incompatible_machine_probability
        self.task_duration_low=task_duration_low
        self.task_duration_hi=task_duration_hi
        self.task_consumption_low=task_consumption_low
        self.task_consumption_hi=task_consumption_hi
        self.jobs = {}
        for j_id in range(number_of_jobs):
            earliest_start_time_int = random.randrange(
                self.start_time_int,
                self.start_time_int + (self.finish_time_int - self.start_time_int) // 2,
            )
            number_of_tasks = random.randint(min_number_of_tasks, max_number_of_tasks)
            task_list = []
            total_time=0
            for t_id in range(number_of_tasks):
                attr = random.choice(range(self.number_of_attributes))
                duration = random.randint(task_duration_low, task_duration_hi)+1
                total_time+=duration
                a_task = Task(id=t_id,job_id=j_id,attribute_id=attr,
                earliest_start_time=int_to_datetime(self.start_time,earliest_start_time_int))
                task_list.append(a_task)                
            latest_finish_time_int = earliest_start_time_int + flexibility_multiplier*total_time
            if latest_finish_time_int > self.finish_time_int:
                latest_finish_time_int = self.finish_time_int
            task_dict={}
            dep_list=[]
            task_dict[task_list[0].id] = task_list[0]
            #TODO make more generic dependencies in the future
            for t0, t1 in zip(task_list, task_list[1:]):
                task_dict[t1.id]=t1
                min_time = random.randint(0,task_dependency_time_min)
                max_time = random.randint(min_time,task_dependency_time_max)
                consumption=float(random.randint(0,task_dependency_consumption))
                td = DependencyItem(
                    task_id1=t0.id,
                    task_id2=t1.id,
                    min_time=min_time,
                    max_time=max_time,
                    consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = consumption
                            )
                            ]
                    )
                dep_list.append(td)
            a_job = Job(id=j_id,
                earliest_start_time=int_to_datetime(self.start_time,earliest_start_time_int),
                latest_finish_time=int_to_datetime(self.start_time,latest_finish_time_int),
                tasks=task_dict,
                task_dependencies=dep_list                
                )
            a_job._est = earliest_start_time_int
            a_job._lft = latest_finish_time_int
            self.jobs[a_job.id]=a_job
        self.jobs_specs = True

    def gen_job_time_consumptions(self,factory:Factory,
            incompatible_op_mode_probability = 0.1,
            machine_operational_mode_variance = 0.3):
        for j_id in self.jobs:
            m_count=0
            job = self.jobs[j_id]
            duration = job.latest_finish_time - job.earliest_start_time
            duration_int = duration.total_seconds() / 60
            ntasks = len(job.tasks)
            task_duration_int = duration_int // (2*ntasks)
            if task_duration_int<self.task_duration_low:
                task_duration_int = self.task_duration_low
            for t_id in job.tasks:
                task = job.tasks[t_id]
                for m_id,machine in factory.get_machines().items():
                    if random.random() < self.incompatible_machine_probability:
                        continue
                    else:
                        m_count+=1
                    base_time = random.randrange(self.task_duration_low, task_duration_int)
                    base_consumption = random.randint(self.task_consumption_low, self.task_consumption_hi)
                    op_count = 0
                    for opm in machine.operational_modes:
                        if random.random() < incompatible_op_mode_probability:
                            continue
                        else:
                            op_count+=1
                        variance = random.uniform(-machine_operational_mode_variance,machine_operational_mode_variance)
                        time = int((1+variance)*base_time)
                        consumption = round((1+variance)*base_consumption,2)
                        usage = FixedTimeTaskResourceUseProperties(
                            job_id=j_id,
                            task_id=t_id,
                            time=time,
                            consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = consumption
                            )
                            ]
                        )
                        machine.add_machine_use(usage, opm)

    def fix_job_task_feasibility(self):
        for j_id, job in self.scenario.problem.jobs.items():
            for t_id,task in job.tasks.items():
                m_count = self.scenario.problem.eligible_machines_for_task(j_id,t_id)
                if len(m_count)==0:
                    factory = random.choice(list(self.scenario.problem.factories.values()))
                    machine = random.choice(list(factory.get_machines().values()))
                    opm = random.choice(list(machine.operational_modes.keys()))
                    n_tasks = len(job.tasks)
                    duration = int(int(job._lft - job._est) / (2*n_tasks))
                    time = random.randrange(self.task_duration_low, duration)
                    consumption = random.randint(self.task_consumption_low, self.task_consumption_hi)
                    usage = FixedTimeTaskResourceUseProperties(
                            job_id=j_id,
                            task_id=t_id,
                            time=time,
                            consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = consumption
                            )
                            ]
                        )
                    machine.add_machine_use(usage, opm)
                    


    def set_defaults(self, number_of_attributes=2):
        self.set_parameters_specs(id="1", scenario="scenario1")
        self.set_optional_parameters_specs()
        self.set_core_specs(
            start_time=datetime.fromisoformat("2022-05-24 00:00:00"),
            finish_time=datetime.fromisoformat("2022-05-25 11:59:59")
        )
        self.set_attribute_specs(number_of_attributes=number_of_attributes)
        self.fix_feasibility = True
        

    def generate(self):
        if not self.parameters_specs:
            logging.error("Parameters specs not set")
        if not self.core_specs:
            logging.error("Core specs not set")
        if not self.attributes_specs:
            logging.error("Job attributes specs not set")
        if not self.jobs_specs:
            logging.error("Jobs specs not set")
        if not self.energy_sources_specs:
            logging.error("Energy cost specs not set")
        if not self.factory_specs:
            logging.error("Parameters specs not set")
        self.problem = Problem(
            start_time=self.start_time,
            finish_time=self.finish_time,
            task_attributes=self.attributes,
            jobs=self.jobs,
            factories=self.factories,
        )
        if hasattr(self,"productSequences"):
            self.problem.productSequences = self.productSequences
        if hasattr(self,"in_products"):
            self.problem.in_products = self.in_products
        if hasattr(self,"out_products"):
            self.problem.out_products = self.out_products
        if hasattr(self,"initial_solution"):
            if hasattr(self,"generate_initial_solution"):
                self.generate_initial_solution(self.problem)
            initial_solution = self.initial_solution
        else:
            initial_solution = None
        self.scenario = Scenario(
            parameters=self.parameters,
            problem=self.problem,
            energy_sources=self.energy_markets,
            initial_solution=initial_solution
        )
        self.scenario.fix_after_load_from_json()
        if self.fix_feasibility:
            self.fix_job_task_feasibility()     


    def save(self, filename="./data/synthetic_problem.json"):
        with open(filename, "w", encoding="utf-8") as f:
            txt = self.get_json()
            f.write(txt)

    def get_json(self):
        self.generate()
        dict = self.scenario.model_dump_json(exclude_unset=True, indent=3)
        return dict

if __name__ == "__main__":
    number_of_job_attributes = 3
    number_of_jobs = 10
    max_number_of_tasks = 5
    number_of_energy_markets=1
    number_of_factories= 1
    number_of_machines = 4
    fname = f"data/V5/synthetic_problem_f{number_of_factories}_j{number_of_jobs}_m{number_of_machines}_a{number_of_job_attributes}.json"
    spg = ProblemGeneratorV5()
    spg.set_defaults(
        number_of_job_attributes
    )
    spg.set_energy_sources_specs(
        number_of_energy_markets=number_of_energy_markets
    )
    spg.set_jobs_specs(number_of_jobs=number_of_jobs,
        	max_number_of_tasks=max_number_of_tasks
    )    
    spg.set_factories_specs(
        number_of_factories=number_of_factories,
        number_of_machines_low = 2,
        number_of_machines_high = 4
    )
    spg.save(
        filename=fname
    )