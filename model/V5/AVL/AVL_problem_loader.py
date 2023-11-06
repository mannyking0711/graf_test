from datetime import datetime,timedelta
import logging
import copy
import random

from .AVL_IO import import_testbed_transition, import_tests, load_suites, SPECIAL_TESTS
from .AVL_IO import read_testbed_transition_xls,read_suites_from_xlsx,read_tests_xls
from .AVL_IO import read_testbed_transition_json,read_suites_from_json,read_tests_json
from .AVL_IO import read_testbed_transition_json_str,read_suites_from_json_str,read_tests_json_str
from .AVL_IO import load_energy_market, read_energy_market_from_json, read_energy_market_from_json_str

from ..synthetic_problem_generator import ProblemGeneratorV5
from ..scenario import CostComponent, ResourceUse, Resource, ResourceType, EmissionsGenerated
from ..scenario import Factory, ResourceUse, FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties
from ..scenario import Machine, MachineOperationalMode, MachineProcessingType, SetupProperties, MachineConsumptionAggregation
from ..scenario import Job, Task, Attribute, DependencyItem
from ..scenario import Product, ProductSequence
from ..scenario import Problem,Solution
from ..scenario import datetime_to_int, int_to_datetime
from ..scenario import CostInterval,EnergySource

class AVL_PG(ProblemGeneratorV5):
    
    def __init__(self):
        super().__init__()    
    
    def load_problem_json_str(self,transitions_json, tests_json, reservations_json, energy_prices_json=None):
        if not reservations_json:
            self.load_static_dictionaries_from_json_str(transitions_json, tests_json)
            self.static_init()
        else:
            if energy_prices_json == None:
                self.load_dictionaries_from_json_str(transitions_json, tests_json, reservations_json)
            else:
                self.load_dictionaries_from_json_str(transitions_json, tests_json, reservations_json, energy_prices_json)
            self.full_init()

    def load_problem_json(self,path,transitions_json, tests_json, reservations_json, energy_prices_json=None):
        if not reservations_json:
            self.load_static_dictionaries_from_json(path+"data/AVL/"+transitions_json, path+"data/AVL/"+tests_json)
            self.static_init()
        else:
            if energy_prices_json == None:
                self.load_dictionaries_from_json(path+"data/AVL/"+transitions_json, path+"data/AVL/"+tests_json, path+"data/AVL/"+reservations_json)
            else:
                self.load_dictionaries_from_json(path+"data/AVL/"+transitions_json, path+"data/AVL/"+tests_json, path+"data/AVL/"+reservations_json, path+"data/AVL/"+energy_prices_json)
            self.full_init()    
    
    def load_problem_reservations_json(self,path, reservations_input_file):
        self.load_static_dictionaries_from_xlsx(path)
        if not reservations_input_file:
            self.static_init()
        else:
            self.suites_data, self.vehicle_data, self.schedule_start_dt, self.schedule_end_dt, self.work_periods, self.no_work_periods = load_suites(read_suites_from_json(path+"data/AVL/"+reservations_input_file))
            self.energy_market = None        
            self.full_init()
    
    def load_problem_xlsx(self,path, xls_input_file="problem01_reservations_anon.xlsx"):
        if not xls_input_file:
            self.load_static_dictionaries_from_xlsx(path)
            self.static_init()
        else:
            self.load_dictionaries_from_xlsx(path,xls_input_file)
            self.full_init()
    
    def load_dictionaries_from_json_str(self, transitions_json, tests_json, reservations_json, energy_prices_json=None):
        # load from xls
        self.load_static_dictionaries_from_json_str(transitions_json, tests_json)
        self.suites_data, self.vehicle_data, self.schedule_start_dt, self.schedule_end_dt, self.work_periods, self.no_work_periods = load_suites(read_suites_from_json_str(reservations_json))
        if energy_prices_json!= None:
            self.energy_market = load_energy_market(read_energy_market_from_json_str(energy_prices_json)) 

    def load_dictionaries_from_json(self, transitions_json, tests_json, reservations_json, energy_prices_json=None):
        # load from xls
        self.load_static_dictionaries_from_json(transitions_json, tests_json)
        self.suites_data, self.vehicle_data, self.schedule_start_dt, self.schedule_end_dt, self.work_periods, self.no_work_periods = load_suites(read_suites_from_json(reservations_json))
        if energy_prices_json!= None:
            self.energy_market = load_energy_market(read_energy_market_from_json(energy_prices_json)) 

    def load_dictionaries_from_xlsx(self, path, xls_input_file="problem01_reservations_anon.xlsx"):
        # load from xls
        self.load_static_dictionaries_from_xlsx(path)
        self.suites_data, self.vehicle_data, self.schedule_start_dt, self.schedule_end_dt, self.work_periods, self.no_work_periods = load_suites(read_suites_from_xlsx(path+"data/AVL/"+xls_input_file))
    
    def load_static_dictionaries_from_xlsx(self, path):
        # load from xls
        self.transition_dict = import_testbed_transition(read_testbed_transition_xls(path+"data/AVL/transitions_v2.1.xlsx"))
        self.all_tests, self.all_resources, self.test_states, self.all_consumptions, self.all_conditioning = import_tests(read_tests_xls(path+"data/AVL/tests_v2.0.xlsx"))

    def load_static_dictionaries_from_json(self, transitions_json, tests_json):
        # load from xls
        self.transition_dict = import_testbed_transition(read_testbed_transition_json(transitions_json))
        self.all_tests, self.all_resources, self.test_states, self.all_consumptions, self.all_conditioning = import_tests(read_tests_json(tests_json))   
    
    def load_static_dictionaries_from_json_str(self, transitions_json, tests_json):
        # load from xls
        self.transition_dict = import_testbed_transition(read_testbed_transition_json_str(transitions_json))
        self.all_tests, self.all_resources, self.test_states, self.all_consumptions, self.all_conditioning = import_tests(read_tests_json_str(tests_json))   


    def full_init(self):

        #Perform relaxation
        self.schedule_start_dt = self.schedule_start_dt - timedelta(days=1)
        self.schedule_end_dt = self.schedule_end_dt + timedelta(hours=8)

        self.set_defaults()
        self.set_energy_sources_specs(
            number_of_energy_markets=1
        )
        self.gen_products(number_of_in_products=len(self.vehicle_data))
        self.set_jobs_specs(
            number_of_jobs=len(self.suites_data)
        )    
        self.set_factories_specs()
        self.initial_solution = None
    
    
    def static_init(self):
        #Perform relaxation
        self.schedule_start_dt = datetime.now()
        self.schedule_end_dt = datetime.now()

        self.set_defaults()  
        self.energy_markets={}
        self.energy_sources_specs = True

        self.jobs = {}
        self.jobs_specs = True

        self.set_factories_specs()
        self.initial_solution = None

    def set_defaults(self, number_of_job_attributes=2):
        self.set_parameters_specs(id="1", scenario="scenario1")
        self.parameters.resources ["Electricity"] = Resource(
            name="Electricity",
            measurement_unit="KWh",
            measurement_type=ResourceType.CONTINUES
        )
        self.parameters.resources ["Cooling"] = Resource(
            name="Cooling",
            measurement_unit="KWh",
            measurement_type=ResourceType.CONTINUES
        )
        self.parameters.resources["Heating"] = Resource(
            name="Heating",
            measurement_unit="KWh",
            measurement_type=ResourceType.CONTINUES
        )
        self.parameters.resources["Time"] = Resource(
            name="Working Time",
            measurement_unit="min",
            measurement_type=ResourceType.DISCRETE
        )
        self.set_optional_parameters_specs()
        self.set_core_specs(
            start_time=self.schedule_start_dt,
            finish_time=self.schedule_end_dt
        )
        self.set_attribute_specs(number_of_attributes=len(self.test_states))
        self.fix_feasibility = False
    
    def set_attribute_specs(self, number_of_attributes):
        self.attributes = {}
        for attrib in self.test_states.values():
            self.attributes[
                attrib.attribute_id
            ] = attrib
        
        # Also add from transition table
        for (attrib_id1,attrib_id2),value in self.transition_dict.items():
            ta1 = f"T{int(attrib_id1)}"
            ta2 = f"T{int(attrib_id2)}"
            if ta1 not in self.test_states:
               attr =  Attribute(
                    attribute_id=len(self.attributes),
                    description=ta1,
                    state={
                        "Temperature":float(value[0]),
                        "Humidity":float(value[1]),              
                    }
                )
               self.attributes[len(self.attributes)]=attr
               self.test_states[ta1]=attr
               
            if ta2 not in self.test_states:
               attr =  Attribute(
                    attribute_id=len(self.attributes),
                    description=ta2,
                    state={
                        "Temperature":float(value[2]),
                        "Humidity":float(value[3]),              
                    }
               )
               self.attributes[len(self.attributes)]=attr
               self.test_states[ta2]=attr
        self.number_of_attributes = len(self.attributes.keys())

        self.attributes_specs = True
    
    '''
        For the AVL case, the in product and the out product are the same, we just associate test suite with each car
    '''
    def gen_products(self,number_of_in_products=0):
        self.in_products={}
        self.out_products={}
        #Helper dictionary
        self.vehicle_name_to_id = {}
        v_id=0
        for v in self.vehicle_data:
            self.in_products[v_id] = Product(
                id = v_id,
                name=v
            )
            self.out_products[v_id]=self.in_products[v_id]
            self.vehicle_name_to_id[v] = v_id
            v_id+=1
        self.products_specs = True

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
        temp_to_attr={}
        #TODO: Check if humidity is important
        for attr_id,attr in self.attributes.items():
            temp_to_attr[int(attr.state["Temperature"])]=attr_id
                         
        self.machines = [
            Machine(0,name="TB401_GRZ",can_process=[temp_to_attr[23]]),
            Machine(1,name="TB402_GRZ",can_process=[temp_to_attr[23]]),
            Machine(2,name="TB403_GRZ",
                        can_process=[
                            #temp_to_attr[-30],
                            # TODO: Talk to AVL
                            temp_to_attr[-10],
                            temp_to_attr[-7],temp_to_attr[0],temp_to_attr[1],
                            temp_to_attr[8],temp_to_attr[10],temp_to_attr[14],
                            temp_to_attr[23],temp_to_attr[25], temp_to_attr[35],
                            temp_to_attr[38]
                            ]
                    ),
            Machine(3, name="CR1_GRZ", capacity=80,
                        can_process=[temp_to_attr[23]], 
                        processing_type=MachineProcessingType.CASCADING,
                        consumption_aggregation=MachineConsumptionAggregation.MAXIMUM
                    ),
            Machine(4, name="CR2_GRZ", capacity=7,
                        can_process=[
                            #temp_to_attr[-30],
                            # TODO: Talk to AVL
                            temp_to_attr[-10],
                            temp_to_attr[-7],temp_to_attr[0],temp_to_attr[1],
                            temp_to_attr[8],temp_to_attr[10],temp_to_attr[14],
                            temp_to_attr[23],temp_to_attr[25], temp_to_attr[35],
                            temp_to_attr[38]
                        ], 
                        processing_type=MachineProcessingType.CASCADING,
                        consumption_aggregation=MachineConsumptionAggregation.MAXIMUM
                    )
        ]

        # create opm and setup for each machine

        for m in self.machines:
            m_id = m.id
            
            # TODO: Single Operation mode. Is it correct??
            op_dict={}
            op_dict[0] = MachineOperationalMode(id=0)
            m.operational_modes = op_dict

            # TODO: The same setup time and consumptions per machine. This should be extracted dynamically through ML from sensor data

            for attr1_id,attr1 in self.attributes.items():
                if attr1_id not in m.can_process:
                    continue
                for attr2_id,attr2 in self.attributes.items():
                    if attr2_id not in m.can_process:
                        continue
                    setupP = SetupProperties(
                        task_attribute1=attr1_id, task_attribute2=attr2_id,
                        time=self.transition_dict[attr1.state["Temperature"], attr2.state["Temperature"]][4],
                        consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = self.transition_dict[attr1.state["Temperature"], attr2.state["Temperature"]][5]
                            ),
                            ResourceUse(
                                resource_name= "Heating",
                                consumption = self.transition_dict[attr1.state["Temperature"], attr2.state["Temperature"]][6]
                            ),
                            ResourceUse(
                                resource_name= "Cooling",
                                consumption = self.transition_dict[attr1.state["Temperature"], attr2.state["Temperature"]][7]
                            )
                        ]
                    )
                    m.add_machine_setup(setupP)
            factory.add_machine(m)
        self.machines_specs = True

    def set_jobs_specs(
        self,
        number_of_jobs
    ):
        self.number_of_jobs = number_of_jobs
        self.jobs = {}
        self.productSequences={}

        #Generate initial solution
        self.initialSolutionComponents = {}

        j_id=0

        for name,suite in self.suites_data.items():
            vehicle_id = suite[0]
            vehicle = self.out_products[self.vehicle_name_to_id[vehicle_id]]
            in_products=[(vehicle.id,1.0)]
            out_products=[(vehicle.id,1.0)]
            earliest_chamber_start_time = suite[1]
            latest_finish_time = suite[2]
            planned_chamber_start_time = suite[3]
            planned_finish_time = suite[4]
            planned_chamber = suite[9]
            duration = suite[5]
            attr = self.test_states.get(f"T{suite[6]}")
            if attr == None:
                print(f"Not matching temperature for suite {name}. Temperature {suite[6]} not found in transition table. Skipping.")
                continue
            if not attr:
                logging.error(
                    f"Not matching temperature for suite {name}: {suite[6]}"
                )
            test_list = suite[7].split(",")
            number_of_tasks = len(test_list)+3
            task_list = []

            #Allow conditioning to start in the previous day
            #TODO: Check what happens if the conditioning time of a suite is before the horizon start time
            conditioning_start_time = earliest_chamber_start_time-timedelta(days=1)
            #TODO: Check what happens if the latest finish time of a suite is beyond the horizon finish time
            #TODO: Check with AVL to allow that the finish time is the start of next day
            latest_finish_time = max(latest_finish_time, earliest_chamber_start_time + timedelta(days=1))
            latest_finish_time = min(self.finish_time,latest_finish_time)
            
            t_id=0
            #Add the conditioning task if the start time is mode than the horizon start time
            if conditioning_start_time>=self.start_time:
                task_list.append(
                    Task(id=t_id,job_id=j_id,name="Conditioning", 
                        earliest_start_time= conditioning_start_time, 
                        attribute_id=attr.attribute_id,
                        in_products=in_products,
                        out_products=out_products
                        )
                )
                t_id+=1

            in_chamber_total_time=0

            #Add the setup task
            task_list.append(
                Task(id=t_id,job_id=j_id,name="Set-up time", 
                     attribute_id=attr.attribute_id, 
                     earliest_start_time=earliest_chamber_start_time,
                     in_products=in_products,
                     out_products=out_products
                     )
            )
            self.initialSolutionComponents[(j_id,t_id)]=[planned_chamber_start_time+timedelta(minutes=in_chamber_total_time)]
            t_id+=1
            in_chamber_total_time+=self.all_consumptions["Set-up time"].time

            has_variable_time_task = False
            num_of_variable_time_tasks = 0

            max_conditioning_time = 0
            for test_name in test_list:
                if test_name == "Project Delay" or test_name not in self.all_consumptions:
                    print(f"Skipping test {name}-{test_name}. No consumption has been defined")
                    continue
                a_task = Task(id=t_id,job_id=j_id,name=test_name, 
                              attribute_id=attr.attribute_id, 
                              earliest_start_time=earliest_chamber_start_time+timedelta(minutes=in_chamber_total_time),
                              in_products=in_products,
                              out_products=out_products
                              )
                ru = self.all_consumptions[test_name]  
                self.initialSolutionComponents[(j_id,t_id)]=[planned_chamber_start_time+timedelta(minutes=in_chamber_total_time)]
                if isinstance(ru, VariableTimeTaskResourceUseProperties):
                    #TODO: we have a variable time task, skip and allow next tasks to start earlier than what they really can start
                    print("Variable time task: ["+test_name+"] as task("+str(j_id)+","+str(t_id)+")")
                    has_variable_time_task = True
                    num_of_variable_time_tasks+=1
                else:
                    in_chamber_total_time+=self.all_consumptions[test_name].time         

                if self.all_conditioning[test_name]>max_conditioning_time:
                      max_conditioning_time = self.all_conditioning[test_name]  
                task_list.append(a_task)
                t_id+=1

            #Add the Dismantling task
            task_list.append(
                Task(id=t_id,job_id=j_id,name="Dismantling time", 
                     attribute_id=attr.attribute_id, 
                     earliest_start_time=earliest_chamber_start_time+timedelta(minutes=in_chamber_total_time),
                     in_products=in_products,
                     out_products=out_products
                     )
            )
            self.initialSolutionComponents[(j_id,t_id)]=[planned_chamber_start_time+timedelta(minutes=in_chamber_total_time)]
            in_chamber_total_time+=self.all_consumptions["Dismantling time"].time

            #calculate the available time between tasks of the same job if all are fixed and there is still time available
            available_delay_time=duration-in_chamber_total_time
            
            if has_variable_time_task and available_delay_time>0:
                available_delay_time=0
            
            if available_delay_time<0:
                print(f"Planned time for job {name} with id {j_id} has negative available_delay_time = {available_delay_time}")
                duration +=-available_delay_time
            
            #Store planned times for job
            self.initialSolutionComponents[(j_id)]=[planned_chamber_start_time,planned_finish_time,duration,in_chamber_total_time,planned_chamber,available_delay_time]
            self.initialSolutionComponents[(j_id,0)]=[planned_chamber_start_time-timedelta(hours=max_conditioning_time)]

            #TODO: Check with AVL how they should be distributed

            diff = duration - in_chamber_total_time

            if (diff<0):
                diff = 0
            if not has_variable_time_task:
                diff /= t_id-1
            else:
                diff = 0

            task_dict={}
            dep_list=[]
            task_dict[task_list[0].id] = task_list[0]
            #TODO make more generic dependencies in the future
            for t0, t1 in zip(task_list, task_list[1:]):
                task_dict[t1.id]=t1
                min_time = 0
                max_time = int(diff)
                consumption=0 #TODO: Check with AVL if what is the transition energy consumption
                if t0.name=="Conditioning":
                    use_same_machine = False
                    keep_together = True
                    max_time=0
                else:
                    use_same_machine=True
                    keep_together=True
                td = DependencyItem(
                    task_id1=t0.id,
                    task_id2=t1.id,
                    min_time=min_time,
                    max_time=max_time,
                    use_same_machine=use_same_machine,
                    keep_together=keep_together,
                    consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = consumption
                            )
                            ]
                    )
                dep_list.append(td)
            
            # if conditioning_start_time < self.start_time:
            j_est = earliest_chamber_start_time
            # else:
            #     j_est = conditioning_start_time
            j_lft = latest_finish_time
            a_job = Job(id=j_id,
                        name=str(name),
                        product_name=vehicle.name,
                        earliest_start_time=j_est,
                        latest_finish_time=j_lft,
                        tasks=task_dict,
                        task_dependencies=dep_list,
                        duration_limits=[duration,duration]               
                )
            a_job._est = datetime_to_int(self.start_time, j_est)
            a_job._lft = datetime_to_int(self.start_time, j_lft)
            self.jobs[a_job.id]=a_job

            if len(self.vehicle_data[vehicle.name])>1:
                if vehicle.id not in self.productSequences:
                    self.productSequences[vehicle.id]=ProductSequence(
                        product_id=vehicle.id,
                        job_ids=[]
                    )
                self.productSequences[vehicle.id].job_ids.append(a_job.id)
            j_id+=1
        
        #Generate maintenance jobs for each chamber
        # for f in 

        self.jobs_specs = True
    
    def gen_cost_component_specs(self,
                                 a_factory:Factory, 
                                 number_of_cost_components=1,
                                 period_low=1,
                                 period_high=10,
                                 cost_per_measurement_unit_multiplier_low=0.0,
                                 cost_per_measurement_unit_multiplier_hi=2.0
        ):
        if not hasattr(self,"no_work_periods"):
            return
        for c_id in range(number_of_cost_components):
            c_name = f"Work Periods Cost #{c_id}"
            #generate costs
            cost_periods = self.no_work_periods+self.work_periods
            a_cost = CostComponent(name=c_name, cost_intervals = cost_periods,resource_name="Time")
            a_factory.add_cost_component(a_cost)

    def gen_job_time_consumptions(self,factory:Factory,
            incompatible_op_mode_probability = 0.1,
            machine_operational_mode_variance = 0.3):
        for j_id in self.jobs:
            m_count=0
            job = self.jobs[j_id]
            variable_tasks_ru = {}
            for t_id in job.tasks:
                task = job.tasks[t_id]
                if (j_id,t_id) in self.initialSolutionComponents:
                    self.initialSolutionComponents[(j_id,t_id)].append([])
                for m_id,machine in factory.get_machines().items():
                    #Only allow compatible machines
                    variable_tasks_ru[m_id]=[0,[]]
                    if task.attribute_id not in machine.can_process:
                        continue
                    if machine.capacity==1 and task.name != "Conditioning":
                        usage = copy.deepcopy(self.all_consumptions[task.name])
                        usage.job_id=j_id
                        usage.task_id=t_id
                        if isinstance(usage,VariableTimeTaskResourceUseProperties):
                            variable_tasks_ru[m_id][1].append(usage)
                        else:
                            variable_tasks_ru[m_id][0] += usage.time
                            self.initialSolutionComponents[(j_id,t_id)][1].append((m_id,usage.time,usage.time))
                    elif machine.capacity>1 and task.name == "Conditioning":
                        usage = self.generate_conditioning_resource_use(job)
                        self.initialSolutionComponents[(j_id,t_id)][1].append((m_id,usage.min_time,usage.max_time))
                    else:
                        continue     
                    #TODO: Is the consumption the same on all machines and operational modes???
                    for opm in machine.operational_modes:
                        machine.add_machine_use(usage, opm)
                for m_id in variable_tasks_ru:
                    if len(variable_tasks_ru[m_id][1])>0:
                        diff_low = job.duration_limits[0]-variable_tasks_ru[m_id][0]
                        diff_high = job.duration_limits[1]-variable_tasks_ru[m_id][0]
                        for usage in variable_tasks_ru[m_id][1]:
                            usage.min_time = diff_low
                            usage.max_time = diff_high
                            if ((usage.job_id,usage.task_id)) in self.initialSolutionComponents:
                                self.initialSolutionComponents[(usage.job_id,usage.task_id)][1].append((m_id,diff_low,diff_high))
        
    def generate_conditioning_resource_use(self,job:Job):
        min_conditioning_time = 0
        cond_task=None
        consumption = 0.5 #TODO: Talk with AVL 
        generated_co2 = 1.2 #TODO: Talk with AVL 
        # find max conditioning in a job
        for t_id,task in job.tasks.items():
            if task.name == "Conditioning":
                cond_task = task
            elif task.name in self.all_conditioning:
                if self.all_conditioning[task.name]*60>min_conditioning_time:
                    min_conditioning_time = self.all_conditioning[task.name]*60
        
        #TODO: check with AVL if this is correct
        max_conditioning_time = 2 * min_conditioning_time
        
        #Assume that the conditioning time has been performed before the schedule start time
        if job.earliest_start_time - timedelta(minutes=min_conditioning_time) < self.start_time:
            # #Assume that the conditioning time has been partially be performed before the schedule start time
            diff_time = (job.earliest_start_time - self.start_time)
            min_conditioning_time = 0
            max_conditioning_time = int(diff_time.total_seconds()/60)
            job.earliest_start_time = self.start_time 
            # conditioning_time = 0
        else:
           job.earliest_start_time =  max(job.earliest_start_time - timedelta(minutes=max_conditioning_time),self.start_time)
        
        job._est = datetime_to_int(self.start_time, job.earliest_start_time)
        cond_task.earliest_start_time = job.earliest_start_time
                            
        ru = VariableTimeTaskResourceUseProperties(
                            job_id=job.id,
                            task_id=cond_task.id,
                            min_time=min_conditioning_time,
                            max_time=max_conditioning_time, #TODO: talk with AVL
                            consumptions_per_min=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = consumption
                            )
                            ],
                            emissions_per_minute=[
                                EmissionsGenerated(
                                    emission_name= "CO2",
                                    generated_amount=float(generated_co2)
                                )
                            ]
                            )
        return ru
    
    def generate_initial_solution(self,problem:Problem):
        sol = Solution(id=1,problem_id="1",generated_date=datetime.now(),schedule=[])
        sol.fix_after_load_from_json(problem)
        machine_schedule = {}
        for m in self.machines:
            if m.capacity == 1:
                machine_schedule[m.id] = [0,[]]
            else:
                machine_schedule[m.id] = [0,[]]
        sorted_by_est = sorted(self.jobs.items(), key=lambda x:self.initialSolutionComponents[x[1].id,x[1].tasks[1].id][0])
        for j,job in sorted_by_est:
            if job.tasks[0].name == "Conditioning":
                setup_task = job.tasks[1]
            else:
                setup_task = job.tasks[0]
            
            cand_m = []
            setup_start_time = datetime_to_int(self.start_time,self.initialSolutionComponents[j,setup_task.id][0])
            for m_id,low,high in self.initialSolutionComponents[j,setup_task.id][1]:
                if machine_schedule[m_id][0]-5*60<setup_start_time:
                    cand_m.append(m_id)
            if len(cand_m)==0:
                min_diff = 10E10
                cand_m.append(-1)
                for m_id,low,high in self.initialSolutionComponents[j,setup_task.id][1]:
                    diff = machine_schedule[m_id][0] - setup_start_time 
                    if diff<min_diff:
                        cand_m[0]=m_id
            #assign it to the specified machine if it is been given
            if self.initialSolutionComponents[j][4]:
                m_name = self.initialSolutionComponents[j][4]
                for machine in self.machines:
                    if machine.name==m_name:
                        job_m=machine.id
                        break
                if job_m not in cand_m:
                    print (f"Proposed machine assignment for job {job.id}/{job.name}->{m_name} not in candidate machine list {cand_m}")
                    job_m = -1
                    setup_cand_est = 10E10
                    for j_m in cand_m:
                        if machine_schedule[j_m][0]>0:
                            last_j_att = problem.jobs[machine_schedule[j_m][1][-1][0]].tasks[machine_schedule[j_m][1][-1][1]].attribute_id
                            if last_j_att != setup_task.attribute_id:
                                cand_transition_time = self.machines[j_m].setup_time_consumption_from_attr1_to_attr2(last_j_att,setup_task.attribute_id)[0]
                            else:
                                cand_transition_time = 0
                            cand_machine_est = max(machine_schedule[j_m][0]+cand_transition_time, datetime_to_int(self.start_time, self.initialSolutionComponents[j][0]))
                            if cand_machine_est<setup_cand_est:
                                setup_cand_est = cand_machine_est
                                job_m = j_m
                        else:
                            job_m = random.choice(cand_m)
            else:
                job_m = -1
                setup_cand_est = 10E10
                for j_m in cand_m:
                    if machine_schedule[j_m][0]>0:
                        last_j_att = problem.jobs[machine_schedule[j_m][1][-1][0]].tasks[machine_schedule[j_m][1][-1][1]].attribute_id
                        if last_j_att != setup_task.attribute_id:
                            cand_transition_time = self.machines[j_m].setup_time_consumption_from_attr1_to_attr2(last_j_att,setup_task.attribute_id)[0]
                        else:
                            cand_transition_time = 0
                        cand_machine_est = max(machine_schedule[j_m][0]+cand_transition_time, datetime_to_int(self.start_time, self.initialSolutionComponents[j][0]))
                        if cand_machine_est<setup_cand_est:
                            setup_cand_est = cand_machine_est
                            job_m = j_m
                    else:
                        job_m = random.choice(cand_m)

            # self.initialSolutionComponents[(j_id)]=[planned_chamber_start_time,planned_finish_time,duration,in_chamber_total_time]   
            duration = self.initialSolutionComponents[j][2]   
            in_chamber_total_time = self.initialSolutionComponents[j][3]  
            available_variable_time = duration - in_chamber_total_time  

            transition_time=0
            if machine_schedule[job_m][0]>0:
                last_job_attibute = problem.jobs[machine_schedule[job_m][1][-1][0]].tasks[machine_schedule[job_m][1][-1][1]].attribute_id
                if last_job_attibute != setup_task.attribute_id:
                    transition_time = self.machines[job_m].setup_time_consumption_from_attr1_to_attr2(last_job_attibute,setup_task.attribute_id)[0]

            job_time = max(machine_schedule[job_m][0]+transition_time, datetime_to_int(self.start_time, self.initialSolutionComponents[j][0]))
            
            
            if job.tasks[0].name == "Conditioning":
                cond_machine=-1
                cond_est = 10E10
                new_job_time = job_time
                for m_id,low,high in self.initialSolutionComponents[j,0][1]:
                    if machine_schedule[m_id][0]==0:
                        cond_last_job_attibute=setup_task.attribute_id
                    else:
                        cond_last_job_attibute = problem.jobs[machine_schedule[m_id][1][-1][0]].tasks[machine_schedule[m_id][1][-1][1]].attribute_id
                    if cond_last_job_attibute != setup_task.attribute_id:
                        cond_transition_time = self.machines[m_id].setup_time_consumption_from_attr1_to_attr2(cond_last_job_attibute,setup_task.attribute_id)[0]
                        m_est = machine_schedule[m_id][0]+cond_transition_time
                    else: # make an assumption that enough capacity is available and try to overlap conditioning
                        m_est = job_time - low
                    if m_est<cond_est:
                        cond_est = m_est
                        cond_machine=m_id      
                        if cond_est + low > new_job_time:
                            new_job_time = cond_est + low       
                
                if cond_est < job._est:
                    cond_est = job._est
                
                job_time = new_job_time

                cond_duration = job_time-cond_est
                sol.schedule_task_to_machine(j,0,0,cond_machine,0,cond_est,cond_duration,0,0,[])
                machine_schedule[cond_machine][1].append((j,0,cond_est,cond_duration))
                machine_schedule[cond_machine][0]=cond_est+cond_duration
                
            for t,task in job.tasks.items():
                if task.name == "Conditioning":
                    continue
                for m_id,low,high in self.initialSolutionComponents[j,t][1]:
                    if m_id == job_m:
                        if low == high:
                            duration = low
                        else:
                            max_avail = min(available_variable_time,high)
                            duration = random.randint(low,max_avail)
                            available_variable_time -= duration
                        sol.schedule_task_to_machine(j,t,0,m_id,0,job_time,duration,0,0,[])
                        machine_schedule[m_id][1].append((j,t,job_time,duration))
                        job_time += duration
                        machine_schedule[m_id][0]=job_time
                        break
                
        self.initial_solution = sol
    
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
        self.energy_sources_specs = True

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
        if self.energy_market == None:
            super().generate_energy_markets(number_of_energy_markets,
                                            cost_per_measurement_unit_low,
                                            cost_per_measurement_unit_hi,
                                            cost_per_measurement_unit_multiplier_low,
                                            cost_per_measurement_unit_multiplier_hi,
                                            co2_per_measurement_unit_multiplier_low,
                                            co2_per_measurement_unit_multiplier_hi)
        else:
            m_name = f"Austria Electricity Energy Market"
            #generate costs
            
            charge_intervals = []
            for pred in self.energy_market:
                start = pred[0]
                end = pred[1]
                cost_per_kwh = pred[2]/1000
                co2_per_kwh = 0
                ci = CostInterval(from_datetime=start, to_datetime=end, cost = cost_per_kwh, emissions=co2_per_kwh)
                charge_intervals.append(ci)
            
            a_market = EnergySource(name=m_name, cost_intervals = charge_intervals)
            self.energy_markets[m_name]=a_market    

         

    
