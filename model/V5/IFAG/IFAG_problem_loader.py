from datetime import datetime,timedelta
import logging
import copy

from .IFAG_IO import import_static_information

from ..synthetic_problem_generator import ProblemGeneratorV5
from ..scenario import CostComponent, ResourceUse, Resource, ResourceType
from ..scenario import Factory, SetupProperties, ResourceUse, FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties
from ..scenario import Machine, MachineOperationalMode, MachineProcessingType, MachineConsumptionAggregation
from ..scenario import Job, Task, Attribute, DependencyItem
from ..scenario import Product, ProductSequence
from ..scenario import datetime_to_int, int_to_datetime

class IFAG_PG(ProblemGeneratorV5):
    def __init__(self, path, xls_input_file="problem01_reservations_anon.xlsx"):
        super().__init__()      
        # load from xls
        dict_df = import_static_information(path + '\\' + 'EnerMan_DataScheduling.xlsx')
        
        self.set_defaults()
        self.set_energy_sources_specs(
            number_of_energy_markets=1
        )
        self.gen_products(number_of_in_products=len(self.vehicle_data))
        self.set_jobs_specs(
            number_of_jobs=len(self.suites_data)
        )    
        self.set_factories_specs()
        self.generate_initial_solution()

    def set_defaults(self, number_of_job_attributes=2):
        self.set_parameters_specs(id="1", scenario="scenario1")
        self.parameters.resources ["Electricity"] = Resource(
            name="Electricity",
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
            Machine(0,name="TB401",
                    can_process=[temp_to_attr[23],
                                 temp_to_attr[25], 
                                 temp_to_attr[35]]
                    ),
            Machine(1,name="TB402",
                    can_process=[temp_to_attr[23],
                                 temp_to_attr[25],
                                 temp_to_attr[35]]
                    ),
            Machine(2,name="TB403",
                    can_process=[
                                #temp_to_attr[-30],
                                # TODO: Talk to AVL
                                temp_to_attr[-10],
                                temp_to_attr[-7],temp_to_attr[0],temp_to_attr[1],
                                temp_to_attr[8],temp_to_attr[10],temp_to_attr[14],
                                temp_to_attr[23],temp_to_attr[25], temp_to_attr[35]
                                 ]
                    ),
            Machine(3, name="CR1", capacity=80,
                    can_process=[temp_to_attr[23]], 
                    processing_type=MachineProcessingType.CASCADING,
                    consumption_aggregation=MachineConsumptionAggregation.MAXIMUM
                    ),
            Machine(4, name="CR2", 
                    capacity=7,
                    can_process=[
                        #temp_to_attr[-30],
                        # TODO: Talk to AVL
                        temp_to_attr[-10],
                        temp_to_attr[-7],temp_to_attr[0],temp_to_attr[1],
                        temp_to_attr[8],temp_to_attr[10],temp_to_attr[14],
                        temp_to_attr[23],temp_to_attr[25], temp_to_attr[35]
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
            attr = self.test_states.get(f"T{suite[6]}")
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
            latest_finish_time = min(self.finish_time,latest_finish_time)
            
            t_id=0
            #Add the conditioning task if the start time is mode than the horizon start time
            if conditioning_start_time>self.start_time:
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
            t_id+=1
            in_chamber_total_time+=self.all_consumptions["Set-up time"].time

            has_variable_time_task = False
            num_of_variable_time_tasks = 0

            for test_name in test_list:
                if test_name == "Project Delay":
                    continue
                a_task = Task(id=t_id,job_id=j_id,name=test_name, 
                              attribute_id=attr.attribute_id, 
                              earliest_start_time=earliest_chamber_start_time+timedelta(minutes=in_chamber_total_time),
                              in_products=in_products,
                              out_products=out_products
                              )
                ru = self.all_consumptions[test_name]
                if isinstance(ru, VariableTimeTaskResourceUseProperties):
                    #TODO: we have a variable time task, skip and allow next tasks to start earlier than what they really can start
                    print("Variable time task: ["+test_name+"] as task("+str(j_id)+","+str(t_id)+")")
                    has_variable_time_task = True
                    num_of_variable_time_tasks+=1
                    pass
                else:
                    in_chamber_total_time+=self.all_consumptions[test_name].time               
                task_list.append(a_task)
                t_id+=1

            #Add the Dismantling task
            task_list.append(
                Task(id=t_id,job_id=j_id,name="Dismantling time", 
                     attribute_id=attr.attribute_id, 
                     earliest_start_time=suite[1]+timedelta(minutes=in_chamber_total_time),
                     in_products=in_products,
                     out_products=out_products
                     )
            )
            in_chamber_total_time+=self.all_consumptions["Dismantling time"].time
            
            #TODO: Check with AVL how they should be distributed
            duration = suite[5]
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
                    keep_together = False
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
            
            if conditioning_start_time < self.start_time:
                j_est = earliest_chamber_start_time
            else:
                j_est = conditioning_start_time
            j_lft = latest_finish_time
            a_job = Job(id=j_id,
                        name=name,
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

        self.jobs_specs = True
    
    def gen_cost_component_specs(self,
                                 a_factory:Factory, 
                                 number_of_cost_components=1,
                                 period_low=1,
                                 period_high=10,
                                 cost_per_measurement_unit_multiplier_low=0.0,
                                 cost_per_measurement_unit_multiplier_hi=2.0
        ):
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
                for m_id,machine in factory.get_machines().items():
                    #Only allow compatible machines
                    if task.attribute_id not in machine.can_process:
                        continue
                    if machine.capacity==1 and task.name != "Conditioning":
                        usage = copy.deepcopy(self.all_consumptions[task.name])
                        usage.job_id=j_id
                        usage.task_id=t_id
                        if task.name.split("#")[0] in SPECIAL_TESTS:
                            variable_tasks_ru[m_id][1].append(usage)
                        else:
                            if m_id not in variable_tasks_ru:
                                variable_tasks_ru[m_id]=[0,[]]
                            variable_tasks_ru[m_id][0] += usage.time
                    elif machine.capacity>1 and task.name == "Conditioning":
                        usage = self.generate_conditioning_resource_use(job)
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
        
    def generate_conditioning_resource_use(self,job:Job):
        conditioning_time = 0
        cond_task=None
        consumption = 2.1 #TODO: Talk with AVL 
        # find max conditioning in a job
        for t_id,task in job.tasks.items():
            if task.name == "Conditioning":
                cond_task = task
            elif task.name in self.all_conditioning:
                if self.all_conditioning[task.name]*60>conditioning_time:
                    conditioning_time = self.all_conditioning[task.name]*60
        
        #Assume that the conditioning time has been performed before the schedule start time
        if job.earliest_start_time - timedelta(minutes=conditioning_time) < self.start_time:
            # #Assume that the conditioning time has been partially be performed before the schedule start time
            # conditioning_time = int((job.earliest_start_time - self.start_time).seconds()/60)
            # job.earliest_start_time = self.start_time    
            conditioning_time = 0
        else:
           job.earliest_start_time =  max(job.earliest_start_time - 2*timedelta(minutes=conditioning_time),self.start_time)
           job._est = datetime_to_int(self.start_time, job.earliest_start_time)
        
        cond_task.earliest_start_time = job.earliest_start_time
                            
        ru = VariableTimeTaskResourceUseProperties(
                            job_id=job.id,
                            task_id=cond_task.id,
                            min_time=conditioning_time,
                            max_time=2*conditioning_time, #TODO: talk with AVL
                            consumptions_per_min=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = consumption
                            )
                            ]
                            )
        return ru
    
    def generate_initial_solution(self):
        pass
        self.initial_solution = None
        

         

    
