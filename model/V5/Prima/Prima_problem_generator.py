from datetime import datetime,timedelta
import logging
from copy import copy, deepcopy
import random

from .Prima_IO import import_product_transitions, import_products, load_orders_from_xlsx

from ..synthetic_problem_generator import ProblemGeneratorV5
from ..scenario import CostComponent, ResourceUse, Resource, ResourceType
from ..scenario import Factory, ResourceUse, FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties
from ..scenario import Machine, MachineOperationalMode, MachineProcessingType, SetupProperties, MachineConsumptionAggregation
from ..scenario import Job, Task, Attribute, DependencyItem
from ..scenario import Product, ProductSequence
from ..scenario import Problem,Solution
from ..scenario import datetime_to_int, int_to_datetime

class Prima_PG(ProblemGeneratorV5):
    def __init__(self, path, xls_input_file="Prima_problem_1.xlsx"):
        super().__init__()      
        # load from xls
        self.transition_dict = import_product_transitions(path+"data/3DNT/Prima_transitions_v1.0.xlsx")
        self.all_product_tasks,self.states, self.all_consumptions, self.out_products, self.all_machine_modes = import_products(path+"data/3DNT/Prima_products_v1.0.xlsx")
        self.order_data, self.product_data, self.schedule_start_dt, self.schedule_end_dt = load_orders_from_xlsx(path+"data/3DNT/"+xls_input_file)

        self.products_specs = True

        #Perform relaxation
        # self.schedule_start_dt = self.schedule_start_dt - timedelta(days=1)
        # self.schedule_end_dt = self.schedule_end_dt + timedelta(hours=8)

        self.set_defaults()
        self.set_energy_sources_specs(
            number_of_energy_markets=1
        )
        self.set_jobs_specs(
            number_of_jobs=len(self.order_data)
        )    
        self.set_factories_specs(
            number_of_factories=1,
            number_of_machines_high=4
        )
        self.initial_solution = None

    def set_defaults(self, number_of_job_attributes=2):
        self.set_parameters_specs(id="1", scenario="scenario1")
        self.parameters.resources ["Electricity"] = Resource(
            name="Electricity",
            measurement_unit="KWh",
            measurement_type=ResourceType.CONTINUES
        )
        self.set_optional_parameters_specs()
        self.set_core_specs(
            start_time=self.schedule_start_dt,
            finish_time=self.schedule_end_dt
        )
        self.set_attribute_specs(number_of_attributes=len(self.states))
        self.fix_feasibility = False
    
    def set_attribute_specs(self, number_of_attributes):
        self.attributes = {}
        for attrib in self.states.values():
            self.attributes[
                attrib.attribute_id
            ] = attrib
        
        # Also add from transition table
        for (attrib_id1,attrib_id2),value in self.transition_dict.items():
            ta1 = "Product_"+str(attrib_id1)
            ta2 = "Product_"+str(attrib_id2)
            if ta1 not in self.states:
               attr =  Attribute(
                    attribute_id=len(self.attributes),
                    description=ta1,
                    state={
                        "Product":float(attrib_id1),          
                    }
                )
               self.attributes[len(self.attributes)]=attr
               self.states[ta1]=attr
               
            if ta2 not in self.states:
               attr =  Attribute(
                    attribute_id=len(self.attributes),
                    description=ta2,
                    state={
                        "Product":float(attrib_id2),          
                    }
                )
               self.attributes[len(self.attributes)]=attr
               self.states[ta2]=attr
        self.number_of_attributes = len(self.attributes.keys())

        self.attributes_specs = True
    
    '''
        For the Prima case, there is no in product (should it be some material???) and will generate a variable quantity of the same out product
    '''

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
        #Make the assumption that all machines can process all products         
        self.machines = []        
        for m in range(number_of_machines):
            self.machines.append(Machine(m,name="Prima_"+str(m)))
                                 
        # create opm and setup for each machine
        for m in self.machines:
            m_id = m.id            
            # TODO: Single Operation mode. Is it correct??
            op_dict={}
            for opm_mode,opm in self.all_machine_modes.items():
               op_dict[opm.id] = deepcopy(opm)
            m.operational_modes = op_dict

            # TODO: The same setup time and consumptions per machine. This should be extracted dynamically through ML from sensor data
            for attr1_id,attr1 in self.attributes.items():
                for attr2_id,attr2 in self.attributes.items():
                    setupP = SetupProperties(
                        task_attribute1=attr1_id, task_attribute2=attr2_id,
                        time=self.transition_dict[attr1_id, attr2_id][2],
                        consumptions=[
                            ResourceUse(
                                resource_name= "Electricity",
                                consumption = self.transition_dict[attr1_id, attr2_id][3]
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
        # break the order up to 5 production tasks
        number_of_tasks = 5
        self.number_of_jobs = number_of_jobs
        self.jobs = {}
        self.productSequences={}

        j_id=0

        for order_id,order in self.order_data.items():
            earliest_start_time = order[1]
            latest_finish_time = order[2]
            attr = order[0]
            product = self.out_products[attr]
            if attr not in self.product_data:
                logging.error(
                    f"Not matching product for order {order_id}: {order[1]}"
                )
            quantity = order[3]

            task_list = []

            # make sure that orders are withing the horizon
            latest_finish_time = min(self.finish_time,latest_finish_time)
            
            total_time=0
            has_variable_time_task = True
            num_of_variable_time_tasks = 3

            for t_id in range(number_of_tasks):
                a_task = deepcopy (self.all_product_tasks[attr])
                a_task.id = t_id
                a_task.job_id = order_id
                a_task.earliest_start_time = earliest_start_time
                a_task.latest_finish_time = latest_finish_time
                a_task.product_limits = [(attr,int(quantity/(int(num_of_variable_time_tasks/2)+1))-1,quantity)]
                task_list.append(a_task)
                
            
            task_dict={}
            dep_list=[]
            task_dict[task_list[0].id] = task_list[0]
            #TODO make more generic dependencies in the future
            for t0, t1 in zip(task_list, task_list[1:]):
                task_dict[t1.id]=t1
                # min_time = 0
                # use_same_machine = False
                # keep_together = False
                # consumption = 0.0
                # td = DependencyItem(
                #     task_id1=t0.id,
                #     task_id2=t1.id,
                #     min_time=min_time,
                #     use_same_machine=use_same_machine,
                #     keep_together=keep_together,
                #     consumptions=[
                #             ResourceUse(
                #                 resource_name= "Electricity",
                #                 consumption = consumption
                #             )
                #             ]
                #     )
                # dep_list.append(td)
            
            j_est = earliest_start_time
            j_lft = latest_finish_time
            a_job = Job(id=j_id,
                        product_name=product.name,
                        earliest_start_time=j_est,
                        latest_finish_time=j_lft,
                        tasks=task_dict,
                        task_dependencies=dep_list,
                        product_limits = [(attr,quantity,quantity)]               
                )
            a_job._est = datetime_to_int(self.start_time, j_est)
            a_job._lft = datetime_to_int(self.start_time, j_lft)
            self.jobs[a_job.id]=a_job

            if len(self.product_data[product.id])>1:
                if product.id not in self.productSequences:
                    self.productSequences[product.id]=ProductSequence(
                        product_id=product.id,
                        job_ids=[]
                    )
                self.productSequences[product.id].job_ids.append(a_job.id)
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
        # for c_id in range(number_of_cost_components):
        #     c_name = f"Work Periods Cost #{c_id}"
        #     #generate costs
        #     cost_periods = self.no_work_periods+self.work_periods
        #     a_cost = CostComponent(name=c_name, cost_intervals = cost_periods,resource_name="Time")
        #     a_factory.add_cost_component(a_cost)
        pass

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
                    for opm in machine.operational_modes:
                        #skip modes that the product cannot be processed
                        if (task.attribute_id,opm) not in self.all_consumptions:
                            continue
                        usage = deepcopy(self.all_consumptions[task.attribute_id,opm])
                        usage.job_id=j_id
                        usage.task_id=t_id
                        for prod,min_prod,max_prod in task.product_limits:
                            if usage.production[0][0] == prod:
                                # Allow 0 time to skip this task if need to 
                                usage.min_time = 0 
                                usage.max_time = int(max_prod*usage.production[0][1]/usage.production[0][2])  
                        machine.add_machine_use(usage, opm)
    
    def generate_initial_solution(self,problem:Problem):
        pass

        

         

    
