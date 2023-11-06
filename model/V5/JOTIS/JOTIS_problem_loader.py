from datetime import datetime,timedelta
import logging
import copy
import random

from .JOTIS_IO import import_static_information, import_production_information, conches, gen_shifts

from ..synthetic_problem_generator import ProblemGeneratorV5
from ..scenario import CostComponent, ResourceUse, Resource, ResourceType
from ..scenario import Factory, ResourceUse, FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties
from ..scenario import Machine, MachineOperationalMode, MachineProcessingType, SetupProperties, MachineConsumptionAggregation
from ..scenario import Job, Task, Attribute, DependencyItem, ProcessingType
from ..scenario import Product, ProductSequence, ProductStorage
from ..scenario import Problem,Solution
from ..scenario import datetime_to_int, int_to_datetime

class JOTIS_PG(ProblemGeneratorV5):
    #TODO: talk with JOTIS about production resolution
    SHIFT_DURATION_HOURS = 8
    GRANULARITY_IN_MINUTES = 120

    def __init__(self, path, xls_input_file="jotis_prod.xlsx"):
        super().__init__()      
        # load from xls
        self.mixtures,self.mix2fin,self.fin2mix,self.finals,self.fin2pl,self.reservoirs, self.machines_dict, self.mrc_mixtures_sequence,self.machine_descr2m_id = import_static_information(path + '\\' + 'YIOTIS_ALL_v1_4a.xlsx')
        self.productions, self.parameters = import_production_information(path + '\\' + xls_input_file)
        self.mixtures_quantities = self.gen_mixture_quantities()

        if 'Horizon start' in self.parameters:
            self.schedule_start_dt = self.parameters.get('Horizon start')
        else:
            self.schedule_start_dt: datetime = datetime.strptime(
                "2023-01-09 06:00", "%Y-%m-%d %H:%M"
            )  # Monday
        if 'Horizon end' in self.parameters:
            self.schedule_end_dt = self.parameters.get('Horizon end')
        else:
            self.schedule_end_dt: datetime = datetime.strptime(
                "2023-01-13 22:00", "%Y-%m-%d %H:%M"
            )  # Friday
        
        #Generate shifts
        self.work_periods, self.no_work_periods = gen_shifts(self.schedule_start_dt,self.schedule_end_dt)
    
        #Perform relaxation
        self.schedule_start_dt = self.schedule_start_dt - timedelta(days=1)
        self.schedule_end_dt = self.schedule_end_dt + timedelta(hours=8)

        self.set_defaults()
        self.set_energy_sources_specs(
            number_of_energy_markets=1
        )
        self.gen_products(number_of_in_products=len(self.finals))
        self.set_jobs_specs(
            number_of_jobs=len(self.productions)
        )    
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
        self.parameters.resources["Operators"] = Resource(
            name="Operators number",
            measurement_unit="person",
            measurement_type=ResourceType.DISCRETE
        )
        self.set_optional_parameters_specs()
        self.set_core_specs(
            start_time=self.schedule_start_dt,
            finish_time=self.schedule_end_dt
        )
        self.set_attribute_specs(number_of_attributes=len(self.mixtures))
        self.fix_feasibility = False
    
    def set_attribute_specs(self, number_of_attributes):
        self.attributes = {}
        for attrib_id,attrib in self.mixtures.items():
            self.attributes[
                attrib_id
            ] = Attribute(attribute_id=attrib_id,description=attrib[0])
        for attrib_id,attrib in self.finals.items():
            self.attributes[
                attrib_id
            ] = Attribute(attribute_id=attrib_id,description=attrib)
        self.number_of_attributes = len(self.attributes.keys())

        self.attributes_specs = True
    
    '''
        For the JOTIS case, for each machine the in product and the out product are associated 1:1 ???
    '''
    def gen_products(self,number_of_in_products=0):
        self.in_products={}
        self.out_products={}

        for m_id,v in self.mixtures.items():
            mix_id = int(m_id)
            self.in_products[mix_id] = Product(
                id = int(mix_id),
                name=v[0]
            )
            #The in and out products are mixtures for the previous phases before the final one
            self.out_products[mix_id] = self.in_products[mix_id]

        for f_id,v in self.finals.items():
            final_id = int(f_id)
            
            if final_id in self.fin2pl:
                gr_per_pcs = self.fin2pl[f_id][4][2]
                weight = gr_per_pcs
            else:
                weight = 0

            self.out_products[final_id] = Product(
                id = final_id,
                name=v,
                weight=weight,
            )
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
        self.machines = []
        for m_id,m_rec in self.machines_dict.items():
            # create opm and setup for each machine
            # TODO: Single Operation mode. Is it correct??
            op_dict={}
            op_dict[0] = MachineOperationalMode(id=0)

            #TODO: is it correct?? no specific setup

            # all machines can process one task at a time expect the "wait" virtual machine 
            if m_rec[3] == -1:
                capacity = 100
            else:
                capacity = 1

            machine = Machine(m_id,name=str(m_rec[1]),stage=m_rec[0],capacity=capacity,operational_modes=op_dict,)

            #create list of what attributes the machine can process
            if m_id in self.mrc_mixtures_sequence:
                machine.can_process = copy.deepcopy(self.mrc_mixtures_sequence[m_id])
            # elif m_id in self.

            self.machines.append(machine)                        
            factory.add_machine(machine)
        self.machines_specs = True

    def set_jobs_specs(
        self,
        number_of_jobs
    ):
        self.jobs = {}
        self.productSequences={}
        j_id=0
        self.gen_mixture_jobs(j_id)
        j_id = len(self.jobs)       
        self.gen_dosage_packaging_jobs(j_id)

        self.number_of_jobs = j_id
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
        for j_id,job in self.jobs.items():
            variable_tasks_ru = {}
            prod_code = int(job.product_name)
            
            #check if it is a mixture or a final product
            if prod_code in self.mixtures:
                m_id = self.mixtures[prod_code][1]
                machine = self.machines[m_id]
                stage_0 = "Refiner-Conche"
                production_type = -1
                variable_tasks_ru[m_id]=[0,[]]

                for t_id,task in job.tasks.items():
                    
                    if task.attribute_id not in machine.can_process:
                        print(f'Task({j_id}/{t_id}) attempted to be processed by Machine({m_id} which can not process it, skipping!!!')
                        continue
                    if task.name == "Conching Setup":
                        #TODO: check with Yiotis if there is any consumption during loading
                        time = int(self.machines_dict[m_id][5]*60)
                        consumption_idle_min = self.machines_dict[m_id][10]/60.0
                        electricity = consumption_idle_min*time
                        operators = self.machines_dict[m_id][5]
                        usage = FixedTimeTaskResourceUseProperties(
                            job_id=j_id,task_id=t_id,
                            time=time,
                            consumptions=[
                                ResourceUse(
                                    resource_name= "Electricity",
                                    consumption = electricity
                                ),
                                ResourceUse(
                                    resource_name= "Operator",
                                    consumption = float(operators)
                                )
                            ]
                        )
                    elif task.name == "Conching Unload Waiting":
                        #TODO: check with Yiotis if there is any consumption during waiting period
                        consumption_wait_min = self.machines_dict[m_id][9]/60.0
                        usage = VariableTimeTaskResourceUseProperties(
                            job_id=j_id,task_id=t_id,
                            min_time=0,
                            max_time=24*60, #TODO: talk with Yiotis Allow 1 day???
                            consumptions_per_min=[
                                ResourceUse(
                                    resource_name= "Electricity",
                                    consumption = consumption_wait_min
                                )
                            ]
                        )
                    elif task.name == "Conching Unload":
                        #TODO: check with Yiotis if there is any consumption during unloading
                        time = int(self.machines_dict[m_id][6]*60)
                        consumption_wait_min = self.machines_dict[m_id][9]/60.0
                        electricity = consumption_wait_min*time
                        operators = self.machines_dict[m_id][5]
                        usage = FixedTimeTaskResourceUseProperties(
                            job_id=j_id,task_id=t_id,
                            time=time,
                            consumptions=[
                                ResourceUse(
                                    resource_name= "Electricity",
                                    consumption = electricity
                                ),
                                ResourceUse(
                                    resource_name= "Operator",
                                    consumption = float(operators)
                                )
                            ]
                        )
                    else:
                       #TODO: check with Yiotis how much is the consumption
                        time = int(self.mixtures[task.attribute_id][2]*60)
                        consumption_min = consumption_wait_min = self.machines_dict[m_id][8]/60.0
                        electricity = consumption_min*time
                        operators = self.machines_dict[m_id][5]
                        quantity = self.machines_dict[m_id][3]*1000 # tn to kg
                        usage = FixedTimeTaskResourceUseProperties(
                            job_id=j_id,task_id=t_id,
                            time=time,
                            consumptions=[
                                ResourceUse(
                                    resource_name= "Electricity",
                                    consumption = electricity
                                ),
                                ResourceUse(
                                    resource_name= "Operator",
                                    consumption = float(operators)
                                )
                            ],
                            production=[(task.attribute_id,time,quantity)]
                        )
                    if isinstance(usage,VariableTimeTaskResourceUseProperties):
                        variable_tasks_ru[m_id][1].append(usage)
                    else:
                        variable_tasks_ru[m_id][0] += usage.time
                
                    #TODO: Is the consumption the same on all machines and operational modes???
                    for opm in machine.operational_modes:
                        machine.add_machine_use(usage, opm)
            elif prod_code in self.finals:
                m_id = self.fin2pl[0]
                machine = self.machines[m_id]
                stage_0 = "Refiner-Conche"
                m_id_1 = self.fin2pl[1]
                stage_1 = "Refiner-Conche"
                m_id_2 = self.fin2pl[2]  
                stage_2 = "Refiner-Conche" 
                production_type = self.fin2pl[3] 
            else:
                m_id = -1
    
    def generate_initial_solution(self,problem:Problem):
        sol = None
        # sol = Solution(id=1,problem_id="1",generated_date=datetime.now(),schedule=[])
        # sol.fix_after_load_from_json(problem)
        # machine_schedule = {}
        # for m in self.machines:
        #     if m.capacity == 1:
        #           machine_schedule[m.id] = [0,[]]
        # for j,job in self.jobs.items():
        #     if job.tasks[0].name == "Conditioning":
        #         setup_task = job.tasks[1]
        #     else:
        #         setup_task = job.tasks[0]
            
        #     cand_m = []
        #     for m_id,low,high in self.initialSolutionComponents[j,setup_task.id][1]:
        #         start_time = datetime_to_int(self.start_time,self.initialSolutionComponents[j,setup_task.id][0])
        #         if machine_schedule[m_id][0]<start_time:
        #             cand_m.append(m_id)
        #     if len(cand_m)==0:
        #         min_diff = 10E10
        #         cand_m.append(-1)
        #         for m_id,low,high in self.initialSolutionComponents[j,setup_task.id][1]:
        #             start_time = datetime_to_int(self.start_time,self.initialSolutionComponents[j,setup_task.id][0])
        #             diff = machine_schedule[m_id][0] - start_time 
        #             if diff<min_diff:
        #                 cand_m[0]=m_id
        #     job_m = random.choice(cand_m)

        #     # self.initialSolutionComponents[(j_id)]=[planned_chamber_start_time,planned_finish_time,duration,in_chamber_total_time]   
        #     duration = self.initialSolutionComponents[j][2]   
        #     in_chamber_total_time = self.initialSolutionComponents[j][3]  
        #     available_variable_time = duration - in_chamber_total_time  
        #     job_time = max(machine_schedule[m_id][0], datetime_to_int(self.start_time, self.initialSolutionComponents[j][0]))
        #     for t,task in job.tasks.items():
        #         if task.name == "Conditioning":
        #             continue
        #         for m_id,low,high in self.initialSolutionComponents[j,t][1]:
        #             if m_id == job_m:
        #                 if low == high:
        #                     duration = low
        #                 else:
        #                     max_avail = min(available_variable_time,high)
        #                     duration = random.randint(low,max_avail)
        #                     available_variable_time -= duration
        #                 sol.schedule_task_to_machine(j,t,0,m_id,0,job_time,duration,0,0,[])
        #                 machine_schedule[m_id][1].append((j,t,job_time,duration))
        #                 job_time += duration
        #                 machine_schedule[m_id][0]=job_time
                
        self.initial_solution = sol

         
    def gen_mixture_quantities(self):
        required_mixture_quantities = {}
        for f_id,mix_id,quantity in self.productions.values():
            if mix_id not in required_mixture_quantities:
                required_mixture_quantities[mix_id] = 0
            if f_id in self.fin2pl:
                prod_type = self.fin2pl[f_id][3]
                #TODO: Check with JOTIS, we do not have gr/pcs values THIS IS NOT CORRECT
                # if prod_type in [1,2,5]:
                #     gr_per_pcs = self.fin2pl[f_id][4][2]
                #     mix_quantity = int((quantity * gr_per_pcs)/1000)+1
                # else:
                #     gr_per_pcs = self.fin2pl[f_id][5][2]
                #     mix_quantity = int((quantity * gr_per_pcs)/1000)+1
                gr_per_pcs = self.fin2pl[f_id][4][2]
                mix_quantity = int((quantity * gr_per_pcs)/1000)+1
            else:
                if f_id in self.finals:
                    print(f"Invalid final product ({f_id}) requested - No production process defined")
                else:
                    print(f"Final product ({f_id}) production requested but is not defined in active final products - skipping production process")
            required_mixture_quantities[mix_id]+= mix_quantity
        return required_mixture_quantities
    
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
        for s_id, (s_name, s_val) in enumerate(self.reservoirs.items()):
            eligible = [str(x) for x in list(dict(filter(lambda elem: s_name in elem[1][4], self.mixtures.items())).keys())]
            #skip unused storage
            if len(eligible)==0:
                continue
            a_storage = ProductStorage(id=s_id, name=s_name, capacity=s_val[1], min_value=0, max_value=s_val[1],eligible_products=eligible)
            factory.add_storage(a_storage)
        self.storages_specs = True
    

    def gen_mixture_jobs(self,j_id):
        #Generate the mixture generation tasks
        for mix_id, quantity in self.mixtures_quantities.items():
            task_list = []
            description, mrc_id, mrc_hours, network, reservoirs = self.mixtures[mix_id]
            stage,descr,capacity,capacity_used_tn,comments,setup,unload,operators,consumption_working,consumption_waiting,consumption_idle = self.machines_dict[mrc_id]

            earliest_start_time = self.start_time
            latest_finish_time = self.finish_time

            conching_tasks_number = conches(quantity,capacity_used_tn*1000)
            out_products = [(mix_id, capacity_used_tn*1000)]

            minimum_total_time=0
            
            t_id=0                
               
            for i in range(conching_tasks_number):
                if setup>0:
                    task_list.append(
                            Task(id=t_id,job_id=j_id,name="Conching Setup", 
                                earliest_start_time= earliest_start_time+timedelta(minutes=minimum_total_time),
                                attribute_id=mix_id,
                                )
                    )
                    minimum_total_time += setup*60
                    t_id+=1
            
                a_task = Task(id=t_id,job_id=j_id,name=str(mix_id)+"_#"+str(i),
                              attribute_id=mix_id,
                              earliest_start_time=earliest_start_time+timedelta(minutes=minimum_total_time),
                              out_products=out_products
                              )
                minimum_total_time += mrc_hours*60
                task_list.append(a_task)
                t_id+=1

                # Generate 2 tasks, one that is the wait time for the unloading
                # The actual unloading which allocates also a network and moves the product quantity to the reservoir
                if unload>0:
                    task_list.append(
                        Task(id=t_id,job_id=j_id,name="Conching Unload Waiting", 
                            earliest_start_time= earliest_start_time+timedelta(minutes=minimum_total_time),
                            attribute_id=mix_id,
                            )
                    )
                    t_id+=1

                    task_list.append(
                        Task(id=t_id,job_id=j_id,name="Conching Unload", 
                            earliest_start_time= earliest_start_time+timedelta(minutes=minimum_total_time),
                            attribute_id=mix_id,
                            operation_type=ProcessingType.SYNCHRONOUS_MACHINES
                            )
                    )
                    minimum_total_time += unload*60
                    t_id+=1
            
            #Create dependencies between tasks
                        
            task_dict={}
            dep_list=[]
            task_dict[task_list[0].id] = task_list[0]
            #TODO make more generic dependencies in the future
            for t0, t1 in zip(task_list, task_list[1:]):
                task_dict[t1.id]=t1
                min_time = 0
                max_time = int(60*3*24) #Allow up to 3 days for
                consumption=0 #TODO: Check with JOTIS if what is the transition energy consumption
                
                use_same_machine = True
                keep_together = True           
                
                if t0.name!="Conching Unload":
                    max_time=0 #TODO: Check with JOTIS the time between unload and setup of conching

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
            
            # Impose no restrictions in the start/end time of a job
            j_est = earliest_start_time
            j_lft = latest_finish_time
            a_job = Job(id=j_id,
                        name="Mixture_"+str(mix_id),
                        product_name=str(mix_id),
                        earliest_start_time=j_est,
                        latest_finish_time=j_lft,
                        tasks=task_dict,
                        task_dependencies=dep_list,
                        # duration_limits=[duration,duration]               
                )
            a_job._est = datetime_to_int(self.start_time, j_est)
            a_job._lft = datetime_to_int(self.start_time, j_lft)
            self.jobs[a_job.id]=a_job

            if len(self.mrc_mixtures_sequence[mrc_id])>1:
                if mrc_id not in self.productSequences:
                    self.productSequences[mrc_id]=ProductSequence(
                        product_id=mrc_id,
                        job_ids=[]
                    )
                    self.productSequences[mrc_id].job_ids.append(a_job.id)
            j_id+=1
        
        # Sort product sequences by mrc sequence 
        for mrc_id in self.productSequences:
            tmp=[]
            for s_mix_id in self.mrc_mixtures_sequence[mrc_id]:
                for j_id in self.productSequences[mrc_id].job_ids:
                    if self.jobs[j_id].product_name == str(s_mix_id):
                        tmp.append(j_id)
            self.productSequences[mrc_id].job_ids = tmp
    
    def gen_dosage_packaging_jobs(self, j_id):
        # Generate the dosage / packaging jobs

        for id,product in self.productions.items():
            final_id = product[0]

            #Skip the product if not production method is not available
            if final_id not in self.fin2pl:
                print (f'Product {final_id} requested to be produced but no production method is available, skipping')
                continue

            mixture_id = product[1]
            quantity = product[2]
            final = self.out_products[final_id]


            #Calculate tighter earliest time
            earliest_start_time = self.schedule_start_dt
            latest_finish_time = self.schedule_end_dt
                        
            task_list = []

            final_name = self.finals[final_id]
            if final_id in self.fin2pl:
                (dosage_name,packaging_1_name,packaging_2_name, production_type, (dosage_package1_kg_shift,dosage_package1_pcs_shift,dosage_package1_gr_pcs), (package2_kg_shift,package2_pcs_shift,package2_gr_pcs)) = self.fin2pl[final_id]
            else:
                final_production_line = None
                        
            if mixture_id in self.fin2mix[final_id]:
                mixture_description, mrc_id, mrc_hours, network, reservoirs_for_mix = self.mixtures[mixture_id]
            
            mix_quantity = int((quantity * dosage_package1_gr_pcs)/1000)+1
            shifts_needed_dp1 = (mix_quantity / dosage_package1_kg_shift)
            production_rate = dosage_package1_kg_shift/(self.SHIFT_DURATION_HOURS*60)
            
            final_tasks_needed = self.split_shifts2(shifts_needed_dp1)
            
            # full_shifts_needed_dp1 = int(shifts_needed_dp1)
            # extra_time_needed_dp1_in_min = int((shifts_needed_dp1 - full_shifts_needed_dp1)*8*60)+1
            # task_number = full_shifts_needed_dp1
            
            # if extra_time_needed_dp1_in_min>0:
            #     task_number = task_number + 1

            #TODO: check with JOTIS if the processing in packaging is continuous for a single final (non preemptive)
            
            j_id = len(self.jobs)

            for t_id,duration in enumerate(final_tasks_needed):
                in_products=[(mixture_id,production_rate*duration)]
                out_products=[(final_id,production_rate*duration)]
                a_task = Task(id=t_id,job_id=j_id,name=f'M{mixture_id}_F{final_id}_{t_id}',
                                attribute_id=final_id, 
                                earliest_start_time=earliest_start_time,
                                in_products=in_products,
                                out_products=out_products
                                )
        #         ru = self.all_consumptions[test_name]
        #         if isinstance(ru, VariableTimeTaskResourceUseProperties):
        #             #TODO: we have a variable time task, skip and allow next tasks to start earlier than what they really can start
        #             print("Variable time task: ["+test_name+"] as task("+str(j_id)+","+str(t_id)+")")
        #             has_variable_time_task = True
        #             num_of_variable_time_tasks+=1
        #         else:
        #             in_chamber_total_time+=self.all_consumptions[test_name].time               
        #         task_list.append(a_task)
        #         t_id+=1


        #     task_dict={}
        #     dep_list=[]
        #     task_dict[task_list[0].id] = task_list[0]
        # #     #TODO make more generic dependencies in the future
        #     for t0, t1 in zip(task_list, task_list[1:]):
        #         task_dict[t1.id]=t1
        #         min_time = 0
        #         max_time = int(diff)
        #         consumption=0 #TODO: Check with AVL if what is the transition energy consumption
        #         if t0.name=="Conditioning":
        #             use_same_machine = False
        #             keep_together = True
        #             max_time=0
        #         else:
        #             use_same_machine=True
        #             keep_together=True
        #         td = DependencyItem(
        #             task_id1=t0.id,
        #             task_id2=t1.id,
        #             min_time=min_time,
        #             max_time=max_time,
        #             use_same_machine=use_same_machine,
        #             keep_together=keep_together,
        #             consumptions=[
        #                     ResourceUse(
        #                         resource_name= "Electricity",
        #                         consumption = consumption
        #                     )
        #                     ]
        #             )
        #         dep_list.append(td)
            
        #     # if conditioning_start_time < self.start_time:
        #     j_est = earliest_start_time
        #     # else:
        #     #     j_est = conditioning_start_time
        #     j_lft = latest_finish_time
        #     a_job = Job(id=j_id,
        #                 name=name,
        #                 product_name=vehicle.name,
        #                 earliest_start_time=j_est,
        #                 latest_finish_time=j_lft,
        #                 tasks=task_dict,
        #                 task_dependencies=dep_list,
        #                 duration_limits=[duration,duration]               
        #         )
        #     a_job._est = datetime_to_int(self.start_time, j_est)
        #     a_job._lft = datetime_to_int(self.start_time, j_lft)
        #     self.jobs[a_job.id]=a_job

        #     if len(self.vehicle_data[vehicle.name])>1:
        #         if vehicle.id not in self.productSequences:
        #             self.productSequences[vehicle.id]=ProductSequence(
        #                 product_id=vehicle.id,
        #                 job_ids=[]
        #             )
        #         self.productSequences[vehicle.id].job_ids.append(a_job.id)
        #     j_id+=1
        
        # #Generate maintenance jobs for each machine????
        # # for f in 

    # returns a list of time intervals (e.g. 2.5 shifts -> [480, 480, 240] in minutes)
    def split_shifts(self, duration_in_shifts):
        duration_in_minutes = duration_in_shifts * self.SHIFT_DURATION_HOURS * 60
        full_shifts = int(duration_in_minutes / (self.SHIFT_DURATION_HOURS * 60))
        time_intervals = [self.SHIFT_DURATION_HOURS * 60] * full_shifts
        if full_shifts * self.SHIFT_DURATION_HOURS * 60 < duration_in_minutes:
            time_intervals.append(
                int(
                    duration_in_minutes
                    - full_shifts * Problem.SHIFT_DURATION_HOURS * 60
                )
            )
        return time_intervals

    # returns a list of time intervals based on GRANULARITY_IN_MINUTES
    # (e.g., for GRANULARITY_IN_MINUTES=60 1.6 shifts = 768 minutes-> 12 x 60minutes + 1 x 48minutes)
    def split_shifts2(self, duration_in_shifts):
        duration_in_minutes = duration_in_shifts * self.SHIFT_DURATION_HOURS * 60
        splits = int(duration_in_minutes / self.GRANULARITY_IN_MINUTES)
        time_intervals = [self.GRANULARITY_IN_MINUTES] * splits
        if splits * self.GRANULARITY_IN_MINUTES < duration_in_minutes:
            time_intervals.append(
                int(duration_in_minutes - splits * self.GRANULARITY_IN_MINUTES)
            )
        return time_intervals

    def finish_jobs_needed_per_final(self, frontend_production):
        self.backend_production = []
        self.frontend_production = []
        previous_mixture_id = None
        for (
            mixture_id,
            final_id,
            mrc_id,
            mrc_time,
            tank_id,
            tempering_network_id,
        ) in frontend_production:
            (
                production_type,
                dosage_machine_id,
                package1_machine_id,
                dosage_package1_kg_shift,
                dosage_package1_gr_per_piece,
                package2_machine_id,
                package2_kg_shift,
                package2_gr_per_piece,
            ) = self.fin_production_line[final_id]
            if mixture_id != previous_mixture_id:
                idx = 1
                previous_mixture_id = mixture_id
            else:
                idx += 1
            frontend_task_id = f"FE_{mixture_id}_{final_id}_{mrc_id}_b{idx}"
            qty = self.get_mrc_capacity(mrc_id)
            # if production_type not in (1, 2, 3, 4):  # παραδοχή 5
            if production_type not in (1, 2, 3):
                continue
            if production_type == 1 or production_type == 2:
                backend_task_id = f"BE{frontend_task_id[2:]}"
                shifts = qty / dosage_package1_kg_shift
                if production_type == 1:
                    type_label = "TYPE1:D"
                    machines = [dosage_machine_id]
                else:
                    type_label = "TYPE2:DP"
                    machines = [dosage_machine_id, package1_machine_id]
                backend_production_record = (
                    backend_task_id,
                    type_label,
                    final_id,
                    machines,
                    int(shifts * Problem.SHIFT_DURATION_HOURS * 60),
                    [],
                )
                total = 0
                for i, time_interval in enumerate(self.split_shifts2(shifts)):
                    outflow_qty = math.ceil(
                        dosage_package1_kg_shift
                        / (Problem.SHIFT_DURATION_HOURS * 60)
                        * time_interval
                    )
                    backend_production_record[5].append(
                        (f"{backend_task_id}_{i+1}", time_interval)
                    )
                    self.reservoirs_out_dict[tank_id].append(
                        (f"{backend_task_id}_{i+1}", outflow_qty)
                    )
                    total += outflow_qty
                # adjust the quantity of the last subtask so as the total quantity
                # of the subtasks to match the quantity of the umbrella task
                self.reservoirs_out_dict[tank_id][-1] = (
                    f"{backend_task_id}_{i+1}",
                    outflow_qty - (total - qty),
                )
                assert backend_production_record[4] == sum(
                    x[1] for x in backend_production_record[5]
                )
                self.backend_production.append(backend_production_record)
            elif production_type == 3:
                # DOSAGE
                backend_task_id = f"BE{frontend_task_id[2:]}_D"
                shifts = qty / dosage_package1_kg_shift
                backend_production_record = (
                    backend_task_id,
                    "TYPE3:D",
                    final_id,
                    [dosage_machine_id],
                    int(shifts * Problem.SHIFT_DURATION_HOURS * 60),
                    [],
                )
                total = 0
                for i, time_interval in enumerate(self.split_shifts2(shifts)):
                    outflow_qty = math.ceil(
                        dosage_package1_kg_shift
                        / (Problem.SHIFT_DURATION_HOURS * 60)
                        * time_interval
                    )
                    backend_production_record[5].append(
                        (f"{backend_task_id}_{i+1}", time_interval)
                    )
                    self.reservoirs_out_dict[tank_id].append(
                        (f"{backend_task_id}_{i+1}", outflow_qty)
                    )
                    total += outflow_qty
                # adjust the quantity of the last subtask so as the total quantity
                # of the subtasks to match the quantity of the umbrella task
                self.reservoirs_out_dict[tank_id][-1] = (
                    f"{backend_task_id}_{i+1}",
                    outflow_qty - (total - qty),
                )
                self.backend_production.append(backend_production_record)

                # PACKAGE
                backend_task_id2 = f"BE{frontend_task_id[2:]}_P"
                shifts = qty / package2_kg_shift
                backend_production_record2 = (
                    backend_task_id2,
                    "TYPE3:P",
                    final_id,
                    [package2_machine_id],
                    int(shifts * Problem.SHIFT_DURATION_HOURS * 60),
                    [],
                )
                for i, time_interval in enumerate(self.split_shifts2(shifts)):
                    backend_production_record2[5].append(
                        (f"{backend_task_id2}_{i+1}", time_interval)
                    )
                self.backend_production.append(backend_production_record2)

                # παραδοχή 6 (όλα της φάσης D θα πρέπει να ολοκληρωθούν πριν ξεκινήσει η φάση P)
                self.order_pairs.append((backend_task_id, backend_task_id2))
            # elif production_type == 4:
            #     before = []
            #     shifts1 = qty / dosage_package1_kg_shift
            #     backend_task_id += 1
            #     for i, time_interval in enumerate(self.split_shifts(shifts1)):
            #         backend_production_record1 = (
            #             f"BE{backend_task_id}_{i+1}",
            #             "TYPE4:DP",
            #             final_id,
            #             [dosage_machine_id, package1_machine_id],
            #             time_interval,
            #         )
            #         before.append(f"BE{backend_task_id}_{i+1}")
            #         backend_production.append(backend_production_record1)
            #         outflow_qty = math.ceil(
            #             dosage_package1_kg_shift
            #             / (Problem.SHIFT_DURATION_HOURS * 60)
            #             * time_interval
            #         )
            #         self.reservoirs_out_dict[tank_id].append(
            #             (f"BE{backend_task_id}_{i+1}", outflow_qty)
            #         )

            #     after = []
            #     shifts2 = qty / package2_kg_shift
            #     backend_task_id += 1
            #     for i, time_interval in enumerate(self.split_shifts(shifts2)):
            #         backend_production_record2 = (
            #             f"BE{backend_task_id}_{i+1}",
            #             "TYPE4:P2",
            #             final_id,
            #             [package2_machine_id],
            #             time_interval,
            #         )
            #         after.append(f"BE{backend_task_id}_{i+1}")
            #         backend_production.append(backend_production_record2)

            #     # παραδοχή 6 (όλα της φάσης DP θα πρέπει να ολοκληρωθούν πριν ξεκινήσει η φάση P2)
            #     for t1 in before:
            #         for t2 in after:
            #             task_order_pairs.append((t1, t2))
            self.frontend_production.append(
                (
                    frontend_task_id,
                    mixture_id,
                    final_id,
                    mrc_id,
                    mrc_time,
                    tank_id,
                    tempering_network_id,
                )
            )
            self.reservoirs_in_dict[tank_id].append(
                (f"{frontend_task_id}", self.get_mrc_capacity(mrc_id))
            )