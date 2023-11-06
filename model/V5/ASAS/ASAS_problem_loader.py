from datetime import datetime,timedelta
import logging
import copy
import random
import pandas as pd

from .ASAS_IO import import_electricity_consumption_and_production_information, import_problem

from ..synthetic_problem_generator import ProblemGeneratorV5
from ..scenario import CostComponent, CostInterval, EnergySource, ResourceUse, Resource, ResourceType
from ..scenario import Factory, ResourceUse, FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties
from ..scenario import Machine, MachineOperationalMode, MachineProcessingType, SetupProperties, MachineConsumptionAggregation
from ..scenario import Job, Task, Attribute, DependencyItem, ProcessingType
from ..scenario import Product, ProductSequence, ProductStorage
from ..scenario import Problem,Solution
from ..scenario import datetime_to_int, int_to_datetime

class ASAS_PG(ProblemGeneratorV5):

    def __init__(self, path, previous_prod_cons = 'data/ASAS/Anodizing Hourly Production Data UoP 2.xlsx', problem_definition="ASAS_Problem_Input_Rev26092023.xlsx"):
        super().__init__()      
        # load from xls
        self.daily_consumptions,self.daily_production = import_electricity_consumption_and_production_information(path + '\\' + previous_prod_cons)
        self.consumption_multiplier = self.calculate_consumption_multiplier()

        self.market_energy_price,self.production,self.parameters, self.generators = import_problem(path + '\\' + 'data/ASAS/'+problem_definition) 
     
        self.set_energy_sources_specs()

        D = self.calculate_hourly_product_and_energy_demand()

        schedule = self.generate_schedule(D)

        print("\t\t\t",end=' ')
        for gen_id in self.generators:
            print(gen_id,end='\t')
        print("Cost")
        for datetime, (combination, cost) in schedule.items():
            print(datetime,end='\t')
            for gen_id in self.generators:
                print(combination[gen_id],end='\t')
            print(round(cost,2))


    '''
        The method calculates the energy consumption of a Solution per task
    '''
    def calculate_consumption_multiplier(self):
        a_sum = 0
        count=0
        for date,dp in self.daily_production.items():
            if date in self.daily_consumptions:
                total_daily_consumption = self.daily_consumptions[date]
            else: # skip if we do not know the consumption
                continue
            daily_sum=0
            for products in dp.values():
                daily_sum+=(products[0]+products[1])*2+(products[2]+products[3])*3+products[4]*4+products[5]*6
            if daily_sum>0:
                a_sum+=total_daily_consumption/daily_sum
                count+=1
        a_sum/=count

        return a_sum
        
    def create_real_energy_cost(self):
        self.market_energy_cost = {}
        last_month_usage_fee = self.parameters['P24'][3]
        last_month_usage_consumption = self.parameters['P61'][3]
        usage_cost_per_kwh = last_month_usage_fee / last_month_usage_consumption
        renewable_tax = self.parameters['P25'][3]

        m_name = "IST_Energy_Market"

        charge_intervals = []

        for date,time_periods in self.market_energy_price.items():
            for (tp_start,tp_end),cost in time_periods.items():
                real_cost_per_kwh = cost/1000 + usage_cost_per_kwh + renewable_tax/1000
                start = datetime.combine(date, tp_start)
                end = datetime.combine(date, tp_end)
                ci = CostInterval(from_datetime=start, to_datetime=end, cost = real_cost_per_kwh)
                charge_intervals.append(ci)
                self.market_energy_cost[start]=real_cost_per_kwh
        a_market = EnergySource(name=m_name, cost_intervals = charge_intervals)
        self.energy_markets[m_name]=a_market
    
    def create_self_production_cost(self):
        self.self_generation_energy_cost = {}
        consumed_gas_per_kwh = self.parameters['P29'][3]
        last_month_natural_gas_price = self.parameters['P26'][3]
        natural_gas_cost = consumed_gas_per_kwh * last_month_natural_gas_price
        operation_cost = self.parameters['P27'][3]

        m_name = "Self_Generation"

        charge_intervals = []

        for date,time_periods in self.market_energy_price.items():
            for (tp_start,tp_end),cost in time_periods.items():
                real_cost_per_kwh = natural_gas_cost + operation_cost
                start = datetime.combine(date, tp_start)
                end = datetime.combine(date, tp_end)
                ci = CostInterval(from_datetime=start, to_datetime=end, cost = real_cost_per_kwh)
                charge_intervals.append(ci)
                self.self_generation_energy_cost[start]=real_cost_per_kwh
        a_market = EnergySource(name=m_name, cost_intervals = charge_intervals)
        self.energy_markets[m_name]=a_market

    def calculate_hourly_product_and_energy_demand(self):

        hourly_product_and_energy_demand = {}
    
        boiler_natural_gas_cost_per_m3 = self.parameters['P28'][3]
        steam_energy_per_kg = self.parameters['P62'][3]
        hot_water_energy_per_kg = self.parameters['P63'][3]
        electric_energy_per_kg = self.parameters['P64'][3]
        energy_to_natural_gas_per_kg = self.parameters['P65'][3]

        for date,dp in self.production.items():
            daily_products=0
            daily_consumption=0
            for (tp_start,tp_end),products in dp.items():
                hourly_products_in_kg=(products[0]+products[1])+(products[2]+products[3])+products[4]+products[5]
                hourly_electric_consumption = ((products[0]+products[1])*2+(products[2]+products[3])*3+products[4]*4+products[5]*6)*self.consumption_multiplier
                start = datetime.combine(date, tp_start)
                end = datetime.combine(date, tp_end)

                hourly_steam_energy = steam_energy_per_kg * hourly_products_in_kg
                hourly_hot_water_energy = hot_water_energy_per_kg * hourly_products_in_kg
                hourly_market_source_electric_energy = electric_energy_per_kg * hourly_products_in_kg
                hourly_boiler_natural_gas_in_m3 = (hourly_steam_energy+hourly_hot_water_energy)/energy_to_natural_gas_per_kg

                hourly_cost_no_self_generation = (hourly_boiler_natural_gas_in_m3 * boiler_natural_gas_cost_per_m3 + 
                                                        hourly_electric_consumption * self.market_energy_cost[start])
                
                profitability_market_per_kg = hourly_cost_no_self_generation / hourly_products_in_kg

                hourly_self_generation_electric_energy_cost = hourly_electric_consumption * self.self_generation_energy_cost[start]

                profitability_self_generation_per_kg = hourly_self_generation_electric_energy_cost / hourly_products_in_kg

                hourly_product_and_energy_demand[start] = (hourly_products_in_kg, hourly_electric_consumption, profitability_market_per_kg,profitability_self_generation_per_kg)
                
                daily_products += hourly_products_in_kg
                daily_consumption+=hourly_electric_consumption

        return hourly_product_and_energy_demand
    
    def generate_schedule(self,hourly_product_and_energy_demand):
        schedule = {}
        for date_time, (hourly_products_in_kg, hourly_electric_consumption, profitability_market_per_kg,profitability_self_generation_per_kg) in hourly_product_and_energy_demand.items():
            #D2
            date = date_time.date()
            day_start = pd.Timestamp(year=date_time.year, month=date_time.month, day=date_time.day)
            start = date_time.time()
            end = (date_time + timedelta(hours=1)).time()
            market_price = self.market_energy_price[day_start][(start,end)]/1000
            self_generation_cost = self.self_generation_energy_cost[date_time]
            if market_price > self_generation_cost:
                state={}
                total_generation_energy = 0 
                for gen_id,generator in self.generators.items():
                    max_generation_energy_mode = 0
                    max_generation_energy=0
                    for mode,properties in generator.items():
                       if properties[0]>max_generation_energy:
                           max_generation_energy_mode = mode
                           max_generation_energy = properties[0]
                    total_generation_energy += max_generation_energy
                    state[gen_id] = mode
                surplus_energy = hourly_electric_consumption - total_generation_energy
                cost = (self_generation_cost - market_price)*surplus_energy
            else:
                #D3
                if hourly_electric_consumption == 0: # stop everything
                    state={}
                    for gen_id,generator in self.generators.items():
                        state[gen_id] = 0
                    cost = 0
                else:                    
                    if profitability_market_per_kg<profitability_self_generation_per_kg:
                        #D4 True
                        #Operate the boiler:
                        state={}
                        for gen_id,generator in self.generators.items():
                            if gen_id != "Boiler":
                                state[gen_id] = 0
                            else:
                                state[gen_id] = 1
                        cost = hourly_products_in_kg*profitability_market_per_kg
                    else:
                        #D4 False
                        #Find the minimal cost combination
                        self_generation_energy_cost = self.self_generation_energy_cost[date_time]
                        market_energy_cost = self.market_energy_cost[date_time]
                        required_energy = hourly_electric_consumption
                        boiler_natural_gas_cost_per_m3 = self.parameters['P28'][3]
                        boiler_operation_cost = self.generators["Boiler"][1][5]*boiler_natural_gas_cost_per_m3
                        combs = self.generate_combinations()
                        gens_state,cost = self.calculate_generators_combinations_cost(combs,self_generation_energy_cost,
                                                                                 required_energy,hourly_products_in_kg,
                                                                                 profitability_market_per_kg,boiler_operation_cost,
                                                                                 market_energy_cost)
                        state={}
                        for gen_id,mode in gens_state:
                            state[gen_id] = mode    
                        for gen_id in self.generators:
                            if gen_id not in state:
                                state[gen_id] = 0             

            schedule[date_time]=(state,cost) 
        
        return schedule

    def generate_combinations(self):
        gen_mode = []
        for gen_id,generator in self.generators.items():
            if gen_id == "Boiler":
                continue
            else:
                modes = list(generator.keys())
                gen_mode.append((gen_id,modes))
        
        import itertools
        combinations_list = []
        for g_size in range(1,len(gen_mode)+1):
            combinations = list(itertools.combinations(gen_mode, g_size))
            for gen_id_list in combinations:
                if len(gen_id_list) == 1:
                    gen_id = gen_id_list[0][0]
                    for mode in gen_id_list[0][1]:
                        combinations_list.append([(gen_id,mode)])
                elif len(gen_id_list) == 2:
                    list1 = gen_id_list[0][1]
                    list2 = gen_id_list[1][1]
                    lcb = list(itertools.product(list1, list2))
                    for comb in lcb:
                        combinations_list.append([(gen_id_list[0][0],comb[0]),(gen_id_list[1][0],comb[1])])
                else:
                    pass
        
        return combinations_list

    def calculate_generators_combinations_cost(self,combinations_list, self_generation_energy_cost, 
                                               required_energy,hourly_products_in_kg,
                                               profitability_market_per_kg, boiler_operation_cost,
                                               market_energy_cost):
        combination = None
        combination_cost = 10E20
        for comb in combinations_list:
            cost = 0
            generated_elec = 0
            use_boiler = False 
            for gen_id,mode in comb:
                gen_electricity = self.generators[gen_id][mode][0]
                generated_elec += gen_electricity
                gen_cost = gen_electricity * self_generation_energy_cost
                cost += gen_cost
            if generated_elec < required_energy: #TODO: check with ASAS this is not correct
                # perc = (required_energy - generated_elec)/required_energy
                # prod_using_external = perc*hourly_products_in_kg
                # extra_cost= prod_using_external * profitability_market_per_kg
                
                #Add boiler cost
                extra_electricity_cost = (required_energy - generated_elec) *market_energy_cost
                extra_cost = boiler_operation_cost + extra_electricity_cost          
                cost+= extra_cost
                use_boiler = True
            
            if cost < combination_cost:
                combination = copy.deepcopy(comb)
                combination_cost = cost
                if use_boiler:
                    combination.append(("Boiler",1))
        
        return combination, combination_cost
        

    #Compatibility methods

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
        self.number_of_energy_markets = 1
        self.generate_energy_market()
        self.generate_local_sources()
        self.energy_sources_specs = True

    def generate_energy_market(
        self
    ):  
       self.create_real_energy_cost()
       self.create_self_production_cost()
    
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
