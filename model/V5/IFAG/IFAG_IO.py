import os
import logging
import pandas as pd
import openpyxl
from datetime import datetime, timedelta

from ..scenario import AbstractOperation, Attribute, CostInterval
from ..scenario import FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties, Resource, ResourceUse, ResourceType
from ..scenario import Machine, MachineProcessingType, MachineConsumptionAggregation, MachineOperationalMode

def import_static_information(fn:str):
    dict_df = pd.read_excel(fn, 
                        sheet_name=['Toolgroups','Energy_Consumption_Area','Energy_Consumption_Toolgroup',
                                    'Lotrelease - variable due dates','Delivery_Data_WSPW',
                                    'Lithography_Product_Product','Implantation_Product_Product', 
                                    'Route_Product_1', 'Route_Product_2', 'Route_Product_3', 
                                    'Route_Product_4', 'Route_Product_5', 'Route_Product_6', 'Route_Product_7',
                                      'Route_Product_8', 'Route_Product_9', 'Route_Product_10'])
    #import information about machine groups that are available at a factory
    machines = import_toolgroups(dict_df)
    #import recipes for products
    for i in range(1,11):
        read_product_route(dict_df,'Route_Product_'+str(i))

def import_toolgroups(dict_df):
    machines = {}
    toolgroups_df = dict_df.get('Toolgroups')
    head = toolgroups_df.head()
    areas = sorted(toolgroups_df["AREA"].unique())
    m_id = 0
    for area in areas:
        area_toolgroups = toolgroups_df.loc[toolgroups_df["AREA"] == area]
        for index,tg in area_toolgroups.iterrows():
            process_type = MachineProcessingType.SEQUENTIAL
            consumption_aggregation = MachineConsumptionAggregation.PROPORTIONAL
            tg_name = tg["TOOLGROUP"]
            tg_tools_num = tg["NUMBER OF TOOLS"]
            tg_cascading = tg["CASCADINGTOOL"]
            if tg_cascading =="YES":
                process_type = MachineProcessingType.CASCADING
            tg_batching = tg["BACTHINGTOOL"]
            if tg_batching == "YES":
                process_type = MachineProcessingType.BATCHING
                tg_batching_criterion = tg["BATCHCRITERION"]
                tg_batching_unit = tg["BATCHING UNIT"]
                consumption_aggregation = MachineConsumptionAggregation.MAXIMUM
            
            # TODO: Single Operation mode. Is it correct??
            op_dict={}
            op_dict[0] = MachineOperationalMode(id=0)
            
            machine = Machine(id=m_id, name=tg_name,capacity=tg_tools_num,
                          processing_type=process_type,
                          consumption_aggregation=consumption_aggregation,
                          operational_modes=op_dict,
                          stage = area)
            machines[m_id] = machine
            m_id += 1
            print(machine)
    
    return machines


def get_step_details(product_df):
    steps = {}
    for i in range(len(product_df)):
        step_description = product_df.iloc[i]['STEP DESCRIPTION']
        area = product_df.iloc[i]['AREA']
        tool_group = product_df.iloc[i]['TOOLGROUP']
        processing_unit = product_df.iloc[i]['PROCESSING UNIT']
        execution_time = product_df.iloc[i]['MEAN']
        steps[step_description] = (area, tool_group, execution_time, processing_unit)
    return steps


def read_product_route(dict_df,product_route_name):
    processes = {}
    product_df = dict_df.get(product_route_name)




def import_product_product_transition(fn: str) -> None:
    df = pd.read_excel(fn)
    transition_dict = dict()
    for _, row in df.iterrows():
        temperature1 = row["Temperature1"]
        humidity1 = row["Humidity1"]
        temperature2 = row["Temperature2"]
        humidity2 = row["Humidity2"]
        duration = row["Duration"]
        electricity = row["Electricity"]
        heating = row["Heating"]
        cooling = row["Cooling"]
        transition_dict[temperature1, temperature2] = (
            temperature1,
            humidity1,
            temperature2,
            humidity2,
            duration,
            electricity,
            heating,
            cooling,
        )
    return transition_dict
    
def import_tests(fn: str)-> dict:
    df = pd.read_excel(fn, na_filter = False)
    all_resources={}
    all_tests = {}
    all_states = {}
    all_consumptions = {}
    all_conditioning = {}

    for _, row in df.iterrows():
        name = row["Name"]
        test_conditioning = row["Minimum Conditioning time [h]"]
        test_duration = row["Duration [min]"]
        test_humidity = row["Humidity [%]"]
        test_temperature = row["Temp [°C]"]
        test_consumption_electrical = row["Pel absolut [kWh]"]
        test_consumption_cooling = row["QthC [kWh]"]
        test_consumption_heating = row["QthH [kWh]"]

        if test_temperature == None or test_temperature=='':
            test_temperature = 23
        if test_humidity == None or test_humidity=='':
            test_humidity = 45
        
        if name in SPECIAL_TESTS:
            name = f"{name}#{test_temperature}"
        
        #TODO: Talk with AVL about adding this info to the suite file - 29032023 DONE
        # a_key = f"T{test_temperature}_H{test_humidity}"
        a_key = f"T{int(test_temperature)}"
        if not a_key in all_states:
            all_states[a_key] = Attribute(
                attribute_id=len(all_states),
                description=a_key,
                state={
                    "Temperature":float(test_temperature),
                    "Humidity":float(test_humidity),              
                }
            )
        attribute_id = all_states[a_key].attribute_id
        
        if test_duration == -1:
            tru = VariableTimeTaskResourceUseProperties(
                job_id=-1,task_id=-1,
                consumptions_per_min=[
                    ResourceUse(
                        resource_name= "Electricity",
                        consumption = float(test_consumption_electrical)
                    ),
                    ResourceUse(
                        resource_name= "Cooling",
                        consumption = float(test_consumption_cooling)
                    ),
                    ResourceUse(
                        resource_name= "Heating",
                        consumption = float(test_consumption_heating)
                    )
                ]
            )        
        else:
            tru = FixedTimeTaskResourceUseProperties(
                job_id=-1,task_id=-1,
                time=test_duration,
                consumptions=[
                    ResourceUse(
                        resource_name= "Electricity",
                        consumption = float(test_consumption_electrical)
                    ),
                    ResourceUse(
                        resource_name= "Cooling",
                        consumption = float(test_consumption_cooling)
                    ),
                    ResourceUse(
                        resource_name= "Heating",
                        consumption = float(test_consumption_heating)
                    )
                ]
            )

        a_test = AbstractOperation(
            name=name,
            attribute_id = attribute_id
        )

        all_tests[name] = a_test
        all_consumptions[name] = tru
        all_conditioning[name] = test_conditioning

    return all_tests, all_resources, all_states, all_consumptions,all_conditioning

def gen_work_periods(no_work_hour_interval=(22, 6), schedule_start_dt=None, schedule_end_dt=None):
    """no work from 22:00 to 6:00"""
    work_periods = []
    current_dt = schedule_start_dt
    while True:
        from_dt = current_dt.replace(hour=no_work_hour_interval[1], minute=0)
        to_dt = current_dt.replace(hour=no_work_hour_interval[0], minute=0)
        work_periods.append(
            CostInterval(
                from_datetime=from_dt,
                to_datetime=to_dt - timedelta(minutes=1),
                cost=0.0
            )
        )
        current_dt += timedelta(days=1)
        if current_dt > schedule_end_dt:
            break
    return work_periods

def get_no_work_periods(no_work_hour_interval=(22, 6),schedule_start_dt=None, schedule_end_dt=None):
    """no work from 22:00 to 6:00"""
    no_work_periods = []
    if schedule_start_dt.hour < no_work_hour_interval[1]:
        no_work_periods.append(
            CostInterval(
                from_datetime=schedule_start_dt,
                to_datetime=schedule_start_dt.replace(
                    hour=no_work_hour_interval[1], minute=0
                ),
                cost=1.0
            )
        )
    current_dt = schedule_start_dt
    while True:
        from_dt = current_dt.replace(hour=no_work_hour_interval[0], minute=0)
        to_dt = (current_dt + timedelta(days=1)).replace(
            hour=no_work_hour_interval[1], minute=0
        )
        if from_dt >= schedule_end_dt:
            break
        elif to_dt > schedule_end_dt:
            no_work_periods.append(
                CostInterval(
                    from_datetime=from_dt,
                    to_datetime=schedule_end_dt - timedelta(minutes=1),
                    cost=1.0
                )
                )
            break
        no_work_periods.append(
            CostInterval(
                    from_datetime=from_dt,
                    to_datetime=to_dt - timedelta(minutes=1),
                    cost=1.0
            )
        )
        current_dt += timedelta(days=1)
    return no_work_periods

def load_suites_from_xlsx(fn_xlsx: str) -> None:
    workbook = openpyxl.load_workbook(fn_xlsx)
    worksheet = workbook.active

    # if isinstance(worksheet["B1"].value, datetime):
    schedule_start_dt = worksheet["B1"].value
    # elif isinstance(worksheet["B1"].value, str):
    #     schedule_start_dt = datetime.strptime(worksheet["B1"].value, "%Y-%m-%d %H:%M:%S")
    
    # if isinstance(worksheet["B2"].value, datetime):
    schedule_end_dt = worksheet["B2"].value
    # elif isinstance(worksheet["B2"].value, str):
    #     schedule_end_dt = datetime.strptime(worksheet["B2"].value, "%Y-%m-%d %H:%M:%S")

    suites_data = {}
    vehicle_data = {}
    df = pd.read_excel(fn_xlsx, skiprows=2)
    for index, row in df.iterrows():
        suite_id = row.iloc[0]
        tests = row.iloc[8]
        temperature = row.iloc[7] 
        if tests in SPECIAL_TESTS:
            tests = f"{tests}#{temperature}"
        if suite_id not in suites_data:
            vehicle_id = row.iloc[1]
            # est = datetime.strptime(row.iloc[2], "%Y-%m-%d %H:%M:%S")
            # lft = datetime.strptime(row.iloc[3], "%Y-%m-%d %H:%M:%S")
            # pst = datetime.strptime(row.iloc[4], "%Y-%m-%d %H:%M:%S")
            # pft = datetime.strptime(row.iloc[5], "%Y-%m-%d %H:%M:%S")
            est = row.iloc[2]
            lft = row.iloc[3]
            pst = row.iloc[4]
            pft = row.iloc[5]
            duration = row.iloc[6]
            coast_down = row.iloc[9]
            employee = row.iloc[10]
            suites_data[suite_id] = [
                vehicle_id,
                est,
                lft,
                pst,
                pft,
                duration,
                temperature,
                tests,
                employee
            ]
        else:
            #TODO: [CG] Is it correct? 
            if temperature != suites_data[suite_id][6]:
                logging.error(
                    f"Not matching temperatures for tests of the same suite {suite_id}: {temperature} vs {suites_data[suite_id][6]}"
                )
                raise ValueError("Not matching temperatures")
            suites_data[suite_id][7] += "," + tests
        if coast_down == 'YES':
            suites_data[suite_id][7] += "," + 'Coast down'
        if vehicle_id not in vehicle_data:
            vehicle_data[vehicle_id]=[suite_id]
        elif suite_id not in vehicle_data[vehicle_id]:
            vehicle_data[vehicle_id].append(suite_id)
    
    work_periods =  gen_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    no_work_periods = get_no_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    
    return suites_data, vehicle_data, schedule_start_dt, schedule_end_dt, work_periods, no_work_periods

   


    