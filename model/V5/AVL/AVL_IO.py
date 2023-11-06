import os
import logging
import pandas as pd
import openpyxl
import copy
from datetime import datetime, timedelta, time

from typing import Any, Tuple, Dict, List, Optional, Union
from pydantic import BaseModel, Field, confloat, conint, PrivateAttr

from ..scenario import AbstractOperation, Attribute, CostInterval, EmissionsGenerated
from ..scenario import FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties, Resource, ResourceUse, ResourceType

#AVL json input data class

class ProblemSuite(BaseModel):
    Suite:str
    Vehicle:str
    CST:str
    EST:datetime
    LFT:datetime
    PST:datetime
    PFT:datetime
    Duration:int
    Temperature:int
    Tests:str
    CoastDown:str
    Testbed:str

class ProblemSuites(BaseModel):
    suites: List[ProblemSuite]

class Test(BaseModel):
    name:str
    conditioning:int
    duration: int
    humidity: float
    temperature:int
    consumption_electrical:float
    consumption_cooling:float
    consumption_heating:float
    co2_emissions:float

class ProblemTests(BaseModel):
    tests: List[Test]

class Transition(BaseModel):
    temperature1:int
    humidity1:int
    temperature2:int
    humidity2:int
    duration:int
    electricity:float
    heating:float
    cooling:float

class ProblemTransitions(BaseModel):
    transitions: List[Transition]

SPECIAL_TESTS = {
    "Chassis Dyno Time",
    "Coast down",
    # "Dismantling time",
    # "Load Adaption",
    # "Set-up time",
    "SpaceVelocityTest",
    "Stillstand",
    "Warm up, Load Adoption"
    #  "Project Delay",
    #  "Maintenance",
}

def read_testbed_transition_xls(fn: str) -> None:
    df = pd.read_excel(fn)
    return df

def read_testbed_transition_json(transitions_json: str) -> None:
    df = pd.read_json(transitions_json)
    return df

def read_testbed_transition_json_str(transitions_json: str) -> None:
    df = pd.read_json(transitions_json)
    return df

def import_testbed_transition(df: pd.DataFrame) -> None:
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

def read_tests_xls(fn: str):
    df = pd.read_excel(fn, na_filter = False)
    return df

def read_tests_json(tests_json: str):
    df = pd.read_json(tests_json)
    return df

def read_tests_json_str(tests_json: str):
    df = pd.read_json(tests_json)
    return df

def import_tests(df: pd.DataFrame):
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
        test_co2_emissions = row["EpT[kg CO2eq]"]
        # test_co2_emissions = ""

        if test_temperature == None or test_temperature=='':
            test_temperature = 23
        if test_humidity == None or test_humidity=='':
            test_humidity = 45
        
        if pd.isna(test_co2_emissions):
            test_co2_emissions = 0
        
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
                ],
                emissions_per_minute=[
                    EmissionsGenerated(
                        emission_name= "CO2",
                        generated_amount=float(test_co2_emissions)
                    )
                ]
            )        
        else:
            tru = FixedTimeTaskResourceUseProperties(
                job_id=-1,task_id=-1,
                time=int(test_duration),
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
                ],
                emissions=[                    
                    EmissionsGenerated(
                        emission_name= "CO2",
                        generated_amount=float(test_co2_emissions)
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
                cost=10.0
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
                    cost=10.0
                )
                )
            break
        no_work_periods.append(
            CostInterval(
                    from_datetime=from_dt,
                    to_datetime=to_dt - timedelta(minutes=1),
                    cost=10.0
            )
        )
        current_dt += timedelta(days=1)
    return no_work_periods

def read_suites_from_xlsx(fn_xlsx: str):
    # workbook = openpyxl.load_workbook(fn_xlsx)
    # worksheet = workbook.active

    # if isinstance(worksheet["B1"].value, datetime):
    # schedule_start_dt = worksheet["B1"].value
    # elif isinstance(worksheet["B1"].value, str):
    #     schedule_start_dt = datetime.strptime(worksheet["B1"].value, "%Y-%m-%d %H:%M:%S")
    
    # if isinstance(worksheet["B2"].value, datetime):
    # schedule_end_dt = worksheet["B2"].value
    # elif isinstance(worksheet["B2"].value, str):
    #     schedule_end_dt = datetime.strptime(worksheet["B2"].value, "%Y-%m-%d %H:%M:%S")

    df = pd.read_excel(fn_xlsx)

    return df

def read_suites_from_json(suites_json: str):
    df = pd.read_json(suites_json,convert_dates=True)
    return df

def read_suites_from_json_str(suites_json: str):
    df = pd.read_json(suites_json,convert_dates=True)
    return df

def load_suites(df: pd.DataFrame):
    suites_data = {}
    vehicle_data = {}

    for index, row in df.iterrows():
        suite_id = row["Suite"]
        tests = row["Tests"]
        temperature = row["Temperature"] 
        if tests in SPECIAL_TESTS:
            tests = f"{tests}#{temperature}"
        if suite_id not in suites_data:
            vehicle_id = str(row["Vehicle"])
            est = row["EST"]
            lft = row["LFT"]
            pst = row["PST"]
            pft = row["PFT"]
            duration = row["Duration"]
            coast_down = row["CoastDown"]
            employee = "NO"
            testbed = row["Testbed"]
            if isinstance(duration,time): 
                duration = duration.hour*60+duration.minute
            if isinstance(est,str):
                est = datetime.strptime(est, "%Y-%m-%d %H:%M:%S")
            if isinstance(lft,str):
                lft = datetime.strptime(lft, "%Y-%m-%d %H:%M:%S")
            if isinstance(pst,str):
                pst = datetime.strptime(pst, "%Y-%m-%d %H:%M:%S")
            if isinstance(pft,str):
                pft = datetime.strptime(pft, "%Y-%m-%d %H:%M:%S")
            #Adapt est / lft to pst and pft
            # Allow up to 2 hours before the planned start time
            if pst<est:
                est = copy.deepcopy(pst)
                est -= timedelta(hours=1)
                est -= timedelta(seconds=est.minute*60)
            # Allow up to 2 hours after the planned finish time
            if pft>lft:
                lft = copy.deepcopy(pft)
                lft += timedelta(hours=2)
                lft -= timedelta(seconds=lft.minute*60)

            suites_data[suite_id] = [
                vehicle_id,
                est,
                lft,
                pst,
                pft,
                duration,
                temperature,
                tests,
                employee,
                testbed
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
            suites_data[suite_id][7] += "," + 'Coast down#'+str(temperature)
        
        if vehicle_id not in vehicle_data:
            vehicle_data[vehicle_id]=[suite_id]
        elif suite_id not in vehicle_data[vehicle_id]:
            vehicle_data[vehicle_id].append(suite_id)
    
    #Find schedule start and end dates
    schedule_start_dt = min([s[1] for s in suites_data.values()])
    schedule_end_dt = max([s[2] for s in suites_data.values()])
    
    work_periods =  gen_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    no_work_periods = get_no_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    
    return suites_data, vehicle_data, schedule_start_dt, schedule_end_dt, work_periods, no_work_periods

def export_suite_data(suites_data):
    ps = ProblemSuites(suites=[])
    for s,sd in suites_data.items():
        tests = sd[7].split(",")
        for i in range(len(tests)):
            test = tests[i]
            if test == 'Coast down':
                continue
            if i+1<len(tests) and tests[i+1] == 'Coast down':
                coast_down = "YES"
            else:
                coast_down = "NO"
            ps.suites.append(
                ProblemSuite(
                    Suite=s,
                    Vehicle=sd[0],
                    CST="",
                    EST=sd[1],
                    LFT=sd[2],
                    PST=sd[3],
                    PFT=sd[4],
                    Duration=sd[5],
                    Temperature=sd[6],
                    Tests=test,
                    CoastDown=coast_down,
                    Testbed=sd[9]
                    )
            )
        
    dict = ps.json(exclude_unset=True, indent=3)
    return dict

def export_test_data(test_data):
    ps = ProblemSuites(suites=[])
    for s,sd in suites_data.items():
        tests = sd[7].split(",")
        for i in range(len(tests)):
            test = tests[i]
            if test == 'Coast down':
                continue
            if i+1<len(tests) and tests[i+1] == 'Coast down':
                coast_down = "YES"
            else:
                coast_down = "NO"
            ps.suites.append(
                ProblemSuite(
                    Suite=s,
                    Vehicle=sd[0],
                    CST="",
                    EST=sd[1],
                    LFT=sd[2],
                    PST=sd[3],
                    PFT=sd[4],
                    Duration=sd[5],
                    Temperature=sd[6],
                    Tests=test,
                    CoastDown=coast_down,
                    Testbed=sd[9]
                    )
            )
        
    dict = ps.json(exclude_unset=True, indent=3)
    return dict

def read_energy_market_from_json(suites_json: str):
    df = pd.read_json(suites_json,convert_dates=True)
    return df

def read_energy_market_from_json_str(suites_json: str):
    df = pd.read_json(suites_json,convert_dates=True)
    return df

def load_energy_market(df: pd.DataFrame) -> None:
    
    predictions_data = []
    for index, (row) in df.iterrows():
        prediction = row["prediction"]
        from_date = prediction["from_datetime"]
        to_date = prediction["to_datetime"]
        value = prediction["value"]
        if isinstance(from_date,str):
            from_date = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
        if isinstance(to_date,str):
            to_date = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
        if isinstance(value,str):
            value = float(value)
        predictions_data.append((from_date,to_date,value))
    return predictions_data
    