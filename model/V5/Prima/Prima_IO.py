import os
import logging
import pandas as pd
import openpyxl
from datetime import datetime, timedelta

from ..scenario import Product, Task, Attribute, CostInterval
from ..scenario import VariableTimeTaskResourceUseProperties, Resource, ResourceUse, ResourceType
from ..scenario import MachineOperationalMode

def import_product_transitions(fn: str) -> None:
    df = pd.read_excel(fn)
    transition_dict = dict()
    for _, row in df.iterrows():
        product_id1 = int(row["Product1"])
        product_id2 = int(row["Product2"])
        duration = row["Duration(min)"]
        electricity = row["Electricity(Kwh)"]
        transition_dict[product_id1, product_id2] = (
            product_id1,
            product_id2,
            duration,
            electricity
        )
    return transition_dict
    
def import_products(fn: str)-> dict:
    df = pd.read_excel(fn, na_filter = False)
    all_product_tasks = {}
    all_states = {}
    all_consumptions = {}
    all_products = {}
    all_machine_modes={}

    for _, row in df.iterrows():
        id = row["Id"]
        name = row["Name"]
        duration = row["Duration [sec]"]
        laser_consumption_electrical = row["Laser Consumption[KWh]"]
        machine_consumption_electrical = row["Machine Consumption[KWh]"]
        max_pieces_per_day = row["Maximum Pieces / day"]
        laser_used = row["Laser used"]

        if machine_consumption_electrical == None or machine_consumption_electrical == '':
            machine_consumption_electrical = 0
        
        a_key = f"Product_{id}"
        if not a_key in all_states:
            all_states[a_key] = Attribute(
                attribute_id=id,
                description=a_key,
                state={
                    "Product": int(id),
                    "Laser": str(laser_used)
                }
            )
        attribute_id = all_states[a_key].attribute_id

        if laser_used not in all_machine_modes:
           opm =  MachineOperationalMode(
               id=len(all_machine_modes),
               description="Laser:"+laser_used,
               task_operation=[]
           )
           all_machine_modes[laser_used] = opm

        opm_mode = all_machine_modes[laser_used].id
        
        if id not in all_products:
            product_creation = Task(
            id=-1, job_id=-1,
            name=name,
            attribute_id = attribute_id,
            out_products=[(id,1.0)],
            is_optional=True
            )

            product = Product(
                id = id,
                name = name
            )
            all_product_tasks[id] = product_creation
            all_products[id] = product

        tru = VariableTimeTaskResourceUseProperties(
            job_id=-1,task_id=-1,
            consumptions_per_min=[
                ResourceUse(
                    resource_name= "Electricity",
                    consumption = float(laser_consumption_electrical/(duration/60.))
                        + float(machine_consumption_electrical/(duration/60.))
                )
            ],
            # time to produce product quantity [product_id,time,quantity]
            production= [(id,(duration/60.),1.0)]
        )

        all_consumptions[id, opm_mode] = tru


    return all_product_tasks, all_states, all_consumptions,all_products, all_machine_modes

# def gen_work_periods(no_work_hour_interval=(22, 6), schedule_start_dt=None, schedule_end_dt=None):
#     """no work from 22:00 to 6:00"""
#     work_periods = []
#     current_dt = schedule_start_dt
#     while True:
#         from_dt = current_dt.replace(hour=no_work_hour_interval[1], minute=0)
#         to_dt = current_dt.replace(hour=no_work_hour_interval[0], minute=0)
#         work_periods.append(
#             CostInterval(
#                 from_datetime=from_dt,
#                 to_datetime=to_dt - timedelta(minutes=1),
#                 cost=0.0
#             )
#         )
#         current_dt += timedelta(days=1)
#         if current_dt > schedule_end_dt:
#             break
#     return work_periods

# def get_no_work_periods(no_work_hour_interval=(22, 6),schedule_start_dt=None, schedule_end_dt=None):
#     """no work from 22:00 to 6:00"""
#     no_work_periods = []
#     if schedule_start_dt.hour < no_work_hour_interval[1]:
#         no_work_periods.append(
#             CostInterval(
#                 from_datetime=schedule_start_dt,
#                 to_datetime=schedule_start_dt.replace(
#                     hour=no_work_hour_interval[1], minute=0
#                 ),
#                 cost=1.0
#             )
#         )
#     current_dt = schedule_start_dt
#     while True:
#         from_dt = current_dt.replace(hour=no_work_hour_interval[0], minute=0)
#         to_dt = (current_dt + timedelta(days=1)).replace(
#             hour=no_work_hour_interval[1], minute=0
#         )
#         if from_dt >= schedule_end_dt:
#             break
#         elif to_dt > schedule_end_dt:
#             no_work_periods.append(
#                 CostInterval(
#                     from_datetime=from_dt,
#                     to_datetime=schedule_end_dt - timedelta(minutes=1),
#                     cost=1.0
#                 )
#                 )
#             break
#         no_work_periods.append(
#             CostInterval(
#                     from_datetime=from_dt,
#                     to_datetime=to_dt - timedelta(minutes=1),
#                     cost=1.0
#             )
#         )
#         current_dt += timedelta(days=1)
#     return no_work_periods

def load_orders_from_xlsx(fn_xlsx: str) -> None:
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

    order_data = {}
    product_data = {}
    df = pd.read_excel(fn_xlsx, skiprows=2)
    for index, row in df.iterrows():
        order_id = row.iloc[0]
        product_id = row.iloc[1]
        quantity = row.iloc[2]
        est = row.iloc[3]
        lft = row.iloc[4]
        order_data[order_id] = [
            product_id,
            est,
            lft,
            quantity
        ]
        if product_id not in product_data:
            product_data[product_id]=[order_id]
        else:
            product_data[product_id].append(order_id)
    
    # work_periods =  gen_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    # no_work_periods = get_no_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    
    return order_data, product_data, schedule_start_dt, schedule_end_dt#, work_periods, no_work_periods

   


    