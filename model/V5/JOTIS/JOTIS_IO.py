from copy import deepcopy
import os
import logging
import pandas as pd
import openpyxl
from datetime import datetime, timedelta

from ..scenario import AbstractOperation, Attribute, CostInterval
from ..scenario import FixedTimeTaskResourceUseProperties, VariableTimeTaskResourceUseProperties, Resource, ResourceUse, ResourceType

def import_static_information(fn:str):
    dict_df = pd.read_excel(fn, sheet_name=['mixtures', 'finals', 'production_lines', 'reservoirs','Machines'])

    mixtures_df = dict_df.get('mixtures')
    mixtures_finals_df = dict_df.get('finals')
    production_lines_df = dict_df.get('production_lines')
    reservoirs_df = dict_df.get('reservoirs')
    machines_df = dict_df.get('Machines')

    mixtures,mrc_mixtures_sequence = get_mixtures_dictionary(mixtures_df)

    mix2fin, fin2mix, finals = get_mixtures_finals_dictionaries(mixtures_finals_df)

    fin2pl = get_production_lines_dictionaries(production_lines_df)

    reservoirs = get_reservoirs_dictionary(reservoirs_df)

    machines,machine_descr2m_id = get_machines_dictionary(machines_df)

    machines2prod,resv2prod = gen_machine2prod(mixtures,finals, fin2pl, machines, mrc_mixtures_sequence, machine_descr2m_id, reservoirs)

    return mixtures,mix2fin,fin2mix,finals,fin2pl,reservoirs,machines,mrc_mixtures_sequence, machine_descr2m_id

def import_production_information(fn:str):
    dict_df = pd.read_excel(fn, sheet_name=['products', 'parameters'])
    products_df = dict_df.get('products')
    parameters_df = dict_df.get('parameters')

    products = get_products_dictionary(products_df)
    parameters = get_parameters(parameters_df)

    return products, parameters

def get_parameters(parameters_df):
    parameters = {}
    for i in range(len(parameters_df)):
        param = parameters_df.iloc[i][0]
        value = parameters_df.iloc[i][1]
        parameters[param] = value        
    return parameters

def get_products_dictionary(products_df):
    products = {}
    for i in range(len(products_df)):
        id = products_df.iloc[i]["ID"]
        mix_id = products_df.iloc[i]["MIXTURE_ID"]
        fin_id = products_df.iloc[i]["FINAL_ID"]
        fin_name = products_df.iloc[i]["FINAL_NAME"]
        quantity = products_df.iloc[i]["QUANTITY"]
        products[id] = (fin_id,mix_id,quantity)
            
    return products

def get_mixtures_dictionary(mixtures_df):
    mixtures = {}
    mrc_mixtures_sequence = {}
    for i in range(len(mixtures_df)):
        mix_id = int(mixtures_df.iloc[i]['MIXTURE_ID'])
        description = mixtures_df.iloc[i]['MIXTURE_DESCRIPTION']
        mrc_id = int(mixtures_df.iloc[i]['MRC_ID'])
        mrc_hours = mixtures_df.iloc[i]['MRC_TIME_HR']
        network = mixtures_df.iloc[i]['NETWORK']
        if pd.isna(network):
            network = None
        reservoirs = mixtures_df.iloc[i]['RESERVOIRS'].split(';')
        if mix_id not in mixtures:
            mixtures[mix_id] = (description, mrc_id, mrc_hours, network, reservoirs)
        else: # dublicated???
            print(f'Mixture {mix_id} defined more than once')
        
        #Makes assumption the mixture sequence per mrc is the one in the file 
        if mrc_id not in mrc_mixtures_sequence:
            mrc_mixtures_sequence[mrc_id] = []
        mrc_mixtures_sequence[mrc_id].append(mix_id)
        
    return mixtures,mrc_mixtures_sequence


def get_mixtures_finals_dictionaries(mixtures_finals_df):
    mix2fin = {}
    fin2mix = {}
    finals = {}
    for i in range(len(mixtures_finals_df)):
        mix_id = int(mixtures_finals_df.iloc[i]["MIXTURE_ID"])
        fin_id = int(mixtures_finals_df.iloc[i]["FINAL_ID"])
        fin_name = mixtures_finals_df.iloc[i]["FINAL_NAME"]
        active = mixtures_finals_df.iloc[i]["ACTIVE"]
        #skip products no longer in production
        if active == 'ΕΚΤΟΣ':
            continue
        if mix_id not in mix2fin:
            mix2fin[mix_id] = [fin_id]
        else:
            mix2fin[mix_id].append(fin_id)
        
        if fin_id not in fin2mix:
            fin2mix[fin_id] = [mix_id]
            finals[fin_id] = fin_name
        else:
            fin2mix[fin_id].append(mix_id)
            if finals[fin_id] != fin_name:
                raise Exception(f"final name mismatch {finals[fin_id]} vs {fin_name}")
            
    return mix2fin, fin2mix, finals

def get_production_lines_dictionaries(production_lines_df):
    fin2pl = {}
    for i in range(len(production_lines_df)):
        fin_id = int(production_lines_df.iloc[i]['FINAL_ID'])
        dosage_name = production_lines_df.iloc[i]['Dosage']
        packaging_1_name = production_lines_df.iloc[i]['Packaging_1']
        packaging_2_name = production_lines_df.iloc[i]['Packaging_2']

        if pd.isna(dosage_name): 
            dosage_name = None
        if pd.isna(packaging_1_name): 
            packaging_1_name = None
        if pd.isna(packaging_2_name): 
            packaging_2_name = None

        #Type refers to the type of production
        # 1 = Τελικό προϊόν που συλλέγεται στη μορφοποιητική
        # 2 = Τελικό προϊόν που λαμβάνεται στο τέλος της γραμμής παραγωγής με σύγχρονη μορφοποίηση / συσκευασία
        # 3 = Τελικό προϊόν που λαμβάνεται στο τέλος της γραμμής παραγωγής με ασύγχρονη μορφοποίηση / συσκευασία 
        # 4 = Τελικό προϊόν λαμβάνεται στο τέλος της γραμμής παραγωγής με σύγχρονη μορφοποίηση / συσκευασία
        #     και με ασύγχρονη επιπλέον συσκευασία
        # 5 = Τελικό προϊόν πραλίνας που λαμβάνεται από το ψυγείο, ενώ έχει προηγηθεί σύγχρονη 
        production_type = production_lines_df.iloc[i]['TYPE']

        #First stage production occupies either a dosage machine or a dosage / packaging machine to create semi-final products
        dosage_package1_kg_shift = production_lines_df.iloc[i]['DOSAGE_PACKAGE1_KG_SHIFT']
        dosage_package1_pcs_shift = production_lines_df.iloc[i]['DOSAGE_PACKAGE_1 (PCS/SHIFT)']
        dosage_package1_gr_pcs = production_lines_df.iloc[i]['GR/PIECE']
        if pd.isna(dosage_package1_kg_shift): 
            dosage_package1_kg_shift = None
        if pd.isna(dosage_package1_pcs_shift): 
            dosage_package1_pcs_shift = None
        if pd.isna(dosage_package1_gr_pcs): 
            dosage_package1_gr_pcs = None
        
        #Second stage production is optional and uses a packaging machine to create final from semi-final products
        package2_kg_shift = production_lines_df.iloc[i]['PACKAGE_2_KG_SHIFT']
        package2_pcs_shift = production_lines_df.iloc[i]['PACKAGE_2 (PCS/SHIFT)']
        package2_gr_pcs = production_lines_df.iloc[i]['GR/PIECE_2']
        if pd.isna(package2_kg_shift): 
            package2_kg_shift = None
        if pd.isna(package2_pcs_shift): 
            package2_pcs_shift = None
        if pd.isna(package2_gr_pcs): 
            package2_gr_pcs = None
        
        if fin_id not in production_lines_df:
            fin2pl[fin_id] = (dosage_name,packaging_1_name,packaging_2_name, production_type, (dosage_package1_kg_shift,dosage_package1_pcs_shift,dosage_package1_gr_pcs), (package2_kg_shift,package2_pcs_shift,package2_gr_pcs))
        else:
            raise Exception(f"Final {fin_id} has more than one production lines")
    return fin2pl

def get_reservoirs_dictionary(reservoirs_df):
    reservoirs = {}
    for i in range(len(reservoirs_df)):
        reservoir_id = reservoirs_df.iloc[i]['RESERVOIR_ID']
        reservoir_descr = reservoirs_df.iloc[i]['DESCRIPTION']
        capacity = reservoirs_df.iloc[i]['CAPACITY_TN']
        reservoirs[reservoir_id] = (reservoir_descr, capacity)
    return reservoirs

def get_machines_dictionary(machines_df):
    machines = {}
    machine_descr2m_id = {}
    for i in range(len(machines_df)):
        machine_id = int(machines_df.iloc[i]['Id'])
        stage = machines_df.iloc[i]['Stage']
        descr = machines_df.iloc[i]['Description']
        capacity = float(machines_df.iloc[i]['Capacity_tn'])
        capacity_used_tn = float(machines_df.iloc[i]['Capacity_used_tn'])
        comments = machines_df.iloc[i]['Comments']
        setup = machines_df.iloc[i]['Setup(h)']
        unload = machines_df.iloc[i]['Unload(h)']
        operators = machines_df.iloc[i]['Operators']
        consumption_working = machines_df.iloc[i]['Consumption Working (KWh/h)']
        consumption_waiting = machines_df.iloc[i]['Consumption Waiting (KWh/h)']
        consumption_idle = machines_df.iloc[i]['Consumption Idle (KWh/h)']
        machines[machine_id] = (stage,descr,capacity,capacity_used_tn,comments,setup,unload,operators,
                                consumption_working,consumption_waiting,consumption_idle)
        machine_descr2m_id[descr] = machine_id
    return machines, machine_descr2m_id

def trace_final(fin_id):
    final_name = finals[fin_id]
    if fin_id in fin2pl:
        final_production_line, dosage_kg_shift, package_kg_shift = fin2pl[fin_id]
    else:
        final_production_line = None
    print(f"\nFINAL: {fin_id}, {final_name}")
    for mix_id in fin2mix[fin_id]:
        mixture_description, mrc_id, mrc_hours, network, reservoirs_for_mix = mixtures[mix_id]
        for reservoir_id in reservoirs_for_mix:
            out = [f"MRC={mrc_id}/{mrc_hours}h", f"MIXTURE={mix_id}/{mixture_description}"]
            if reservoir_id in reservoirs:
                out.append(f"RESERVOIR={reservoir_id}/{reservoirs[reservoir_id]}tn")
            else:
                out.append(f"RESERVOIR={reservoir_id}")
            if network:
                out.append(f"TEMPERING NETWORK={network}")
            if final_production_line:
                out.append(f"PRODUCTION={final_production_line}")
                if dosage_kg_shift:
                    out.append(f"DOSAGE={dosage_kg_shift}kg/shift")
                if package_kg_shift:
                    out.append(f"PACKAGE={package_kg_shift}kg/shift")
            print(", ".join(out)) 

def import_product_transition(fn: str) -> None:
    df = pd.read_excel(fn)
    transition_dict = dict()
    #TODO create a transition dictionary
    # We should check the sequence of products in a cycle and impose the constraints.
    # Also we should check the quantity of cleanup and the conditions of the cleanup process
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
        test_temperature = row["Temp [Β°C]"]
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

def conches(mixture_qty, mrc_capacity):
    c = int(mixture_qty / mrc_capacity)
    if c * mrc_capacity < mixture_qty:
        c += 1
    return c

def gen_machine2prod(mixtures,finals, fin2pl, machines, mrc_mixtures_sequence, machine_descr2m_id, reservoirs):
    machines2prod = {}
    resv2prod = {}

    for mrc_id,seq in mrc_mixtures_sequence.items():
        machines2prod[mrc_id] = deepcopy(seq)
    
    for mix_id,mix in mixtures.items():
        if mix[3]:
            network_id = machine_descr2m_id[mix[3]]
            if network_id not in machines2prod:
                machines2prod[network_id] = []
            machines2prod[network_id].append(mix_id)

        resvs = mix[4]
        if resvs:
            for resv in resvs:
                if resv not in resv2prod:
                    resv2prod[resv] = []
                resv2prod[resv].append(mix_id)
        
    for f_id,final in fin2pl.items():
        dosage_name = final[0]
        packaging_1_name = final[1]
        packaging_2_name = final[2]
        production_type = final[3]

        if dosage_name:
            m_id  = machine_descr2m_id[dosage_name]
            if m_id not in machines2prod:
                machines2prod[m_id] = []
            machines2prod[m_id].append(f_id)
        
        if packaging_1_name:
            m_id  = machine_descr2m_id[packaging_1_name]
            if m_id not in machines2prod:
                machines2prod[m_id] = []
            machines2prod[m_id].append(f_id)

        if packaging_2_name:
            m_id  = machine_descr2m_id[packaging_2_name]
            if m_id not in machines2prod:
                machines2prod[m_id] = []
            machines2prod[m_id].append(f_id)

    return machines2prod,resv2prod

def gen_shifts(schedule_start_dt,schedule_end_dt):
    work_periods =  gen_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    no_work_periods = get_no_work_periods((22,6),schedule_start_dt,schedule_end_dt)
    
    return work_periods, no_work_periods



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
                    cost=1000.0
                )
                )
            break
        no_work_periods.append(
            CostInterval(
                    from_datetime=from_dt,
                    to_datetime=to_dt - timedelta(minutes=1),
                    cost=1000.0
            )
        )
        current_dt += timedelta(days=1)
    return no_work_periods

def get_frontend_for_final(self, final_id):
        # Γίνεται η παραδοχή (1) ότι το πρώτο μείγμα στη λίστα των μειγμάτων που δίνουν το τελικό θα χρησιμοποιηθεί για την παραγωγή
        # Η πληροφορία αυτή θα πρέπει κανονικά να υπάρχει στο production (xlsx)
        if final_id not in self.fin2mix:
            raise ValueError(f"Final id {final_id} not found")
        mixture_id = self.fin2mix[final_id][0]
        return (
            mixture_id,
            self.mixtures[mixture_id][1],
            self.mixtures[mixture_id][2],
            self.mixtures[mixture_id][3],
            self.mixtures[mixture_id][4],
        )

    

   


    