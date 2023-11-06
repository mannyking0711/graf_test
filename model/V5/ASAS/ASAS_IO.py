from copy import deepcopy
import os
import logging
import pandas as pd
import openpyxl
from datetime import datetime, timedelta


def import_electricity_consumption_and_production_information(fn:str):
    dict_df = pd.read_excel(fn, sheet_name=['Anodizing Elec. Consp.', 'Anodizing Production'])

    elec_cons_df = dict_df.get('Anodizing Elec. Consp.')
    production_df = dict_df.get('Anodizing Production')

    daily_consumptions = get_daily_consumption_dictionary(elec_cons_df)

    daily_production = get_daily_production_dictionary(production_df)

    return daily_consumptions,daily_production

def import_problem(fn:str):
    dict_df = pd.read_excel(fn, sheet_name=['Electricity unit price', 'Anodizing Production Plan', 'Parameters', 'Generators'])

    market_cost_df = dict_df.get('Electricity unit price')
    production_df = dict_df.get('Anodizing Production Plan')
    parameters_df = dict_df.get('Parameters')
    generators_df = dict_df.get('Generators')

    market_cost = get_market_dictionary(market_cost_df)
    daily_production = get_daily_production_dictionary(production_df)
    parameters = get_parameters(parameters_df)
    generators = get_generators(generators_df)

    return market_cost,daily_production,parameters,generators

def get_parameters(parameters_df):
    parameters = {}
    for i in range(len(parameters_df)):
        param_id = parameters_df.iloc[i][0]
        param_name = parameters_df.iloc[i][1]
        param_description = parameters_df.iloc[i][2]
        value = parameters_df.iloc[i][3]
        parameters[param_id] = (param_id, param_name, param_description, value)        
    return parameters

def get_daily_consumption_dictionary(elec_cons_df):
    daily_consumptions = {}
    for i in range(len(elec_cons_df)):
        date = elec_cons_df.iloc[i][0]
        value = elec_cons_df.iloc[i][1]
        daily_consumptions[date] = value        
    return daily_consumptions

def get_daily_production_dictionary(production_df):
    production_dict = {}
    for i in range(len(production_df)):
        date = production_df.iloc[i]["DATE"]
        if date not in production_dict:
            production_dict[date] = {}
        start_time = production_df.iloc[i]["START_TIME"]
        finish_time = production_df.iloc[i]["FINISH_TIME"]
        production=[]
        for j in range(3, 9):
            production.append(production_df.iloc[i,j])
        production_dict[date][start_time, finish_time] = production            
    return production_dict
    
def get_market_dictionary(market_cost_df):
    market_cost = {}
    for i in range(len(market_cost_df)):
        date = market_cost_df.iloc[i]["Date"]
        if date not in market_cost:
            market_cost[date] = {}
        start_time = market_cost_df.iloc[i]["Time"]
        finish_time = (datetime.combine(date, start_time) + timedelta(hours=1)).time()
        value = market_cost_df.iloc[i]["Value(TL/MWh)"]
        market_cost[date][(start_time,finish_time)] = value        
    return market_cost
   
def get_generators(generators_df):
    generators = {}
    for i in range(len(generators_df)):
        id = generators_df.iloc[i][0]
        mode = generators_df.iloc[i][1]
        elec_prod = generators_df.iloc[i][2]
        ng_consumption = generators_df.iloc[i][3]
        steam_kg = generators_df.iloc[i][4]
        steam_kcal = generators_df.iloc[i][5]
        hot_water_kg = generators_df.iloc[i][6]
        hot_water_kcal = generators_df.iloc[i][7]
        if id not in generators:
            generators[id]={}
        generators[id][mode] = (elec_prod,steam_kg,steam_kcal,hot_water_kg,hot_water_kcal,ng_consumption)
    return generators

    