from datetime import datetime
import logging

from .scenario import Scenario,Solution

def import_scenario_from_json(input_file_name):
    f = open(input_file_name,encoding="utf-8")
    json_text = f.read()
    return import_scenario_from_json_text(json_text)

def import_scenario_from_json_text(json_text):
    scenario = Scenario.model_validate_json(json_text)
    scenario.fix_after_load_from_json()
    return scenario

def import_solution_from_json_text(json_text):
    a_solution = Solution.model_validate_json(json_text)
    return a_solution

def import_solution_from_json(input_file_name):
    f = open(input_file_name, encoding="utf-8")
    json_text = f.read()
    return import_solution_from_json_text(json_text)

def export_scenario_to_json_text(scenario:Scenario):  
    return scenario.model_dump_json(exclude_unset=True, indent=4)

def export_scenario_to_json(scenario:Scenario, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        txt = export_scenario_to_json_text(scenario)
        f.write(txt)

def export_solution_to_json_as_text(solution: Solution, solution_id: int):
    solution.id = solution_id
    solution.generated_date = datetime.now()
    return solution.model_dump_json(exclude_unset=True, indent=4)

def export_solution_to_json(solution: Solution, solution_id:int, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        txt = export_solution_to_json_as_text(solution,solution_id)
        f.write(txt)