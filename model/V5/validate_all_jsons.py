import os
# import json
import logging
# from solvers.json_validator import validate_scenario_json, validate_solution_json

from pydantic import parse_file_as
from .scenario import Scenario,Solution

def validate_scenario_jsons_in_folder(folder):
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            fn = os.path.join(root, name)
            try:
                scenario = parse_file_as(path=fn, type_=Scenario)
                logging.info(f"Problem {name} validated")
                # print(scenario.json(indent=3))
            except Exception as e:
                logging.info(f"Problem {name} failed. Error = {e}")


def validate_solution_jsons_in_folder(folder):
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            fn = os.path.join(root, name)
            try:
                solution = parse_file_as(path=fn, type_=Solution)
                logging.info(f"Solution {name} validated")
            except Exception as e:
                logging.info(f"Solution {name} failed. Error = {e}")

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    validate_scenario_jsons_in_folder("./data/V4")
    validate_solution_jsons_in_folder("./data/V4_sols")
