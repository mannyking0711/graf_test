import os
from model.V5.synthetic_problem_generator import ProblemGeneratorV5
from model.V5.json_io import export_solution_to_json
from model.V5.Prima.Prima_problem_generator import Prima_PG


# ----------------------
if __name__ == "__main__":    
    path = os.path.dirname(__file__)
    if path == "":
        path = "."

    xls_file_name = "Prima_problem_1.xlsx"
    problem_name = xls_file_name.split(".")[0]

    fname = f"data/V5/Prima_problem_{problem_name}.json"    

    spg = Prima_PG(path+"/",xls_input_file=xls_file_name)
    spg.save(
        filename=fname
    )
