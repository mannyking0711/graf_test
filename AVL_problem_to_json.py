import os
from model.V5.synthetic_problem_generator import ProblemGeneratorV5
from model.V5.json_io import export_solution_to_json
from model.V5.AVL.AVL_problem_loader import AVL_PG

# ----------------------
if __name__ == "__main__":    
    path = os.path.dirname(__file__)
    if path == "":
        path = "."

    # fname = f"data/V4/synthetic_problem_f1_j5_t16_m2_a1.json"
    # sol_fname = f"data/V4_sols/synthetic_problem_f1_j5_t16_m2_a1_sol.json"
    # xls_file_name = "problem01_reservations_anon_tiny.xlsx"
    # xls_file_name = "problem01_reservations_anon_small.xlsx"
    # xls_file_name = "problem_y23_w13_reservations_anon.xlsx"
    # xls_file_name = "problem_y23_w13_reservations_anon_tiny.xlsx"
    # xls_file_name = "problem_y23_w13_reservations_anon_2day.xlsx"
    # xls_file_name = "problem_y23_w17_reservations_anon_no_st.xlsx"
    xls_file_name = "problem_y23_w17_reservations_anon_no_st_small.xlsx"
    # xls_file_name = "problem_y23_w05_reservations_anon.xlsx"
    problem_name = xls_file_name.split(".")[0]

    
    transitions_json_file_name = "transition_table_06102023.json"
    tests_json_file_name = "tests_table_06102023.json"
    reservations_file_name = "reservation_05102023.json"
    energy_prices_file_name = "energy_price_prediction_09102023.json"
    problem_name = reservations_file_name.split(".")[0]   


    fname = f"data/V5/AVL_{problem_name}.json"
    sol_fname = f"data/V5_sols/AVL_{problem_name}_sol.json"

    spg = AVL_PG()


    #LOAD xls reservation data and use xls based static names in transition and tests files
    # spg.load_problem_xlsx(path+"/../",xls_input_file=xls_file_name)

    #LOAD json reservation data and use based static names in transition and tests files
    spg.load_problem_reservations_json(path+"/",reservations_file_name)

    #LOAD json files
    # spg.load_problem_json(path+"/../",transitions_json_file_name,tests_json_file_name,reservations_file_name,energy_prices_file_name)
    

    #LOAD json strings from json files
    # transitions_json_str = open(path+"/../"+"data/AVL/"+transitions_json_file_name,"r",encoding='utf-8').read()
    # tests_json_str = open(path+"/../"+"data/AVL/"+tests_json_file_name,"r",encoding='utf-8').read()
    # reservations_str = open(path+"/../"+"data/AVL/"+reservations_file_name,"r",encoding='utf-8').read()
    # energy_prices_str = open(path+"/../"+"data/AVL/"+energy_prices_file_name,"r",encoding='utf-8').read()

    # spg.load_problem_json_str(transitions_json_str,tests_json_str,reservations_str,energy_prices_str)

    spg.save(
        filename=fname
    )
