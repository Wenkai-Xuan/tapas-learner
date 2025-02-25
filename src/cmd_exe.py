import os
from makespan_utils import *
import random
from datetime import datetime

def latest_sequences(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    sequences = [s for s in subfolders if s.startswith("/home/tapas/multi-agent-tamp-solver/24-data-gen/out/sequence_generation")]  # the list of all generated sequences
    if not sequences:
        return None
    
    latest_sequences = max(sequences, key=os.path.getmtime)

    return latest_sequences

i = 3
ROB_NAME = ["random_robot_output_520_nnew", "husky_robot_output_520_new", "conveyor_robot_output_5_new", "shelf_robot_output_52_new"]
OBJ_NAME = ["random_obj_output_520_nnew", "husky_obj_output_520_new", "conveyor_obj_output_5_new", "shelf_obj_output_52_new"]

solver_path = "/home/tapas/multi-agent-tamp-solver/24-data-gen/"

relative_path_to_input_files = "in/"
relative_path_to_robot_file =  relative_path_to_input_files + "envs/" + ROB_NAME[i] + ".json" 
relative_path_to_obj_file =  relative_path_to_input_files + "objects/" + OBJ_NAME[i] + ".json"
full_path_to_input_files = solver_path + relative_path_to_input_files
print("path ", full_path_to_input_files)
# if not os.path.exists(full_path_to_input_files):
#     os.makedirs(full_path_to_input_files)

r_seed = random.randint(0, 9999)
# Generate sequences
cmd_str = get_cmd_str_to_generate_sequences(relative_path_to_robot_file,
                                            relative_path_to_obj_file,
                                            r_seed)
exec_cmd(cmd_str)

# plan for sequences
seq_name = "sequence_generation"
latest_sequence = latest_sequences(solver_path + "out") # return the absolute path value
seq_fullname = latest_sequence.removeprefix("/home/tapas/multi-agent-tamp-solver/24-data-gen/")
print(seq_fullname)
relative_path_to_seq_file = seq_fullname + "/" + "sequences.json"
cmd_str = get_cmd_str_to_plan_for_sequence(relative_path_to_robot_file, 
                                                    relative_path_to_obj_file, 
                                                    relative_path_to_seq_file, 
                                                    r_seed)
exec_cmd(cmd_str)
