import os
from makespan_utils import *
import random
from datetime import datetime

def get_cmd_str_to_generate_sequences(relative_path_to_robot_file,
                                      relative_path_to_obj_file,
                                      output_path,
                                      r_seed):
    cmd_str = "echo $PWD && cd /home/tapas/multi-agent-tamp-solver/24-data-gen/ && echo $PWD && "
    cmd_str += "xvfb-run -a --server-args=\"-screen 0 480x480x24\" "
    cmd_str += "./x.exe -pnp true -mode generate_candidate_sequences -seed " + str(r_seed) + " "
    cmd_str += "-attempt_all_grasp_directions true "
    cmd_str += "-robot_path " + relative_path_to_robot_file + " "
    cmd_str += "-obj_path " + relative_path_to_obj_file + " "
    cmd_str += "--attempt_komo false -display false -export_images false -verbosity 5 -early_stopping false "
    cmd_str += "-scene_path 'in/scenes/floor.g' "
    cmd_str += "-obstacle_path 'in/obstacles/shelf_bigger.json' "
    cmd_str += "-output_path " + output_path + " "
    return cmd_str

def get_cmd_str_to_plan_for_sequence(relative_path_to_robot_file,
                                     relative_path_to_obj_file,
                                     relative_path_to_seq_file,
                                     output_path,
                                     r_seed):
    cmd_str = "echo $PWD && cd /home/tapas/multi-agent-tamp-solver/24-data-gen/ && echo $PWD && "
    cmd_str += "xvfb-run -a --server-args=\"-screen 0 480x480x24\" "
    cmd_str += "./x.exe -pnp true -mode plan_for_sequence -seed " + str(r_seed) + " "
    cmd_str += "-attempt_all_grasp_directions true "
    cmd_str += "-robot_path " + relative_path_to_robot_file + " "
    cmd_str += "-obj_path " + relative_path_to_obj_file + " "
    cmd_str += "--attempt_komo false -display false -export_images false -verbosity 5 -early_stopping false "
    cmd_str += "-scene_path 'in/scenes/floor.g' "
    cmd_str += "-obstacle_path 'in/obstacles/shelf_bigger.json' "
    cmd_str += "-sequence_path " + relative_path_to_seq_file + " "
    cmd_str += "-output_path " + output_path + " "
    return cmd_str

def latest_sequences(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()] #full paths for all subdirectories within folder_path
    sequences = [s for s in subfolders if s.startswith(f"{folder_path}sequence_generation")]  # the list of all generated sequences
    if not sequences:
        return None
    
    latest_sequences = max(sequences, key=os.path.getmtime)

    return latest_sequences

ENV_NAME = "shelf"
sample_indx = 52
ROB_NAME = f"{ENV_NAME}_robot_output_{sample_indx}_rela"
OBJ_NAME = f"{ENV_NAME}_obj_output_{sample_indx}_rela"

solver_path = "/home/tapas/multi-agent-tamp-solver/24-data-gen/"

# in the folder of this python file, we should use relative path to the input files(replan_data)
rela_path_to_input_files = "../multi-agent-tamp-solver/24-data-gen/replan_data/replan_ini_shelf_52_20250317_143018/"
with open(f"{rela_path_to_input_files}part_hold_info_{ENV_NAME}_{sample_indx}.json", "r") as file:
    part_hold_info = json.load(file)

# when we run cmd_str, we will first cd to the folder of replan_data
path_to_input_files = "replan_data/replan_ini_shelf_52_20250317_143018/"
for i in range(len(part_hold_info)):
    subdir = part_hold_info[i]["hold_step"]
    path_to_robot_file =  path_to_input_files + f"{subdir}"+"/" + ROB_NAME + ".json" 
    path_to_obj_file =  path_to_input_files + f"{subdir}"+"/" + OBJ_NAME + ".json"
    output_path = path_to_input_files + f"{subdir}" + "/"
    print("path ", path_to_input_files)
    # if not os.path.exists(full_path_to_input_files):
    #     os.makedirs(full_path_to_input_files)
    ppp = os.path.join(solver_path, output_path)
    subf = [f.path for f in os.scandir(ppp) if f.is_dir()]
    plan = [s for s in subf if s.startswith(f"{ppp}sequence_plan")]
    if plan:
        print("Already planned for this sequence")
        continue
    
    r_seed = random.randint(0, 9999)
    # Generate sequences
    cmd_str = get_cmd_str_to_generate_sequences(path_to_robot_file,
                                                path_to_obj_file,
                                                output_path,
                                                r_seed)
    exec_cmd(cmd_str)

    # plan for sequences
    seq_name = "sequence_generation"
    latest_sequence = latest_sequences(os.path.join(solver_path, output_path)) # return the absolute path value
    seq_fullname = latest_sequence.removeprefix(f"{solver_path}")
    print(seq_fullname)
    relative_path_to_seq_file = seq_fullname + "/" + "sequences.json"
    cmd_str = get_cmd_str_to_plan_for_sequence(path_to_robot_file, 
                                                        path_to_obj_file, 
                                                        relative_path_to_seq_file, 
                                                        output_path,
                                                        r_seed)
    exec_cmd(cmd_str)
