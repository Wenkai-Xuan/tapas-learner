import os
import random
import subprocess
import time
import matplotlib.pyplot as plt

from MultiArmTampSolverUtils import *

class PseudoDirEntry:
   def __init__(self, path):
      self.path = os.path.realpath(path)
      self.name = os.path.basename(self.path)

# Find last created folder
def last_created_folder(path):
    all_subdirs = [d for d in os.scandir(path) if d.is_dir()]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    print(latest_subdir)
    return latest_subdir


def write_seq_file(full_filename, ordered_idx, samples, global_indexes):
    ordered_sequences = {"sequences": []}
    for i in ordered_idx:
        ordered_sequences["sequences"].append(samples[global_indexes[i].item()]["sequence"])

    # print(ordered_sequences)
    write_json_file(full_filename, ordered_sequences)


def get_cmd_str_to_generate_sequences(env_filename, robot_filename, obj_filename, r_seed):
    cmd_str = "echo $PWD && cd /home/tapas/multi-agent-tamp-solver/24-data-gen/ && echo $PWD && "
    cmd_str += "xvfb-run -a --server-args=\"-screen 0 480x480x24\" "
    cmd_str += "./x.exe -pnp true -mode generate_candidate_sequences -seed " + str(r_seed) + " "
    cmd_str += "-robot_path in/../../../ltampData/" + env_filename + "/" + robot_filename + " "
    cmd_str += "-obj_path in/../../../ltampData/" + env_filename + "/" + obj_filename + " "
    cmd_str += "--attempt_komo false -display false -export_images false -verbosity 5 -early_stopping false "
    cmd_str += "-scene_path 'in/scenes/floor.g' "
    # cmd_str += "-obstacle_path 'in/obstacles/shelf.json' "
    return cmd_str


def get_cmd_str_to_plan_for_sequence(seq_id, env_filename, robot_filename, obj_filename, r_seed):
    cmd_str = "echo $PWD && cd /home/tapas/multi-agent-tamp-solver/24-data-gen/ && echo $PWD && "
    cmd_str += "xvfb-run -a --server-args=\"-screen 0 480x480x24\" "
    cmd_str += "./x.exe -pnp true -mode plan_for_sequence -seed " + str(r_seed) + " "
    cmd_str += "-robot_path in/../../../ltampData/" + env_filename + "/" + robot_filename + " "
    cmd_str += "-obj_path in/../../../ltampData/" + env_filename + "/" + obj_filename + " "
    cmd_str += "--attempt_komo false -display false -export_images false -verbosity 5 -early_stopping false "
    cmd_str += "-scene_path 'in/scenes/floor.g' "
    # cmd_str += "-obstacle_path 'in/obstacles/shelf.json' "
    cmd_str += "-sequence_path in/../../../ltampData/" + env_filename + "/" + seq_id + ".json"
    return cmd_str


def get_cmd_str_to_generate_sequences(relative_path_to_robot_file,
                                      relative_path_to_obj_file,
                                      r_seed):
    cmd_str = "echo $PWD && cd /home/tapas/multi-agent-tamp-solver/24-data-gen/ && echo $PWD && "
    cmd_str += "xvfb-run -a --server-args=\"-screen 0 480x480x24\" "
    cmd_str += "./x.exe -pnp true -mode generate_candidate_sequences -seed " + str(r_seed) + " "
    cmd_str += "-robot_path " + relative_path_to_robot_file + " "
    cmd_str += "-obj_path " + relative_path_to_obj_file + " "
    cmd_str += "--attempt_komo false -display false -export_images false -verbosity 5 -early_stopping false "
    cmd_str += "-scene_path 'in/scenes/floor.g' "
    # cmd_str += "-obstacle_path 'in/obstacles/shelf.json' "
    return cmd_str


def get_cmd_str_to_plan_for_sequence(relative_path_to_robot_file,
                                     relative_path_to_obj_file,
                                     relative_path_to_seq_file,
                                     r_seed):
    cmd_str = "echo $PWD && cd /home/tapas/multi-agent-tamp-solver/24-data-gen/ && echo $PWD && "
    cmd_str += "xvfb-run -a --server-args=\"-screen 0 480x480x24\" "
    cmd_str += "./x.exe -pnp true -mode plan_for_sequence -seed " + str(r_seed) + " "
    cmd_str += "-robot_path " + relative_path_to_robot_file + " "
    cmd_str += "-obj_path " + relative_path_to_obj_file + " "
    cmd_str += "--attempt_komo false -display false -export_images false -verbosity 5 -early_stopping false "
    cmd_str += "-scene_path 'in/scenes/floor.g' "
    # cmd_str += "-obstacle_path 'in/obstacles/shelf.json' "
    cmd_str += "-sequence_path " + relative_path_to_seq_file + " "
    return cmd_str

def exec_cmd(cmd_str):
    print("cmd_str: ", cmd_str)
    try:
        normal = subprocess.run(cmd_str,
                                shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=True,
                                text=True)
    except Exception as e:
        print("Subprocess exception")
        print(e)

    # with open("stderr.txt", "w") as text_file:
    #     text_file.write(normal.stderr.__str__())
    #
    # with open("stdout.txt", "w") as text_file:
    #     text_file.write(normal.stdout.__str__())


def build_raw_sample(sequences, scene):
    samples = []
    for seq in sequences["sequences"]:        
        samples.append({"scene": scene, 
                        "sequence": seq})
        # For compatibility with hf_samples we need to ensure that 
        # sample['metadata']['metadata']['makespan'] is a valid field even it is unused        
        samples[-1]["metadata"] = {"metadata": {"makespan": -1}}
        

    print("num_samples", len(samples))
    return samples


def get_makespans_and_compute_times(output_path):
    # Find last created folder
    candidate_sols_subdir = last_created_folder(output_path)

    makespans = {}
    compute_times = {}
    # for candidate_sol in os.scandir(latest_subdir):
    for i in range(100):
        candidate_sol_path = candidate_sols_subdir.path + "/" + str(i)
        if not os.path.exists(candidate_sol_path):
            continue

        metadata = load_json_file(candidate_sol_path + "/metadata.json")

        makespans[i] = metadata['metadata']['makespan']
        compute_times[i] = metadata['metadata']['cumulative_compute_time']

    print(makespans)
    print(compute_times)
    return makespans, compute_times


def plot_makespans(makespans, compute_times, plot_name):
    l = list(makespans.values())
    min_makespan_found = [min(l[0:i]) for i in range(1, len(l) + 1)]
    x = list(compute_times.values())
    plt.clf()
    plt.plot(x, l)
    plt.plot(x, min_makespan_found)
    # plt.plot( l)
    plt.xlabel('Time ')
    plt.ylabel('Best makespan')
    # plt.show()
    plt.savefig(f'./{plot_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')