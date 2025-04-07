import json
import cbor2
import torch
import torch.nn.functional as F
import os
from collections import Counter

# Required for exporting to parquet format
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

# Required to delete folders
import shutil

def get_entity_id_dict(keys):
    max_entities = len(keys)
    ids = F.one_hot(torch.arange(0, max_entities) % max_entities)
    ids_dict = {}
    for i in range(max_entities):
        ids_dict[keys[i]] = ids[i]
    return ids_dict

tasks_ids_dict = get_entity_id_dict(['pick', 'pickpick1', 'pickpick2', 'handover', 'policy'])
robot_ids_dict = get_entity_id_dict(['a0_', 'a1_', 'a2_', 'a3_'])
objs_ids_dict = get_entity_id_dict([0, 1, 2, 3])
scene_ids_dict = get_entity_id_dict(['conveyor', 'husky', 'shelf', 'random'])
obs_scene_dict = {5: 'conveyor', 44: 'husky', 8: 'shelf', 3: 'random'}

# Define max limits
max_num_objs = len(objs_ids_dict)
max_num_robots = len(robot_ids_dict)
max_num_tasks = len(tasks_ids_dict)
max_num_scenes = len(scene_ids_dict)
max_planSequence_length = 8  #Max planning sequence length should be 2 * num_objects.
encoding_dim = 128  # Same as decision transformers
pose_dim = 7  # 3 coeffs for  positions + 4 coeffs for quaternion              
size_dim = 3  # Dimension of vector describing size

def get_raw_features_dim():
    # token = [robot_id, task_id, obj_id, robot_pos, obj_init_pose, obj_goal_pose, obj_size]
    # return max_num_robots + max_num_tasks + max_num_objs + 3 + 2 * pose_dim + size_dim
    return max_num_tasks + 2 * max_num_robots + max_num_objs + 2 * 3 + 2 * pose_dim + size_dim



# Consider using a proper Environment wrapper (Gymnasium or TorchRL env)
from enum import Enum
class SceneType(Enum):
    random   = 0
    conveyor = 1
    husky    = 2
    shelf    = 3

class CommandMetadata:
    def __init__(self):
        self.data_gen_folder = ""
        self.output_path = ""
        self.objs_filename = ""
        self.robots_filename = ""
        self.extended_seq_filename = ""

    def get_full_output_path(self):
        return self.data_gen_folder + self.output_path


def get_num_files(path):
    if os.path.exists(path):
        return len(os.listdir(path))
    return 0


def get_max_possible_plans(path):
    if not os.path.exists(path):
        return 0
    folders = os.listdir(path)
    folders_num = [int(x) for x in folders]
    return max(folders_num) + 1

def write_dicts_to_file(data_dicts, num_files, max_samples_per_file, output_datafolder):
    if not os.path.exists(output_datafolder):
        os.makedirs(output_datafolder)
    table = pa.Table.from_pylist(data_dicts)
    start_idx = num_files * max_samples_per_file
    final_idx = start_idx + len(data_dicts)
    filename = "samples_" + str(start_idx).zfill(9) + "_to_" + str(final_idx - 1).zfill(9)
    full_path = output_datafolder + filename + ".parquet"
    pq.write_table(table, full_path )

    # Make sure file can be loaded back for a hugging face dataset
    try:
        dataset = load_dataset("parquet", data_files={full_path})
        # Set num_shards >= num_workers.
        # See: https://discuss.huggingface.co/t/num-worker-with-iterabledataset/58914/2
        num_shards = min(8, len(dataset))
        iterable_train_dataset = dataset['train'].to_iterable_dataset(num_shards=num_shards)
    except Exception as e:
        print(f"Error loading file {full_path}")
        print(e)
        print(f"Removing file {full_path}")
        os.remove(full_path)
        return False

    return True


# Utils
def load_text_file(filename):
    if not filename:
        return ""

    with open(filename, 'r') as file:
        data_f = file.read()
        return data_f

def load_compressed_json_file(filename):
    with open(filename, 'rb') as f:
        cbor_data = f.read()
        return cbor2.loads(cbor_data)


def load_json_file(filename, compressed=False):
    if not filename:
        return None

    if compressed:
        return load_compressed_json_file(filename)

    with open(filename) as f:
        return json.load(f)


def write_json_file(filename, data, compressed=False):
    if not compressed:
        with open(filename, 'w') as f:
            json.dump(data, f)
    else:
        with open(filename, 'wb') as fp:
            cbor2.dump(data, fp)


def write_json_file_to_folder(path, filename, data, compressed=False):
    if not os.path.exists(path):
        os.makedirs(path)

    if not compressed:
        with open(path + filename, 'w') as f:
            json.dump(data, f)
    else:
        with open(path + filename, 'wb') as fp:
            cbor2.dump(data, fp)

def load_generated_sequence(folder):
    generated_seqs = None
    try:
        for el in os.scandir(folder):
            if el.is_dir() and el.name.startswith("sequence_generation"):
                generated_seqs = load_json_file(el.path + "/sequences.json", generated_seqs)
    except Exception as e:
        print("Failed to load sequence")
        print(e)
        return None

    return generated_seqs


def get_plans_folder(output_folder):
    try:
        for el in os.scandir(output_folder):
            if el.is_dir() and el.name.startswith("sequence_plan"):
                return el.path
    except Exception as e:
        print("Failed to get_plans_folder")
        print(e)

    return None


def load_plans_from_generated_repeated_seqs(scenarioType, command_metadata, plans_folder, extended_seqs, num_repetitions):

    def fill_in_input_files(scenarioType, command_metadata):
        # Set paths
        robot_path = None
        scene_path = None
        obstacles_path = None
        if scenarioType == SceneType.husky:
            robot_path = command_metadata.data_gen_folder + "in/envs/two_husky.json"
            scene_path = command_metadata.data_gen_folder + "in/scenes/husky.g"
        elif scenarioType == SceneType.conveyor:
            robot_path = command_metadata.data_gen_folder + "in/envs/four_sorting.json"
            scene_path = command_metadata.data_gen_folder + "in/scenes/conveyor.g"
        elif scenarioType == SceneType.shelf:
            robot_path = command_metadata.data_gen_folder + "in/envs/two.json"
            scene_path = command_metadata.data_gen_folder + "in/scenes/floor.g"
            obstacles_path = command_metadata.data_gen_folder + "in/obstacles/shelf_bigger.json"
        elif scenarioType == SceneType.random:
            robot_path = command_metadata.get_full_output_path() + command_metadata.robots_filename

        obj_path = command_metadata.get_full_output_path() + command_metadata.objs_filename
        scenario_with_plans = {}
        scenario_with_plans["obj_file"] = load_json_file(obj_path)
        scenario_with_plans["robot_file"] = load_json_file(robot_path)
        scenario_with_plans["scene_file"] = load_text_file(scene_path)
        scenario_with_plans["obstacles_file"] = load_text_file(obstacles_path)
        scenario_with_plans["scene"] = None

        return scenario_with_plans

    scenario_with_plans = fill_in_input_files(scenarioType, command_metadata)
    scenario_with_plans["sequence_data"] = []

    plan_id = 0
    max_possible_plans = get_max_possible_plans(plans_folder)
    print(f"max_possible_plans: {max_possible_plans}")
    while plan_id < max_possible_plans:
        sequence_data = {}
        sequence_data["sequence"] = extended_seqs["sequences"][plan_id]
        sequence_data["metadatas"] = []
        sequence_data["plans"] = []
        sequence_data["trajectories"] = []

        #extended_seqs["sequences"][plan_id]
        single_plan_folder = plans_folder + "/" + str(plan_id)
        loading_successful = True

        if not os.path.exists(single_plan_folder):
            # Add sequence even if not plans were generated for that specific sequence.
            scenario_with_plans["sequence_data"].append(sequence_data)
            plan_id += num_repetitions
            continue

        #Try to load the next repetitions
        try:
            if scenario_with_plans["scene"] is None:
                scenario_with_plans["scene"] = load_json_file(single_plan_folder + "/scene.json")

            for i in range(num_repetitions):
                current_single_plan_folder = plans_folder + "/" + str(plan_id + i)
                sequence_data["metadatas"].append(load_json_file(current_single_plan_folder + "/metadata.json"))
                sequence_data["plans"].append(load_json_file(current_single_plan_folder + "/plan.json"))
                sequence_data["trajectories"].append(load_json_file(current_single_plan_folder + "/trajectory.json"))

        except Exception as e:
            loading_successful = False
            print("Failed to load plans")
            print(e)

        if loading_successful:
            scenario_with_plans["sequence_data"].append(sequence_data)

        plan_id += num_repetitions

    return scenario_with_plans

def extend_generated_sequence(generated_seqs, num_repetitions = 5):
    #print("len", len(generated_seqs["sequences"]))
    #print("generated_seqs", generated_seqs["sequences"][0]["tasks"])

    # Create hashtable where each sequence from generated_seqs is used as a hash-index,
    # The values of the table are the last indexes in the original list (generated_seqs)
    # that correspond to a sequence. This way, we remove any duplicated sequences.

    hash_table = {}
    for count, generated_seq in enumerate(generated_seqs["sequences"]):
        # Convert sequence to hashable type
        # TODO: Figure out more scalable way to convert dict to tuple
        try:
            s = tuple(
                {tuple([(key, val) if type(val) is not list else (key, tuple(val)) for key, val in dict_el.items()]) for
                 dict_el in generated_seq["tasks"]})
            hash_table[s] = count
        except Exception as e:
            print("Failed to extend sequence (conversion of dict to tuple)")
            print(generated_seq["tasks"])
            print(e)
            return None


    seqs_extended = {"sequences": []}
    for key, original_index in hash_table.items():
        for _ in range(num_repetitions):
            seqs_extended["sequences"].append(generated_seqs["sequences"][original_index])
        # print("val", val)

    return seqs_extended

