import os
import torch
from multiarm_tamp_solver_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("pid", os.getpid(), device)

max_planSequence_length = 8  # Max planning sequence length should be 2 * num_objects.
encoding_dim = 128  # Same as decision transformers
pose_dim = 7  # 3 coeffs for  positions + 4 coeffs for quaternion
size_dim = 3  # Dimension of vector describing size


def get_raw_features_dim():
    # token = [robot_id, task_id, obj_id, robot_pos, obj_init_pose, obj_goal_pose, obj_size], robot_joint
    # Consider: - Removing robot_id
    #           - Adding robot type.
    #           - Adding robot EE pose
    #           - Adding Robot Joint pos.

    # return max_num_robots + max_num_tasks + max_num_objs + 3 + 2 * self.pose_dim + self.size_dim
    return max_num_tasks + 2 * max_num_robots + max_num_objs + 2 * 3 + 2 * pose_dim + size_dim + 6 * 2 + max_num_scenes
    # pose_size * involv_num_robots (or 6 * max_num_robots) + scene_num


def get_raw_features_seq(entry):
    # Zero initialization deals with padding
    token_dim = get_raw_features_dim()
    result = torch.zeros((max_planSequence_length, token_dim))

    token_cntr = 0
    # print("entry ", entry)
    for task in entry['sequence']['tasks']:
        if token_cntr == max_planSequence_length:
            # returning here deals with truncations
            return result
            # Todo: Figure out step id.
        task_id = tasks_ids_dict[task['primitive']]
        obj_id = objs_ids_dict[task['object']]
        # print("obs_key_len:", entry['scene']['Obstacles'].keys())
        count = sum(1 for v in entry['scene']['Obstacles'].values() if v is not None)
        scene_id = scene_ids_dict[obs_scene_dict[count]] # use the length of the obstacles' keys to get the scene id

        robot_name = task['robots'][0]
        robot_id_first = robot_ids_dict[robot_name]
        robot_pos_first = torch.Tensor(entry['scene']['Robots'][robot_name]['base_pose']['pos'])
        robot_joint_first = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])

        robot_id_second = robot_ids_dict[task['robots'][0]]
        robot_pos_second = torch.Tensor(entry['scene']['Robots'][robot_name]['base_pose']['pos'])
        robot_joint_second = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        if task['primitive'] != 'pick':
            robot_name = task['robots'][1]
            robot_id_second = robot_ids_dict[robot_name]
            robot_pos_second = torch.Tensor(entry['scene']['Robots'][robot_name]['base_pose']['pos'])
            robot_joint_second = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        
        # try: 
        #     robot_name = task['robots'][2]
        #     robot_joint_third = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        # except: 
        #     robot_joint_third = torch.zeros(6)
        # try: 
        #     robot_name = task['robots'][3]
        #     robot_joint_fourth = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        # except: 
        #     robot_joint_fourth = torch.zeros(6)
        
        obj_key = 'obj' + str(task['object'] + 1)
        goal_key = 'goal' + str(task['object'] + 1)
        obj_init_pos = torch.Tensor(entry['scene']['Objects'][obj_key]['start']['abs_pos'])
        obj_init_rot = torch.Tensor(entry['scene']['Objects'][obj_key]['start']['abs_quat'])

        obj_goal_pos = torch.Tensor(entry['scene']['Objects'][obj_key]['goal']['abs_pos'])
        obj_goal_rot = torch.Tensor(entry['scene']['Objects'][obj_key]['goal']['abs_quat'])

        obj_size = torch.Tensor(entry['scene']['Objects'][obj_key]['size'][0:3])
        temp = torch.cat((task_id, robot_id_first,
                          robot_id_second, obj_id, scene_id,
                          robot_pos_first, robot_pos_second,
                          robot_joint_first, robot_joint_second, #robot_joint_third, robot_joint_fourth,
                          obj_init_pos, obj_init_rot,
                          obj_goal_pos, obj_goal_rot, obj_size), 0)

        result[token_cntr, :] = temp

        token_cntr = token_cntr + 1

    return result


def hf_get_flat_label(entry):
    return torch.tensor(entry['metadata']['metadata']['makespan'])

def hf_get_flat_label_devel(entry):
    makespans = torch.tensor([x["metadata"]["makespan"] for x in entry["metadatas"]])
    return torch.mean(makespans)

def hf_get_src_key_padding_mask(entry):
    sequence_length = len(entry["sequence"]['tasks'])
    result = torch.ones(max_planSequence_length, dtype=torch.bool)
    result[0:sequence_length] = torch.zeros(sequence_length, dtype=torch.bool)
    return result


# Create columns of tokens, src_key_padding_mask, targets, and indexes with .map
def process_data(entry):
    tokens = get_raw_features_seq(entry)
    mask = hf_get_src_key_padding_mask(entry)
    target = hf_get_flat_label(entry)
    folder = entry['metadata']['metadata']['folder']
    return {'observation': tokens, 'src_key_padding_mask': mask, 'targets': target, 'folders': folder}

def process_data_devel(entry):
    tokens = get_raw_features_seq(entry)
    mask = hf_get_src_key_padding_mask(entry)
    target = hf_get_flat_label_devel(entry)
    #folder = entry['metadata']['metadata']['folder']
    #return {'observation': tokens, 'src_key_padding_mask': mask, 'targets': target, 'folders': folder}
    return {'observation': tokens, 'src_key_padding_mask': mask, 'targets': target}

def get_raw_features_seq_per_time_step(entry):
    # Zero initialization deals with padding
    token_dim = get_raw_features_dim()

    result = torch.zeros((max_planSequence_length, token_dim))
    token_cntr = 0
    # print("entry ", entry)
    for task in entry['sequence']['tasks']:
        if token_cntr == max_planSequence_length:
            # returning here deals with truncations
            return result
            # Todo: Figure out step id.
        task_id = tasks_ids_dict[task['primitive']]
        obj_id = objs_ids_dict[task['object']]
        scene_id = scene_ids_dict[obs_scene_dict[len(entry['scene']['Obstacles'].keys())]] # use the length of the obstacles' keys to get the scene id

        robot_name = task['robots'][0]
        robot_id_first = robot_ids_dict[robot_name]
        robot_pos_first = torch.Tensor(entry['scene']['Robots'][robot_name]['base_pose']['pos'])
        robot_joint_first = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])

        robot_id_second = robot_ids_dict[task['robots'][0]]
        robot_pos_second = torch.Tensor(entry['scene']['Robots'][robot_name]['base_pose']['pos'])
        robot_joint_second = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        if task['primitive'] != 'pick':
            robot_name = task['robots'][1]
            robot_id_second = robot_ids_dict[robot_name]
            robot_pos_second = torch.Tensor(entry['scene']['Robots'][robot_name]['base_pose']['pos'])
            robot_joint_second = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])

        # try: 
        #     robot_name = task['robots'][2]
        #     robot_joint_third = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        # except: 
        #     robot_joint_third = torch.zeros(6)
        # try: 
        #     robot_name = task['robots'][3]
        #     robot_joint_fourth = torch.Tensor(entry['scene']['Robots'][robot_name]['initial_pose'])
        # except: 
        #     robot_joint_fourth = torch.zeros(6)
        
        obj_key = 'obj' + str(task['object'] + 1)
        goal_key = 'goal' + str(task['object'] + 1)
        obj_init_pos = torch.Tensor(entry['scene']['Objects'][obj_key]['start']['abs_pos'])
        obj_init_rot = torch.Tensor(entry['scene']['Objects'][obj_key]['start']['abs_quat'])

        obj_goal_pos = torch.Tensor(entry['scene']['Objects'][obj_key]['goal']['abs_pos'])
        obj_goal_rot = torch.Tensor(entry['scene']['Objects'][obj_key]['goal']['abs_quat'])

        obj_size = torch.Tensor(entry['scene']['Objects'][obj_key]['size'][0:3])
        temp = torch.cat((task_id, robot_id_first,
                          robot_id_second, obj_id, scene_id,
                          robot_pos_first, robot_pos_second,
                          robot_joint_first, robot_joint_second, #robot_joint_third, robot_joint_fourth,
                          obj_init_pos, obj_init_rot,
                          obj_goal_pos, obj_goal_rot, obj_size), 0)

        result[token_cntr, :] = temp

        token_cntr = token_cntr + 1

    return result

def process_data_per_time_step(entry):
    tokens = get_raw_features_seq_per_time_step(entry)
    mask = hf_get_src_key_padding_mask(entry) #Reduce time horizons.
    target = hf_get_flat_label(entry) # Reduce time horizons.
    folder = entry['metadata']['metadata']['folder']
    return {'observation': tokens, 'src_key_padding_mask': mask, 'targets': target, 'folders': folder}


print("\npid", os.getpid(), "declarations done")