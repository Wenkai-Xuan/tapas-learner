#%%
# Imports
import os

import torch.nn as nn
from tqdm import tqdm

import wandb
import random
from datetime import datetime

from makespan_utils import *
from tapasformer import *
from tapas_dataset import *
from tapas_utils import *

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra


#Load envs instances
#Generate sequences
#Load sequences
#Run inference
@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_testing(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))    

    #%%
    # Define params
    ENV_NAME = cfg["env_name"] # "random" "husky" "conveyor" "shelf"
    target_num_robots = cfg["testing"]["target_num_robots"]
    target_num_objs = cfg["testing"]["target_num_objs"]

    #%%
    # Downloading dataset and converting to iterable dataset is faster than using load_dataset() with streaming=True.
    # The iterable dataset will stream from local files. This approach is only feasible provided enough disk space.
    # See: https://huggingface.co/docs/datasets/en/stream#convert-from-a-dataset
    # Note: Avoid download_mode='force_redownload' when using the num_workers argument for the DataLoader().
    dataset = load_dataset("data_replan",
                        data_files={'train': "conveyor_5_rela_train.parquet",
                                    'test': "conveyor_5_rela_test.parquet"})  # download_mode='force_redownload'
    #print("pid", os.getpid(), dataset)
    
    
    #%%
    ### Load pretrained model.
    root_path = "/home/tapas/src/logs/2025_01_14_13_54_43/"
    solver_path = "/home/tapas/multi-agent-tamp-solver/24-data-gen/"

    model_path = root_path + "/model_latest.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(model_path, map_location= map_location)

    normalization_params_path = root_path + "/normalization_params.json" 
    normalization_params = load_json_file(normalization_params_path)

    #test_samples_ids = load_json_file(root_path + "test_samples_ids.json")

    obs_dim = get_raw_features_dim()
    model = TapasFormer(obs_dim=obs_dim, normalization_params=normalization_params)
    model.load_state_dict(state_dict)
    #%%
    len(dataset['test']) 
    #%%
    ###Load envs instances
    ids = []

    for i in tqdm(range(len(dataset['test']))):
        if (dataset['test'][i]["metadata"]["metadata"]["num_robots"] == target_num_robots and  
            dataset['test'][i]["metadata"]["metadata"]["num_objects"] == target_num_objs):
            ids.append(i)
        # Comment the following conditional the search for all samples that have target_num_robots and target_num_objs
        if len(ids) > 1:
            break    

    if len(ids) == 0:
        print("No valid samples found")
        exit()
    else:
        print("# of valid samples: ", len(ids))
        
    env_id = ids[random.randint(0, len(ids) - 1)]
    env_sample = dataset['test'][env_id]

    # Extract suffixes for consistency.
    robot_suffix = "num_robots_" + str(env_sample["metadata"]["metadata"]["num_robots"]) 
    obj_suffix = "num_obj_" + str(env_sample["metadata"]["metadata"]["num_objects"])
    #%%
    # Write required files to disk for solver.
    relative_path_to_input_files = "in/" + ENV_NAME + "_" +  datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "/"
    relative_path_to_robot_file =  relative_path_to_input_files + robot_suffix + ".json" 
    relative_path_to_obj_file =  relative_path_to_input_files + obj_suffix + ".json"
    full_path_to_input_files = solver_path + relative_path_to_input_files
    print("path ", full_path_to_input_files)
    if not os.path.exists(full_path_to_input_files):
        os.makedirs(full_path_to_input_files)

    write_json_file(solver_path + relative_path_to_robot_file , env_sample["robot_file"])
    write_json_file(solver_path + relative_path_to_obj_file, env_sample["obj_file"])

    # print("robot_path", robot_path)        
    # print("obj_path", obj_path)
    # print("scene_path", scene_path)
    #%%
    seqs_names = ["test_seq_predicted_order", "test_seq_original_order", "test_seq_random_order" ]
    output_path = solver_path + "out/"
    DEVICE = "cpu" #"cuda", "cpu"
    # #### Timer should start here?
    # tic = time.perf_counter()
    # toc = time.perf_counter()
    # print(f"Time before actual start of sequence eval: {toc - tic:0.4f} seconds")

    makespans_list = []
    compute_times_list = []
        
    num_tests = 10
    for id_test in tqdm(range(num_tests)):        
        r_seed = random.randint(0, 9999)
        ### Generate sequences
        cmd_str = get_cmd_str_to_generate_sequences(relative_path_to_robot_file,
                                                    relative_path_to_obj_file,
                                                    r_seed)
        exec_cmd(cmd_str)    
        
        ###Load sequences        
        sequences_subdir = last_created_folder(output_path)
        sequences = load_json_file(sequences_subdir.path + "/sequences.json")
        scene = env_sample["scene"] #load_json_file(scene_path)
        samples = build_raw_sample(sequences, scene)
        
        test_dataset = TapasDataset(samples=samples)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=100,
                                shuffle=False) #, num_workers=1, pin_memory=True
        
        ### Generate ordered sequences
        for batch in test_loader:
            model.eval()
            with torch.no_grad():
                outputs = model(observation=batch["observation"].to(DEVICE), src_key_padding_mask=batch["src_key_padding_mask"].to(DEVICE))
                scaled_output = denormalize_predicted_targets(outputs.flatten(), normalization_params)
                
                current_batch_size = batch["observation"].shape[0]
                idx = list(range(0, current_batch_size))
                #print("idx", idx)
                idx_sorted = sorted(idx, key=lambda i : scaled_output[i] )
                print("idx_sorted", idx_sorted)        
                print("scaled_output", scaled_output)
                print("sorted scaled_output", [scaled_output[i].item() for i in idx_sorted])
                
                ### Write ordered sequences to file    
                full_filename = full_path_to_input_files + "test_seq_predicted_order_" + robot_suffix + "_" + obj_suffix + ".json"    
                write_seq_file(full_filename, idx_sorted, test_dataset.samples, batch["indexes"])
                
                print("idx", idx)
                full_filename = full_path_to_input_files + "test_seq_original_order_" + robot_suffix + "_" + obj_suffix + ".json"    
                write_seq_file(full_filename, idx, test_dataset.samples, batch["indexes"])
                
                random.shuffle(idx)
                print("idx", idx)
                full_filename = full_path_to_input_files + "test_seq_random_order_" + robot_suffix + "_" + obj_suffix + ".json"    
                write_seq_file(full_filename, idx, test_dataset.samples, batch["indexes"])
                
                
        ### Evaluate  sequences.    
        for seq_name in seqs_names:
            seq_fullname = seq_name + "_" + robot_suffix + "_" + obj_suffix 
            relative_path_to_seq_file = relative_path_to_input_files + seq_name + "_" + robot_suffix + "_" + obj_suffix + ".json"
            cmd_str = get_cmd_str_to_plan_for_sequence(relative_path_to_robot_file, 
                                                    relative_path_to_obj_file, 
                                                    relative_path_to_seq_file, 
                                                    r_seed)
            exec_cmd(cmd_str)
            
            makespans, compute_times = get_makespans_and_compute_times(output_path)
            #plot_name = "envId_" + str(env_id) + "_test_id_" + str(id_test) + "_" + seq_fullname             
            #plot_makespans(makespans, compute_times, plot_name)
            
            run_name = "envId_" + str(env_id) + "_test_id_" + str(id_test)
            run = wandb.init(project="TestTAMPFormer_" + "envId_" + str(env_id), name=run_name)        
            m = list(makespans.values())
            c = list(compute_times.values())
            for i in range(len(c)):
                wandb.log({"Makespan"+ seq_fullname: m[i]}, step=c[i])
                wandb.log({"MinMakespan"+ seq_fullname: min(m[0:i+1]) }, step=c[i])
            run.finish()                    


#%%
if __name__ == '__main__':
    run_testing()
#%%
