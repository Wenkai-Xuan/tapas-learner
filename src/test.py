#%%
# Imports
import os

import torch.nn as nn
from tqdm import tqdm

import wandb
import random
from datetime import datetime

from makespan_utils import *
from TapasFormer import *
from TapasDataset import *
from TapasUtils import *

from pathlib import Path
#%%
# Define params
ENV_NAME = "random" # "random" "husky" "conveyor" "shelf"
#%%
# Downloading dataset and converting to iterable dataset is faster than using load_dataset() with streaming=True.
# The iterable dataset will stream from local files. This approach is only feasible provided enough disk space.
# See: https://huggingface.co/docs/datasets/en/stream#convert-from-a-dataset
# Note: Avoid download_mode='force_redownload' when using the num_workers argument for the DataLoader().
dataset = load_dataset("MiguelZamoraM/TAPAS",
                       data_files={'train': "env/" + ENV_NAME + "/train/samples_*",
                                   'test': "env/" + ENV_NAME + "/test/samples_*"})  # download_mode='force_redownload'
#print("pid", os.getpid(), dataset)

#Set num_shards >= num_workers.
# See: https://discuss.huggingface.co/t/num-worker-with-iterabledataset/58914/2
iterable_train_dataset = dataset['train'].to_iterable_dataset(num_shards=8)
iterable_test_dataset = dataset['test'].to_iterable_dataset(num_shards=8)

#print("pid", os.getpid(), iterable_train_dataset)
#print("pid", os.getpid(), iterable_test_dataset)

iter_train_ds_processed = iterable_train_dataset.map(process_data, remove_columns=iterable_train_dataset.column_names)
iter_test_ds_processed = iterable_test_dataset.map(process_data, remove_columns=iterable_test_dataset.column_names)

#print("pid", os.getpid(), iter_train_ds_processed)
#print("pid", os.getpid(), iter_test_ds_processed)

shuffled_train_dataset = iter_train_ds_processed.shuffle(seed=42, buffer_size=128)
shuffled_test_dataset = iter_test_ds_processed.shuffle(seed=42, buffer_size=128)

train_ds_torch = shuffled_train_dataset.with_format("torch")
test_ds_torch = shuffled_test_dataset.with_format("torch")

#print("pid", os.getpid(), train_ds_torch)
#print("pid", os.getpid(), test_ds_torch)
#%%
#Load envs instances
#Generate sequences
#Load sequences
#Run inference
#%%
### Load pretrained model.
root_path = "/home/tapas/src/logs/2024_07_03_12_19_25/"
solver_path = "/home/tapas/multi-agent-tamp-solver/24-data-gen/"

model_path = root_path + "/model_latest.pth"
state_dict = torch.load(model_path)

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
target_num_robots = 4
target_num_objs = 3
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
# import csv
# import pandas as pd
# 
# files = ["MinMakespantest_seq_predicted_order_num_robots_4_num_obj_1.csv", 
#          "Makespantest_seq_predicted_order_num_robots_4_num_obj_1.csv",
#          "MinMakespantest_seq_original_order_num_robots_4_num_obj_1.csv",
#          "Makespantest_seq_original_order_num_robots_4_num_obj_1.csv"]
# 
# legends = ["MinMakespan_predicted_order",
#            "Makespan_predicted_order",
#            "MinMakespan_original_order",
#            "Makespan_original_order"]
# 
# fig, ax = plt.subplots()
# for i in range(len(files)):
#     # read csv
#     data = pd.read_csv("plots/" + files[i])
#     np_data = data.to_numpy()
#     ax.plot(np_data[:, 0], np_data[:, 1], '-', label = legends[i])
#     #ax.fill_between(np_data[:, 0], np_data[:, 2], np_data[:, 3], alpha=0.2)
# 
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True)
# 
# ax.legend(legends)


#%%
