#%%
# Imports
import os
import sys
import shutil 

import numpy as np

import psutil
import torch.nn as nn
from tqdm import tqdm

import wandb
import random
from datetime import datetime


from tapasformer import *
from tapas_dataset import *
from tapas_utils import *

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_training_loop(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    #%%
    # Define params
    ENV_NAME = cfg["env_name"] # "random" "husky" "conveyor" "shelf"
    NUM_THREADS = cfg["num_threads"]
    LEARNING_RATE = cfg["learning_rate"]
    NUM_EPOCHS = cfg["num_epochs"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    evals_per_epoch = cfg["evals_per_epoch"]
    batch_size = cfg["batch_size"]
    train_samples_id = cfg["train_samples_id"]   
    test_samples_id = cfg["test_samples_id"]
        
    #%%
    # Downloading dataset and converting to iterable dataset is faster than using load_dataset() with streaming=True.
    # The iterable dataset will stream from local files. This approach is only feasible provided enough disk space.
    # See: https://huggingface.co/docs/datasets/en/stream#convert-from-a-dataset
    # Note: Avoid download_mode='force_redownload' when using the num_workers argument for the DataLoader() .
    dataset = load_dataset("data_replan",
                        data_files={'train': "conveyor_5_rela_train.parquet",
                                    'test': "conveyor_5_rela_test.parquet"})  # ,token=True, download_mode='force_redownload'
    #print("pid", os.getpid(), dataset)

    # Set num_shards >= num_workers.
    # See: https://discuss.huggingface.co/t/num-worker-with-iterabledataset/58914/2
    num_shards_train = min(len(dataset['train']), NUM_THREADS)
    num_shards_test = min(len(dataset['test']), NUM_THREADS)
    iterable_train_dataset = dataset['train'].to_iterable_dataset(num_shards=num_shards_train)
    iterable_test_dataset = dataset['test'].to_iterable_dataset(num_shards=num_shards_test)

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
    
    test_loader = DataLoader(test_ds_torch, batch_size=batch_size, num_workers=NUM_THREADS, persistent_workers=True)
    train_loader = DataLoader(train_ds_torch, batch_size=batch_size, num_workers=NUM_THREADS, persistent_workers=True)
    max_num_batches_train = int(len(dataset['train']) / batch_size)
    max_num_batches_test = int(len(dataset['test']) / batch_size)
    # print("-----------------")
    # print(len(dataset['train']), len(dataset['test']), max_num_batches_train, max_num_batches_test)
    # print("-----------------")

    obs_dim = get_raw_features_dim()
    
    LOG_DIR = os.path.join("./logs", datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    # # Compute sequence length
    def get_makespan_stats(flat_samples):    
        total_makespans = 0
        max_makespans = 0
        min_makespans = flat_samples[0]['metadata']['makespan']
        cntr = 0
        for s in flat_samples:    
            total_makespans = total_makespans + s['metadata']['makespan']         
            max_makespans = max(max_makespans , s['metadata']['makespan'])
            min_makespans = min(min_makespans, s['metadata']['makespan'])
            if s['metadata']['makespan'] == 0:
                print("Zepo makespan found", cntr)
            cntr += 1
        
        avg_makespans = total_makespans / len(flat_samples)    
        return avg_makespans, max_makespans, min_makespans
    
    avg_makespans, max_makespans, min_makespans = get_makespan_stats(dataset['train']['metadata'])
    print(avg_makespans, max_makespans, min_makespans)
    
    avg_makespans_test, max_makespans_test, min_makespans_test = get_makespan_stats(dataset['test']['metadata'])
    print(avg_makespans_test, max_makespans_test, min_makespans_test)    
    
    normalization_params = {"target_min": 0.0, "target_max": max_makespans + 400, "new_min": -1, "new_max": 1}
    write_json_file(LOG_DIR + "/normalization_params.json", normalization_params)

    
    # Log params
    params = {"DEVICE": DEVICE, "LEARNING_RATE": LEARNING_RATE, "NUM_EPOCHS": NUM_EPOCHS, "NUM_THREADS": NUM_THREADS,
              "batch_size": batch_size,
              "avg_makespans": avg_makespans, "max_makespans": max_makespans, "min_makespans": min_makespans,
              "normalization_params": normalization_params}
    write_json_file(LOG_DIR + "/params.json", params)
        
    
    run = wandb.init(project="TapasFormer_env_" + ENV_NAME, config=params)
    #wandb.login()
    #wandb.log({"normalization_params": normalization_params})
    wandb.save(LOG_DIR + "/normalization_params.json")
    wandb.save(LOG_DIR + "/params.json")
            
    model = TapasFormer(obs_dim=obs_dim, normalization_params=normalization_params).to(DEVICE)
    criterion = nn.MSELoss()
    optimiser = torch.optim.RAdam(model.parameters(), weight_decay=0, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.1, total_iters= max_num_batches_train * NUM_EPOCHS)

    print("\npid", os.getpid(), "Start training loop!")
    print("\npid", os.getpid(), "max_num_batches", max_num_batches_train)
    # print(len(dataset['train']), len(dataset['test']), max_num_batches_train, max_num_batches_test)
    step = 0  
    
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epoch",
                               position=0, leave=True)
    for epoch in epoch_bar:
        train_ds_torch.set_epoch(epoch)
        epoch_bar.set_description(f"Epoch {epoch}.")
        epoch_bar.refresh()
        print("\nEpoch ", epoch)

        train_bar = tqdm(train_loader,
                               total=max_num_batches_train,
                               desc="Train",
                               position=0, leave=True)                
        for i, batch in enumerate(train_bar):
            train_bar.set_description(f"Training batch {i}.")
            # Log performance
            if i % int(max_num_batches_train / evals_per_epoch) == 0:
                model.eval()                
                with torch.no_grad():
                    outputs = model(observation=batch["observation"].to(DEVICE),
                                    src_key_padding_mask=batch["src_key_padding_mask"].to(DEVICE))    
                    
                    normalized_targets = normalize_targets(batch["targets"], normalization_params).to(DEVICE)
                    loss = criterion(normalized_targets, outputs.flatten())
                    
                    wandb.log({"Loss/train": loss}, step=step)
                    if step == 0:
                        wandb.log({"Lr": LEARNING_RATE}, step=step)
                    else:
                        wandb.log({"Lr": optimiser.param_groups[0]['lr']}, step=step)
                    
                    eval_data = eval_batch(batch["targets"].to(DEVICE), outputs.flatten(), normalization_params)
                    sample_size = eval_data[2]
                    normalized_total_error = (1.0 / sample_size) * eval_data[0]
                    scaled_total_error = (1.0 / sample_size) * eval_data[1]
                    wandb.log({"Loss/avg_norm_train": normalized_total_error}, step=step)
                    wandb.log({"Loss/avg_unorm_train": scaled_total_error}, step=step)
                    
                    
                    eval_data = torch.zeros((3))
                    test_bar = tqdm(test_loader,
                                      total=max_num_batches_test,
                                      desc="Test batch iter",
                                      position=0, leave=True)
                    for j, test_batch in enumerate(test_bar):
                        test_bar.set_description(f"Testing batch {j}.")
                        
                        outputs = model(observation=test_batch["observation"].to(DEVICE), src_key_padding_mask=test_batch["src_key_padding_mask"].to(DEVICE))
                        eval_data += eval_batch(test_batch["targets"].to(DEVICE), outputs.flatten(), normalization_params)
                    test_bar.refresh()
                    
                    sample_size = eval_data[2]
                    normalized_total_error = (1.0 / sample_size) * eval_data[0]
                    scaled_total_error = (1.0 / sample_size) * eval_data[1]
                    wandb.log({"Loss/avg_norm_test": normalized_total_error}, step=step)
                    wandb.log({"Loss/avg_unorm_test": scaled_total_error}, step=step)
                
            model.train()
            model.zero_grad()
            
            outputs = model(observation=batch["observation"].to(DEVICE), src_key_padding_mask=batch["src_key_padding_mask"].to(DEVICE))
            
            normalized_targets = normalize_targets(batch["targets"], normalization_params).to(DEVICE)
            loss = criterion(normalized_targets, outputs.flatten())        
            loss.backward()
            
            optimiser.step()
            scheduler.step()       

            current_batch_size = batch["observation"].shape[0] 
            step = step + current_batch_size
            wandb.log({"Loss/train": loss}, step=step)
            wandb.log({"Lr": optimiser.param_groups[0]['lr']}, step=step)
            # #To check that everything happens on the same thread and that the batches are different.
            # print("\npid", os.getpid(), "iter", i, "folders", len(batch['folders'][0]))
            # print("pid", os.getpid(), "iter", i, batch['folders'][0])

            #if #int(epoch) % 5 == 0:        
            model_path = os.path.join(LOG_DIR, f"model_{step}.pth")
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"model_latest.pth"))
            run.log_model(path=model_path, name=f"model_{step}.pth")
            
        train_bar.refresh()
    epoch_bar.refresh()
    print("\npid", os.getpid(), "Training loop done!")
    
#%%
if __name__ == '__main__':
    run_training_loop()
#%%
