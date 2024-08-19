# tapas-learner


## Preliminaries
- Setup docker and make sure you have all the [PREREQUISITES](PREREQUISITES.md).

- Clone this repo with all its submodules.
    ```
    git clone --recurse-submodules https://github.com/tapas-dataset/tapas-learner.git
    ```    
    If you have already clone the repo without the `--recurse-submodules` flag, use the following command:
    ```    
    git submodule update --init --recursive
    ```    

## Training
- Launch docker
    - GPU enabled
        ```
        CUSTUM_UID=$(id -u) CUSTUM_GID=$(id -g) docker compose -f gpu-docker-compose-local.yml up -d
        docker exec -it tapasContainer bash
        ```
    - For CPU only, use `cpu-docker-compose-local.yml`
- Compile solver
    ```
    cd /home/tapas/multi-agent-tamp-solver/24-data-gen/
    make
    ```
- Setup wandb:     
    - Either login to wandb
        ```    
        wandb login --relogin
        ```
    - Or set the mode to  offiline:
        ```    
        export WANDB_MODE=offline
        ```

- Run the code
    ```
    cd /home/tapas/src/
    python train.py
    ```

## Evaluation
- Run the code
    ```
    cd /home/tapas/src/
    python test.py
    ```


## Cleanup
Remember to stop the containers after exiting.
```
docker compose down
```

## TODO
    - General code cleanup

# Citation
If you use this codebase in your research, please cite the following paper

```
@article{authors,
  author    = {Zamora, Miguel and Hartmann, Valentin N. and Coros, Stelian},
  title     = {TAPAS: A Dataset for Task Assignment and Planning for Multi Agent Systems},
  year      = {2024},
  journal   = {Workshop on Data Generation for Robotics at Robotics, Science and Systems '24}
}
```