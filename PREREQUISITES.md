## Prerequisites
- Install [docker](https://docs.docker.com/engine/install/ubuntu/) and also [follow the post installation steps](https://docs.docker.com/engine/install/linux-postinstall/) to be able to run docker without using sudo. 

  Make sure you can execute the following command:
  ```
  docker run hello-world
  ```
  Your output should resemble the following output:
  ```
  Hello from Docker!
  This message shows that your installation appears to be working correctly.

  To generate this message, Docker took the following steps:
  1. The Docker client contacted the Docker daemon.
  2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
      (amd64)
  3. The Docker daemon created a new container from that image which runs the
      executable that produces the output you are currently reading.
  4. The Docker daemon streamed that output to the Docker client, which sent it
      to your terminal.

  To try something more ambitious, you can run an Ubuntu container with:
  $ docker run -it ubuntu bash

  Share images, automate workflows, and more with a free Docker ID:
  https://hub.docker.com/

  For more examples and ideas, visit:
  https://docs.docker.com/get-started/
  ```



- Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 

  Make sure you can execute the following command:

  ```
  docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
  ```
  Your output should resemble the following output:
  ```
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
  | N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+

  +-----------------------------------------------------------------------------+
  | Processes:                                                                  |
  |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
  |        ID   ID                                                   Usage      |
  |=============================================================================|
  |  No running processes found                                                 |
  +-----------------------------------------------------------------------------+
  ```