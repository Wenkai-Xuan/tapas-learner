# tapas-containers

It is higly recomnded to use the prebuilt images [`miguelzamoram/tapas-container-cpu`](https://hub.docker.com/repository/docker/miguelzamoram/tapas-container/general) or [`miguelzamoram/tapas-container-gpu`](https://hub.docker.com/repository/docker/miguelzamoram/tapas-container/general). However if you want to build the images locally you can follow the commands below. 

Note that you will also have do modify the compose file e.g `gpu-docker-compose-local.yml` and update the image entry to be `custom-tapas-container-gpu`.

## CPU container    
    docker build -t custom-tapas-container-cpu -f tapas-dockerfile-cpu .
    

## GPU container    
    docker build -t custom-tapas-container-gpu -f tapas-dockerfile-gpu .