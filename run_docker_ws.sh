IMAGE_NAME="openpcdet_cu117"
CONTAINER_NAME="openpcdet_jun"
printf "[Docker] Using docker image %s\n" $IMAGE_NAME

# Port bindings: -p host_port:container_port
data="/data_hdd/datasets"
workspace="data_hdd/jun/repos"
# data="/home/jun/DockerRepos/OpenPCDet/data"
# workspace="/home/jun/DockerRepos/OpenPCDet"
docker rm ${CONTAINER_NAME}
docker run -it --gpus 'all,"capabilities=compute,utility,graphics"' \
           --shm-size=8g \
           -v $data:/data \
           -v $workspace:/workspace \
           -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
           --name ${CONTAINER_NAME} \
           ${IMAGE_NAME}
           
# -it --gpus 'all,"capabilities=compute,utility,graphics"'