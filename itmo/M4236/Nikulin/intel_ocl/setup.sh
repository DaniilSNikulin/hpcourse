#!/bin/bash

docker build -t nikdan/ocl .

docker run -d \
    -it \
    --name ocl_daemon \
    --device /dev/dri:/dev/dri \
    --mount type=bind,source="$(pwd)"/shared_folder,target=/home/nikdan/ocl \
    nikdan/ocl bash

docker container stop ocl_daemon

