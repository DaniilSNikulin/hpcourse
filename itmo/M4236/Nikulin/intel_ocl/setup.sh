#!/bin/bash

docker build -t nikdan/ocl .

docker run \
    -it \
    --name ocl_deamon \
    --device /dev/dri:/dev/dri \
    --mount type=bind,source="$(pwd)"/shared_folder,target=/home/nikdan/ocl \
    nikdan/ocl clinfo

