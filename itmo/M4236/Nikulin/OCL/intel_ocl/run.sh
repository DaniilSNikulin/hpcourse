#!/bin/bash

docker container start ocl_deamon
docker exec -it ocl_deamon bash
docker container stop ocl_deamon


