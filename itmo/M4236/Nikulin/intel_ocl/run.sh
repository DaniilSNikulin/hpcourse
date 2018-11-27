#!/bin/bash

docker container start ocl_daemon
docker exec -it ocl_daemon bash
docker container stop ocl_daemon


