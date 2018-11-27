
FROM chihchun/hashcat-beignet:latest
#MAINTAINER Daniil Nikulin <daniil.s.nikulin@ya.ru>
COPY ubuntu18.04.sources.list /tmp/
RUN cp /tmp/ubuntu18.04.sources.list /etc/apt/sources.list \
        && \
        apt-get update && apt-get install -y \
            apt-utils git less sudo \
            gcc g++ cmake \
            clinfo ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-clhpp-headers \
            beignet beignet-dev beignet-opencl-icd \
            vim fish \
        && \
        apt-get clean \
        && \
        useradd -m nikdan -p nikdan -G sudo \
        && \
        mkdir /home/nikdan/ocl
        && \
        clinfo
WORKDIR /home/nikdan

