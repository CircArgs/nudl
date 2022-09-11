From nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y curl libcurl4-openssl-dev git python

RUN curl https://nim-lang.org/choosenim/init.sh -sSf | bash -s -- -y

ENV PATH=/root/.nimble/bin:$PATH

RUN nimble install -y nimcuda