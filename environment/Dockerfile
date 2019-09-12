FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-cuda9.2-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade --no-cache-dir --upgrade-strategy=only-if-needed \
      tensorflow==1.8.0
