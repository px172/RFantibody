FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.10 python3-pip pipx vim make wget

RUN alias "python"="python3.10"

RUN groupadd -g 1000 px172 && \
    useradd -u 1000 -g px172 -m px172

# Make a virtual env that we can safely install into

RUN python3 -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install poetry

# Set the working directory to the user's home directory
WORKDIR /home

ENTRYPOINT /bin/bash
