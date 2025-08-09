# Base image found at https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Install git and jupyter, git is not necessary for this barebones install but may be needed to pull other repositories
RUN apt-get update && apt-get install -y git \
    && pip install jupyter

RUN apt install cifs-utils -y

# Upgrade pip
WORKDIR /tmp
RUN python -m pip install --upgrade pip

# Install R (requirement for Python pkg)
COPY ./prp/docker/install_r.sh /tmp/install_r.sh

# Requirements.txt must be in the same folder as this dockerfile, contains the packages
# TODO: Automatically pull requirements from github package?
COPY ./prp/docker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mamba-ssm[causal-conv1d] --no-build-isolation

ENV DEBIAN_FRONTEND=noninteractive
RUN echo "tzdata tzdata/Areas select Europe" | debconf-set-selections && \
   echo "tzdata tzdata/Zones/Europe select Amsterdam" | debconf-set-selections

RUN chmod +rwx /tmp/install_r.sh
RUN /tmp/install_r.sh
RUN conda install -y -c ejolly -c conda-forge pymer4
WORKDIR /workspace

COPY ./hmp-ai /workspace/hmp-ai
RUN python -m pip install --no-cache-dir -e /workspace/hmp-ai/src

# Uncomment this if you want to use a custom version of hmp
COPY ./hmp /workspace/hmp_repo
RUN python -m pip install --no-cache-dir -e /workspace/hmp_repo

COPY ./prp/docker/pyrightconfig.json /workspace
# Mounting must be done with added capabilities that are only available after running the dockerfile, so start by running a script that contains that info
COPY ./prp/docker/mount_drive.sh /tmp/mount_drive.sh
RUN chmod +rwx /tmp/mount_drive.sh
RUN mkdir /workspace/data
ENTRYPOINT /tmp/mount_drive.sh 
#ENTRYPOINT ["tail", "-f", "/dev/null"]
