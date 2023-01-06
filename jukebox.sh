#!/bin/bash

# validate Nvidia hardware
lspci | grep -i NVIDIA

# validate Nvidia driver

nvidia-smi -L

# install CUDA driver if needed

sudo apt-get -y install nvidia-cuda-toolkit
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
exec bash # reload shell
# following line validates install
nvcc --version


# mount storage
sudo lsblk
sudo parted /dev/sda --script mklabel gpt mkpart xfspart xfs 0% 100%
sudo mkfs.xfs /dev/sda1
sudo partprobe /dev/sda1
sudo mkdir /datadrive
sudo mount /dev/sda1 /datadrive
sudo blkid

sudo chown azureuser /datadrive

mkdir /datadrive/samples



# install pip and jukebox

sudo apt-get install -y python-is-python3 python3-pip
alias pip=pip3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x Miniconda3-latest-Linux-x86_64.sh
# check hash 1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7
sha256sum Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh # requires interactive - see if can feed from file?
# maybe include directions from https://docs.anaconda.com/anaconda/install/silent-mode/#linux-macos
exec bash # reload shell to get conda command


# the following is from:  https://github.com/openai/jukebox#install

# Required: Sampling
conda create --name jukebox python=3.7.5
conda activate jukebox
conda install -y mpi4py=3.0.3 # if this fails, try: pip install mpi4py==3.0.3
conda install -y pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch

pip install git+https://github.com/openai/jukebox.git
# ^^^ the command above should do all of these? vvv
# git clone https://github.com/openai/jukebox.git
# cd jukebox
# pip install -r requirements.txt
# pip install -e .

# Required: Training
conda install -y av=7.0.01 -c conda-forge 
pip install tensorboardX crc32c
 
# Optional: Apex for faster training with fused_adam
conda install -y pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
