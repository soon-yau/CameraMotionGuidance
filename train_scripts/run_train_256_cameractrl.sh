#!/bin/bash
# docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/runai/clio-base-demo:0.15
PROJECT_NAME=$1

SENSEI_ROOT='/sensei-fs/users/scheong/github'
WORKSPACE=/sensei-fs/users/scheong/scratch/$1
USER=scheong
# Copy OpenSora directory
REPO='Open-Sora'
src_dir=$SENSEI_ROOT/$REPO
dst_dir=$WORKSPACE

# Copy OpenSora directory
if [ -d "$WORKSPACE" ]; then
    rm -rf $WORKSPACE
fi
echo "Copying code to $WORKSPACE."
mkdir -p $WORKSPACE
echo "dst_dir = ${dst_dir}"
rsync -av --exclude='outputs' --exclude='wandb' --exclude='.git' --exclude='ckpts' \
--exclude='*.mp4' --exclude='*.gif' --exclude='notebooks' --exclude='html' --exclude='*.pyc' \
--exclude='*.jpg' --exclude='camera_viz*'  ${src_dir} ${dst_dir} 
echo "Current directory is: $(pwd)"

# Permission. Change your user
export HOME=/home/$USER
sudo mkdir -p $HOME

# give $user ownership of the cache
sudo chown -R $USER: $HOME
mkdir -p /home/$USER/.cache
sudo chown -R $USER: /home/$USER/.cache/
sudo chown -R $USER: /opt/venv/lib/python3.10/site-packages/

# Colmap
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install colmap

# Install
pip install packaging ninja
pip install https://phidias.s3.us-west-2.amazonaws.com/kaiz/softwares/FlashS3DataLoader.zip && flash_s3_dataloader_patch
pip install flash-attn --no-build-isolation

# Build apex
cd $SENSEI_ROOT/apex
echo "Current directory is: $(pwd)"
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

chmod -R 777 ${dst_dir}
dst_dir=$WORKSPACE/$REPO
cd $dst_dir
echo "Current directory is: $(pwd)"
pip install -v .

# Run training
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_camera.py configs/plucker/16x256x256_resume.py
#--ckpt-path /sensei-fs/users/scheong/github/Open-Sora/ckpts/OOO/OpenSora-v1-HQ-16x512x512.pth 

