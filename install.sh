sudo apt-get install git-lfs libgl1 -y
conda create -n opensora python=3.10
conda activate opensora

conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers==0.0.22.post7

pip install https://phidias.s3.us-west-2.amazonaws.com/kaiz/softwares/FlashS3DataLoader.zip
pip install packaging ninja
pip install flash-attn --no-build-isolation

git clone https://git.azr.adobeitc.com/scheong/apex 

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://git.azr.adobeitc.com/scheong/apex 
git clone https://git.azr.adobeitc.com/scheong/Open-Sora
cd Open-Sora
pip install -v .
