## Config Files
Change user specific in the files e.g. path. 
Camera matrix: configs/camera/16x256x256.py
Plucker embedding:  configs/plucker/16x256x256.py

## Train (80GB, reduce batch size to 3 for 40GB)
Download OpenSora-v1-16x256x256.pth from OpenSora repo and change path in config file.

Camera matrix: scripts/run_train_256.sh
Plucker embedding: scripts/run_train_256_plucker.sh

## Inference
Change path in following files and run them. By default, results will be stored in OpenSora/outputs 
Camera matrix: scripts/inference256.sh
Plucker embedding: scripts/inference256plucker.sh

## Create webpage
1. scripts/mp4togif.py --dir <videos dir>
2. change s3_bucket_name and s3_prefix and run 'scripts/makeweb_generic.py --name <url_name> --dir <gif dir>' 