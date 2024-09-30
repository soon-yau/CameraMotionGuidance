TODO:
- [x] confirm the basic environment setup 
- [x] build the error computation script
- [ ] automate the whole process: run COLMAP multiple times, get errors from the evaluation script and average them
- [ ] support user-defined intrinsics


# Setting up and running [COLMAP](https://colmap.github.io/) on runai jobs
1. Installing `colmap` from the conda-forge channel can be achieved by adding conda-forge to your channels with:

    ```
    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict

    # Once the conda-forge channel has been enabled, colmap can be installed with conda:
    $ conda install colmap
    ```
    The command lines above have been tested on a run.ai job created from this [docker image](docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/runai/clio-base-demo:0.06
) and confirmed working.


2. Once installed, run colmap on our generated videos:
    ```
    # The project folder must contain a folder "images" with all the images.
    $ DATASET_PATH=/path/to/project

    $ colmap automatic_reconstructor \
        --workspace_path $DATASET_PATH \
        --image_path $DATASET_PATH/images
    ```

3. The results are stored in `image.bin` and/or `image.txt`. If the program saves only `.bin` files, one can convert them to `txt` through the following command:
    ```
    $ colmap model_converter --input_path <folder_containing_bin> --output_path <output_folder_for_txt> --output_type TXT
    ```

# Error computation
Comparing the output of COLMAP with the ground truth extrinsic