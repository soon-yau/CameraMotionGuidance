#!/bin/bash

# Specify the location of the folders
location=$1

# Use 'find' to locate all directories in the specified location
find "$location" -type d | while read -r folder; do
    # Extract the folder name
    folder_name=$(basename "$folder")

    # Print the folder name
    echo "$folder"/camera.txt
    #echo "Executing Python script in folder: $folder/${folder_name}/camera.txt"
    python visualize_trajectory.py --pose_file_path ${folder}/camera.txt --zval 0.075 --base_xval 0.1 --output_image_path ../view_viz/
done
echo "Completed."