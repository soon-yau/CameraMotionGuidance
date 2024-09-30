#!/bin/bash
input_dir=$1
echo $dir
# Loop through all directories and subdirectories
find "$input_dir" -type d | while read -r dir; do
    echo $dir
    # Find all .mp4 files in the current directory
    find "$dir" -maxdepth 3 -type f -name "*.mp4" | while read -r file; do
        filename=$(basename -- "$file")
        filename="${filename%.mp4}"
        # Create the images subfolder if it doesn't exist
        images_dir="$dir/images/$filename"

        mkdir -p "$images_dir"

        # Extract images from the .mp4 file using ffmpeg
        filename=$(basename -- "$file")
        filename="${filename%.*}"
        echo $file
        ffmpeg -i "$file" "$images_dir/%02d.png" &
        echo $images_dir

    done
done

