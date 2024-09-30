#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if a directory path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

min=0
max=2147483647
BASE_DIR="$1"
BATCH_SIZE=50
WAIT_TIME=15m

# Find all "images" directories
images_dirs=($(find "$BASE_DIR" -type d -name "images"))

# Process "images" directories in batches
for (( i=0; i<${#images_dirs[@]}; i+=BATCH_SIZE )); do
  batch=( "${images_dirs[@]:i:BATCH_SIZE}" )

  for images_dir in "${batch[@]}"; do
    # Determine the parent directory of "images"
    parent_dir=$(dirname "$images_dir")
    echo "Processing images in directory: $images_dir"
    
    # Create "colmap" directory at the same level as "images"
    colmap_dir="$parent_dir/colmap"
    if [ -d "$colmap_dir" ]; then
      rm -rf "$colmap_dir"
    fi
    mkdir -p "$colmap_dir"

    # Run COLMAP in the background for the current images directory
    echo -e "${RED}Processing: $images_dir${NC}"
    random_number=$(( ( RANDOM % (max - min + 1) ) + min ))
    /opt/conda/bin/colmap automatic_reconstructor --workspace_path "$colmap_dir" --image_path "$images_dir" --random_seed "$random_number" &
  done

  # Wait for the batch to complete
  echo "Waiting for $WAIT_TIME..."
  sleep "$WAIT_TIME"
  
  # Kill all running COLMAP processes
  echo "Terminating all COLMAP processes..."
  pkill -f /opt/conda/bin/colmap
done

echo "Colmap estimation done"
