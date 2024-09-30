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
WAIT_TIME=10m

# Find all "images" directories
find "$BASE_DIR" -type d -name "images" | while read -r images_dir; do
  # Determine the parent directory of "images"
  parent_dir=$(dirname "$images_dir")
  echo "Total number of parent dir: ${#parent_dir[@]}"
  
  # Create "colmap" directory at the same level as "images"
  colmap_dir="$parent_dir/colmap"
  if [ -d "$colmap_dir" ]; then
    rm -rf "$colmap_dir"
  fi
  mkdir -p "$colmap_dir"

  # Get list of subdirectories inside "images"
  subdirs=( $(find "$images_dir" -mindepth 1 -maxdepth 3 -type d) )
  echo "Total number of subdirectories: ${#subdirs[@]}"
  # Process subdirectories in batches
  for (( i=0; i<${#subdirs[@]}; i+=BATCH_SIZE )); do
    batch=( "${subdirs[@]:i:BATCH_SIZE}" )
    
    for subdir in "${batch[@]}"; do
      subdir_name=$(basename "$subdir")
      subdir_dir="$colmap_dir/$subdir_name"
      mkdir -p "$subdir_dir"
      rm -rf "${subdir_dir}/*"
      
      # Run colmap in the background
      echo -e "${RED}Processing: $i $subdir_dir${NC}"
      random_number=$(( ( RANDOM % (max - min + 1) ) + min ))
      /opt/conda/bin/colmap automatic_reconstructor --workspace_path "$subdir_dir" --image_path "$images_dir/$subdir_name" --random_seed "$random_number" &
    done
    
    # Wait for the batch to complete
    echo "Waiting for $WAIT_TIME..."
    sleep "$WAIT_TIME"
    
    # Kill all running colmap processes
    echo "Terminating all colmap processes..."
    pkill -f /opt/conda/bin/colmap
  done
done

echo "Colmap estimation done"
