from glob import glob
from pathlib import Path
import argparse
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def convert_(source_file, overwrite):
    gif_dir = Path(source_file).parent/'gif'
    dest_file = gif_dir/os.path.basename(source_file).replace('.mp4','.gif')
    if not os.path.isfile(dest_file) or overwrite:
        os.makedirs(gif_dir, exist_ok=True)
        video_clip = VideoFileClip(source_file)        
        video_clip.write_gif(dest_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help="Path to mp4 files")
    parser.add_argument('--overwrite', action="store_true", help='overwrite existinf gif file')
    args = parser.parse_args()
    convert = partial(convert_, overwrite=args.overwrite)
    source_files = glob(os.path.join(args.dir,'**/*.mp4'), recursive=True)
    with mp.Pool() as pool:
        pool.map(convert, source_files)

if __name__ == "__main__":
    main()
