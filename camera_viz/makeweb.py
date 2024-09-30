from netvision import HtmlGenerator
import os
import argparse
import boto3
from pathlib import Path
from glob import glob

def upload_directory_to_s3(directory_path, bucket_name, s3_prefix=""):
    s3_client = boto3.client('s3')

    # Walk through the directory and upload files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Create the S3 object key (removing the directory_path part)
            s3_key = os.path.relpath(file_path, directory_path)
            if s3_prefix:
                s3_key = f"{s3_prefix}/{s3_key}"
            
            # Upload the file
            ext = os.path.basename(file).split('.')[-1]
            ExtraArgs = {'ContentType': 'text/html'} if ext=='html' else None
            s3_client.upload_file(file_path, bucket_name, s3_key, ExtraArgs)
            print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='camera_viz', type=str ,help="Path to mp4 files")
    parser.add_argument('--viz_dir', type=str ,help="Path to root dir of viz files")

    args = parser.parse_args()
    local_dir = Path('html')/args.dir
    local_path = local_dir/'page.html'
    webpage = HtmlGenerator(path=str(local_path), 
                            title=f'Camera Visualization', local_copy=True)

    viz_files = sorted(glob(os.path.join(args.viz_dir, '**/*.jpg'), recursive=True))
    table1 = webpage.add_table("idx")
    N_COLUMNS = 3

    for n in range(N_COLUMNS):
        table1.add_column(f'video {n}')
        table1.add_column(f'path {n}')


    col_idx = 0
    row = []
    for viz_file in viz_files:
        row.append(webpage.image(viz_file.replace('camera_viz.jpg','video.gif'), size='256px'))
        row.append(webpage.image(viz_file, size='256px'))
        col_idx += 1
        if col_idx >= N_COLUMNS:
            table1.add_row(row)
            col_idx = 0
            row = []
        
    webpage.return_html()

    # Upload
    directory_to_upload = Path(local_dir)  # Replace with your directory path
    s3_bucket_name = 'phidias'  # Replace with your S3 bucket name
    s3_prefix = f'soon/results/{args.dir}'  # Optional S3 prefix path, if you want to upload to a specific folder within the bucket

    upload_directory_to_s3(directory_to_upload, s3_bucket_name, s3_prefix)



if __name__ == "__main__":
    main()