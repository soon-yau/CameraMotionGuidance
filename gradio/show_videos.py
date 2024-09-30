import gradio as gr
import argparse
import os
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dirs", type=str, required=False, nargs='*', default=['outputs/realestate_epoch9_34000/','outputs/samples'])
    parser.add_argument("--text_file", type=str)
    return parser.parse_args()

args = parse_args()

def get_video_number(file_name):
    # Split the file name on '_', take the second last part, and convert it to an integer
    return int(file_name.split('.')[-2].split('/')[-1].split('_')[1])

galleries = []
video_files = []
demo = gr.Blocks()
with demo:
#with gr.Blocks() as demo:
    with gr.Column(scale=1, min_width=600):
        video_dir = args.video_dirs[0]
        videos = sorted(glob(os.path.join(video_dir,'*.mp4')), key=get_video_number)
        video_files.append(videos)
        gallery1 = gr.Video(value=videos[0], height=512, width=512,
                            label='Finetuned')
        galleries.append(gallery1)

    if len(args.video_dirs) > 1:
        with gr.Column(scale=1, min_width=600):
            video_dir = args.video_dirs[1]
            videos = sorted(glob(os.path.join(video_dir,'*.mp4')), key=get_video_number)
            video_files.append(videos)
            gallery2 = gr.Video(value=videos[0], height=512, width=512,
                            label='Pretrained')
            galleries.append(gallery2)    
    
    if args.text_file:
        with open(args.text_file, "r") as file:
            # Read all lines from the file and store them in a list
            texts = file.readlines()
        text = gr.Dropdown(texts, label='text prompt', type='index')
        text.select(fn=lambda i: [g[i] for g in video_files], inputs=[text], outputs=[*galleries])

if __name__ == "__main__":   
    demo.launch(share=False)