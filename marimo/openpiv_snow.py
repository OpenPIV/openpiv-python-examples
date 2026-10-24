# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv>=0.26.0",
#     "numpy",
#     "matplotlib",
#     "imageio",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App()
@app.cell
def _():
    import importlib.metadata
    print("openpiv", importlib.metadata.version("openpiv"))
    try:
        import openpiv_rust
        print("openpiv-rust available — Rust backend enabled")
    except ImportError:
        print("openpiv-rust not installed — pip install openpiv[rust] for faster Rust backend")
    return



@app.cell
def _():



    # misc
    import datetime
    import math
    import os
    import shutil

    # '%matplotlib inline' command supported automatically in marimo
    # image operation
    import cv2

    # plots
    import matplotlib.pyplot as plt

    return cv2, datetime, math, os, plt, shutil


@app.cell
def _():
    # from pytube import YouTube
    # YouTube('https://youtu.be/2lAe1cqCOXo').streams.first().download()
    # yt = YouTube("http://youtube.com/watch?v=2lAe1cqCOXo")
    # yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    return


@app.cell
def _():
    # thanks to the Medium article 
    # https://towardsdatascience.com/the-easiest-way-to-download-youtube-videos-using-python-2640958318ab
    return


@app.cell
def _():
    # Replace with your desired YouTube URL

    # Download the video
    import yt_dlp

    url = 'https://youtu.be/OHl_0s4qqUY'
    ydl_opts = {'outtmpl': '%(title)s.%(ext)s'}  # Save as video title

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return


@app.cell
def _(cv2, datetime, math, os):
    class FrameExtractor():
        '''
        Class used for extracting frames from a video file.
        '''
        def __init__(self, video_path):
            self.video_path = video_path
            self.vid_cap = cv2.VideoCapture(video_path)
            if not self.vid_cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
        def get_video_duration(self):
            duration = self.n_frames / self.fps
            print(f"Duration: {datetime.timedelta(seconds=duration)}")
        
        def get_n_images(self, every_x_frame):
            n_images = math.floor(self.n_frames / every_x_frame) + 1
            print(f"Extracting every {every_x_frame} frame would result in {n_images} images.")
        
        def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.jpg'):
            if not self.vid_cap.isOpened():
                self.vid_cap = cv2.VideoCapture(self.video_path)
                if not self.vid_cap.isOpened():
                    raise ValueError(f"Cannot open video file: {self.video_path}")
        
            if dest_path is None:
                dest_path = os.getcwd()
            else:
                if not os.path.isdir(dest_path):
                    os.makedirs(dest_path, exist_ok=True)
                    print(f"Created the following directory: {dest_path}")
        
            frame_cnt = 0
            img_cnt = 0

            while True:
                success, image = self.vid_cap.read()
                if not success:
                    break
                if frame_cnt % every_x_frame == 0:
                    img_path = os.path.join(dest_path, f"{img_name}_{img_cnt}{img_ext}")
                    if image is not None:
                        cv2.imwrite(img_path, image)
                        img_cnt += 1
                frame_cnt += 1

            self.vid_cap.release()

    return (FrameExtractor,)


@app.cell
def _(FrameExtractor):
    fe = FrameExtractor(video_path="Super-Large-Scale Flow Visualization with Snow.mp4")
    return (fe,)


@app.cell
def _(fe):
    fe.get_n_images(every_x_frame=4)
    return


@app.cell
def _(fe):
    fe.extract_frames(every_x_frame=4, 
                      img_name='test', 
                      dest_path="test_images")
    return


@app.cell
def _():
    import glob
    import re
    image_list = glob.glob("./test_images/test_*.jpg")
    image_list.sort(key=lambda x: float(re.findall('(\\d+)', x)[0]))
    print(image_list)
    return (image_list,)


@app.cell
def _(image_list, plt):
    import imageio
    from openpiv.piv import simple_piv
    plt.imshow(imageio.imread(image_list[130]))
    return imageio, simple_piv


@app.cell
def _(image_list, imageio, simple_piv):
    import numpy as np

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    for i in range(130, 134, 1):
        print(i)
        a = rgb2gray(imageio.imread(image_list[i]))
        b = rgb2gray(imageio.imread(image_list[i+1]))
        # plt.figure()
        # plt.imshow(np.sum(a,axis=2),cmap="gray")
        # plt.figure()
        # plt.imshow(np.sum(b,axis=2),cmap="gray")
        simple_piv(a, b)
    return


@app.cell
def _(shutil):
    shutil.rmtree("./test_images")
    return


if __name__ == "__main__":
    app.run()
