# /// script
# dependencies = ["pytube @ git+https://github.com/pytube/pytube"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    # packages added via marimo's package management: git+https://github.com/pytube/pytube !pip install git+https://github.com/pytube/pytube --upgrade
    return


@app.cell
def _():
    from pytube import YouTube
    # misc
    import os
    import shutil
    import math
    import datetime
    # plots
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    # image operation
    import cv2

    return YouTube, cv2, datetime, math, os, plt, shutil


@app.cell
def _():
    # from pytube import YouTube
    # YouTube('https://youtu.be/2lAe1cqCOXo').streams.first().download()
    # yt = YouTube('http://youtube.com/watch?v=2lAe1cqCOXo')
    # yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    return


@app.cell
def _(YouTube):
    # thanks to the Medium article 
    # https://towardsdatascience.com/the-easiest-way-to-download-youtube-videos-using-python-2640958318ab
    video = YouTube('https://www.youtube.com/watch?v=OHl_0s4qqUY&ab_channel=umnSAFL')
    return (video,)


@app.cell
def _(video):
    video.streams.filter(file_extension = "mp4").all()
    return


@app.cell
def _(video):
    video.streams.get_by_itag(135).download()
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
            self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
        def get_video_duration(self):
            duration = self.n_frames/self.fps
            print(f'Duration: {datetime.timedelta(seconds=duration)}')
        
        def get_n_images(self, every_x_frame):
            n_images = math.floor(self.n_frames / every_x_frame) + 1
            print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')
        
        def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg'):
            if not self.vid_cap.isOpened():
                self.vid_cap = cv2.VideoCapture(self.video_path)
        
            if dest_path is None:
                dest_path = os.getcwd()
            else:
                if not os.path.isdir(dest_path):
                    os.mkdir(dest_path)
                    print(f'Created the following directory: {dest_path}')
        
            frame_cnt = 0
            img_cnt = 0

            while self.vid_cap.isOpened():
            
                success,image = self.vid_cap.read() 
            
                if not success:
                    break
            
                if frame_cnt % every_x_frame == 0:
                    img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                    cv2.imwrite(img_path, image)  
                    img_cnt += 1
                
                frame_cnt += 1
        
            self.vid_cap.release()
            # cv2.destroyAllWindows()
    return (FrameExtractor,)


@app.cell
def _(FrameExtractor):
    fe = FrameExtractor('/home/user/Documents/repos/openpiv-python-examples/notebooks/Super-Large-Scale Flow Visualization with Snow.mp4')
    return (fe,)


@app.cell
def _(fe):
    fe.get_video_duration()
    return


@app.cell
def _(fe):
    fe.get_n_images(every_x_frame=4)
    return


@app.cell
def _(fe):
    fe.extract_frames(every_x_frame=4, 
                      img_name='test', 
                      dest_path='test_images')
    return


@app.cell
def _():
    import glob, re
    image_list = glob.glob('./test_images/test_*.jpg')
    # print(image_list)
    image_list.sort(key=lambda x: float(re.findall('(\\d+)', x)[0]))
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
        # plt.imshow(np.sum(a,axis=2),cmap='gray')
        # plt.figure()
        # plt.imshow(np.sum(b,axis=2),cmap='gray')
        simple_piv(a, b)
    return


@app.cell
def _(shutil):
    shutil.rmtree('./test_images')
    return


if __name__ == "__main__":
    app.run()
