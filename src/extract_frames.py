import os
from pathlib import Path
import cv2
import numpy as np
from torch import nn


def get_frame(filename, n_frames=1):
    """Extract n_frames from each video given in filename"""
    frames = []
    v_cap = cv2.VideoCapture(filename)
    # get total number of rames in each video
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # It evenly sample frames from each video to ensure that n_frames are sampled
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue            
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len

def store_frames(frames, path2store):
    """Stores extracted frames as image in the given path"""
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)
        
def video_to_image(phase='train'):
    """Takes the path to the videos, extract n_fames from each video, stors them as images"""
    extension = ".avi"
    n_frames = 16
    if phase=='train':
        path_data = './data/train_1'
    elif path_data=='test':
        path_data = './data/test'
    for root, dirs, files in os.walk(path_data, topdown=False):
        for name in files:
            if extension not in name:
                continue
            path2vid = os.path.join(root, name)
            frames, vlen = get_frame2(path2vid, n_frames= n_frames)
            print(len(frames))
            if phase=='train':
                path2store = path2vid.replace('train_1', 'train_jpg')
            elif phase=='test':
                path2store = path2vid.replace('test', 'test_jpg')
            path2store = path2store.replace(extension, "")
            print(path2store)
            os.makedirs(path2store, exist_ok= True)
            store_frames(frames, path2store)
            
if __name__=="__main__":
    video_to_image('train')
    
    
