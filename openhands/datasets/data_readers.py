import os
from glob import glob
import pickle
import cv2
import numpy as np
from natsort import natsorted

def load_frames_from_folder(frames_folder, pattern="*.jpg"):
    """
    Reads images files in a directory in sorted order
    """
    images = natsorted(glob(f"{frames_folder}/{pattern}"))
    if len(images) == 0:
        raise ValueError(
            f"Expected variable images to be non empty. {frames_folder} does not contain any frames"
        )

    frames = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

def load_frames_from_video(video_path, start_frame=None, end_frame=None):
    """
    Load the frames of the video
    Returns: numpy array of shape (T, H, W, C)
    """
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames

    # TODO: Update temporary fix
    if total_frames < start_frame:
        start_frame = 0
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(
        min(int(end_frame - start_frame), int(total_frames - start_frame))
    ):
        success, img = vidcap.read()
        if not success:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.asarray(frames)#, dtype=np.float32)

def list_all_files(dir, extensions=[]):
    """
    List all the files of the given extension type in the given path
    """
    if not extensions:
        files = glob(os.path.join(dir, '*'))
        return [f for f in files if os.path.isfile(f)]
    
    files = []
    for extension in extensions:
        files.extend(glob(os.path.join(dir, '*'+extension)))
    return files

def list_all_videos(dir):
    """
    List all video files in the given path
    """
    return list_all_files(dir, extensions=[".mp4", ".avi", ".MOV"])
