import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed

mp_holistic = mp.solutions.holistic

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

def process_body_landmarks(component, width, height, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x * width, p.y * height, p.z] for p in landmarks])
        conf = np.array([p.visibility for p in landmarks])
    return kps, conf


def process_other_landmarks(component, width, height, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x * width, p.y * height, p.z] for p in landmarks])
        conf = np.ones(n_points)
    return kps, conf


def get_holistic_keypoints(frames):
    holistic = mp_holistic.Holistic(static_image_mode=False)

    keypoints = []
    confs = []
    width, height, _ = frames[0].shape

    for frame in frames:
        results = holistic.process(frame)

        body_data, body_conf = process_body_landmarks(
            results.pose_landmarks, width, height, N_BODY_LANDMARKS
        )
        face_data, face_conf = process_other_landmarks(
            results.face_landmarks, width, height, N_FACE_LANDMARKS
        )
        lh_data, lh_conf = process_other_landmarks(
            results.left_hand_landmarks, width, height, N_HAND_LANDMARKS
        )
        rh_data, rh_conf = process_other_landmarks(
            results.right_hand_landmarks, width, height, N_HAND_LANDMARKS
        )

        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_conf, face_conf, lh_conf, rh_conf])

        keypoints.append(data)
        confs.append(conf)

    keypoints = np.stack(keypoints)
    confs = np.stack(confs)
    return keypoints, confs


def load_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.asarray(frames)


def gen_keypoints_for_video(video_path, save_path):
    frames = load_frames_from_video(video_path)
    kps, confs = get_holistic_keypoints(frames)
    confs = np.expand_dims(confs, axis=-1)
    data = np.concatenate([kps, confs], axis=-1)
    np.save(save_path, data)

def generate_pose(dataset, index, save_folder):
    imgs, label, video_id = dataset.read_data(index)
    keypoints, confs = get_holistic_keypoints(imgs.astype(np.uint8))
    confs = np.expand_dims(confs, axis=-1)
    data = np.concatenate([keypoints, confs], axis=-1)
    save_path = os.path.join(save_folder, video_id)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)

def dump_pose_dataset(dataset, save_folder, num_workers=multiprocessing.cpu_count()):
    os.makedirs(save_folder, exist_ok=True)
    Parallel(n_jobs=num_workers, backend="multiprocessing")(
        delayed(generate_pose)(dataset, i, save_folder)
        for i in tqdm(range(len(dataset)))
    )

if __name__ == "__main__":
    n_cores = multiprocessing.cpu_count()

    DIR = "AUTSL/train/"
    SAVE_DIR = "AUTSL/holistic_poses/"

    os.makedirs(SAVE_DIR, exist_ok=True)

    file_paths = []
    save_paths = []
    for file in os.listdir(DIR):
        if "color" in file:
            file_paths.append(os.path.join(DIR, file))
            save_paths.append(os.path.join(SAVE_DIR, file.replace(".mp4", "")))

    Parallel(n_jobs=n_cores, backend="multiprocessing")(
        delayed(gen_keypoints_for_video)(path, save_path)
        for path, save_path in tqdm(zip(file_paths, save_paths))
    )
