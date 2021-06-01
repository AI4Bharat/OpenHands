import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

mp_holistic = mp.solutions.holistic


def process_pose_landmarks(component, width, height, n_points):
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


def get_holistic_keypoints(frames, width, height):
    N_FACE_LANDMARKS = 468
    N_BODY_LANDMARKS = 33
    N_HAND_LANDMARKS = 21

    holistic = mp_holistic.Holistic(static_image_mode=False)

    keypoints = []
    confs = []

    for i, frame in enumerate(tqdm(frames)):
        results = holistic.process(frame)

        body_data, body_conf = process_pose_landmarks(
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


def gen_keypoints_for_video(video_path):
    frames = load_frames_from_video(video_path)
    width, height, _ = frames[0].shape
    kps, confs = get_holistic_keypoints(frames, width, height)

    # TODO: Save
    return {"kps": kps, "confs": confs}


n_cores = multiprocessing.cpu_count()
file_paths = []
DIR = "AUTSL/train/"
for file in os.listdir(DIR):
    if "color" in file:
        file_paths.append(os.path.join(DIR, file))

Parallel(n_jobs=n_cores, backend="multiprocessing")(
    delayed(gen_keypoints_for_video)(path) for path in tqdm(file_paths)
)
