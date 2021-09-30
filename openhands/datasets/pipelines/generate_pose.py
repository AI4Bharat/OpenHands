import cv2
import os, sys, gc
import time
import numpy as np
from tqdm.auto import tqdm
from natsort import natsorted
from glob import glob
import math
import pickle

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21

class MediaPipePoseGenerator:
    def __init__(self, model_complexity=2):
        import mediapipe as mp
        self.holistic=mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=model_complexity)
    
    def __del__(self):
        self.holistic.close()
        # del self.holistic
        # gc.collect()

    def process_body_landmarks(self, component, n_points):
        kps = np.zeros((n_points, 3))
        conf = np.zeros(n_points)
        if component is not None:
            landmarks = component.landmark
            kps = np.array([[p.x, p.y, p.z] for p in landmarks])
            conf = np.array([p.visibility for p in landmarks])
        return kps, conf
    
    def process_other_landmarks(self, component, n_points):
        kps = np.zeros((n_points, 3))
        conf = np.zeros(n_points)
        if component is not None:
            landmarks = component.landmark
            kps = np.array([[p.x, p.y, p.z] for p in landmarks])
            conf = np.ones(n_points)
        return kps, conf
    
    def get_holistic_keypoints(self, frames):
        keypoints = []
        confs = []

        for frame in frames:
            results = self.holistic.process(frame)

            body_data, body_conf = self.process_body_landmarks(
                results.pose_landmarks, N_BODY_LANDMARKS
            )
            face_data, face_conf = self.process_other_landmarks(
                results.face_landmarks, N_FACE_LANDMARKS
            )
            lh_data, lh_conf = self.process_other_landmarks(
                results.left_hand_landmarks, N_HAND_LANDMARKS
            )
            rh_data, rh_conf = self.process_other_landmarks(
                results.right_hand_landmarks, N_HAND_LANDMARKS
            )

            data = np.concatenate([body_data, face_data, lh_data, rh_data])
            conf = np.concatenate([body_conf, face_conf, lh_conf, rh_conf])

            keypoints.append(data)
            confs.append(conf)

        self.holistic.reset()

        keypoints = np.stack(keypoints)
        confs = np.stack(confs)
        return keypoints, confs

    def generate_keypoints_for_frames(self, frames, save_as):
        pose_kps, pose_confs = self.get_holistic_keypoints(frames)

        body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)
        confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)
        d = {"keypoints": body_kps, "confidences": confs}

        with open(save_as, "wb") as f:
            pickle.dump(d, f, protocol=4)

