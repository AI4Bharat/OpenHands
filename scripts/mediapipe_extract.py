import cv2
import os
import time
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed
import math

mp_holistic = mp.solutions.holistic

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21

class Counter(object):
    # https://stackoverflow.com/a/47562583/
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value

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
    '''
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    Probably also OK to create only once? Read why: (hoping tracking is lost for first frame of new videos)
    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode
    '''

    keypoints = []
    confs = []
    width, height, _ = frames[0].shape
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)

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

    holistic.close()
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

def generate_pose(dataset, save_folder, worker_index, num_workers, counter):
    num_splits = math.ceil(len(dataset)/num_workers)
    end_index = min((worker_index+1)*num_splits, len(dataset))
    for index in range(worker_index*num_splits, end_index):
        imgs, label, video_id = dataset.read_data(index)
        keypoints, confs = get_holistic_keypoints(imgs.astype(np.uint8))
        confs = np.expand_dims(confs, axis=-1)
        data = np.concatenate([keypoints, confs], axis=-1)
        save_path = os.path.join(save_folder, video_id)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data)
        counter.increment()

def dump_pose_dataset(dataset, save_folder, num_workers=multiprocessing.cpu_count()):
    os.makedirs(save_folder, exist_ok=True)
    processes = []
    counter = Counter()
    for i in range(num_workers):
        p = multiprocessing.Process(target=generate_pose, args=(dataset, save_folder, i, num_workers, counter))
        p.start()
        processes.append(p)
    
    total_samples = len(dataset)
    with tqdm(total=total_samples) as pbar:  
        while counter.value < total_samples:
            pbar.update(counter.value - pbar.n)
            time.sleep(2)

    for i in range(num_workers):
        processes[i].join()
    print(f"Pose data successfully saved to: {save_folder}")

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
