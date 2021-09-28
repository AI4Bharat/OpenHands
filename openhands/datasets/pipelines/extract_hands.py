import torchvision
import numpy as np
from PIL import Image

def crop_hand(frame, keypoints, wrist_idx, elbow_idx, shoulder_dist, wrist_delta):
    missing_wrist = False
    cropped_frame = None

    wrist_points = keypoints[wrist_idx, 0:2]
    elbow_points = keypoints[elbow_idx, 0:2]
    hand_center = wrist_points + wrist_delta * (wrist_points - elbow_points)

    hand_center_x = hand_center[0]
    hand_center_y = hand_center[1]
    hand_xmin = max(0, int(hand_center_x - shoulder_dist // 2))
    hand_xmax = min(frame.shape[1], int(hand_center_x + shoulder_dist // 2))
    hand_ymin = max(0, int(hand_center_y - shoulder_dist // 2))
    hand_ymax = min(frame.shape[0], int(hand_center_y + shoulder_dist // 2))

    if (
        not np.any(wrist_points)
        or not np.any(elbow_points)
        or hand_ymax - hand_ymin <= 0
        or hand_xmax - hand_xmin <= 0
    ):
        cropped_frame = frame
        missing_wrist = True
        return cropped_frame, missing_wrist

    cropped_frame = frame[hand_ymin:hand_ymax, hand_xmin:hand_xmax, :]
    return cropped_frame, missing_wrist


def get_replace_idx(hand_crops_len, missing_wrists, clip_index):
    replace_idx = -1
    min_distance = np.inf
    for ci in range(hand_crops_len):
        if ci not in missing_wrists:
            dist = abs(ci - clip_index)
            if dist < min_distance:
                min_distance = dist
                replace_idx = ci
    return replace_idx

class ExtractHandCrops:
    """
    Extracts hands given the pose and image
    """

    SHOULDER_DIST_EPSILON = 1.2
    WRIST_DELTA = 0.15

    LEFT_ELBOW_IDX = 13
    RIGHT_ELBOW_IDX = 14
    LEFT_WRIST_IDX = 15
    RIGHT_WRIST_IDX = 16

    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12

    def __init__(self, resize_dims=(128, 128)):
        self.resize_dims = resize_dims

    def __call__(self, data):
        frames = data["frames"]
        keypoints_all = data["keypoints"]

        left_hand_crop, missing_wrist_left = crop_hand(
            frame_,
            keypoints,
            self.LEFT_WRIST_IDX,
            self.LEFT_ELBOW_IDX,
            shoulder_dist,
            self.WRIST_DELTA,
        )
        if missing_wrist_left:
            missing_wrists_left.append(len(left_hand_crops) + 1)

        left_hand_crops.append(
            torchvision.transforms.functional.resize(
                Image.fromarray(left_hand_crop), self.resize_dims
            )
        )

        right_hand_crop, missing_wrist_right = crop_hand(
            frame_,
            keypoints,
            self.RIGHT_WRIST_IDX,
            self.RIGHT_ELBOW_IDX,
            shoulder_dist,
            self.WRIST_DELTA,
        )
        if missing_wrist_right:
            missing_wrists_right.append(len(right_hand_crops) + 1)

        right_hand_crops.append(
            torchvision.transforms.functional.resize(
                Image.fromarray(right_hand_crop), self.resize_dims
            )
        )

        ## imputation
        for clip_index in range(len(left_hand_crops)):
            if clip_index in missing_wrists_left:
                replace_idx = get_replace_idx(
                    len(left_hand_crops), missing_wrists_left, clip_index
                )
                if replace_idx != -1:
                    left_hand_crops[clip_index] = left_hand_crops[replace_idx]

            if clip_index in missing_wrists_right:
                replace_idx = get_replace_idx(
                    len(right_hand_crops), missing_wrists_right, clip_index
                )
                if replace_idx != -1:
                    right_hand_crops[clip_index] = right_hand_crops[replace_idx]

        data["left_hand_crops"] = left_hand_crops
        data["right_hand_crops"] = right_hand_crops
        return data
