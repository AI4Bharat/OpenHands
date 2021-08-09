import torchvision
import numpy as np
from PIL import Image


class ExtractHandCrops:
    """
    Extracts hands given the pose and image
    """

    SHOULDER_DIST_EPSILON = 1.2
    WRIST_DELTA = 0.15

    def __init__(self, resize_dims=(128, 128)):
        self.resize_dims = resize_dims
        self.LEFT_ELBOW_IDX = 13
        self.RIGHT_ELBOW_IDX = 14
        self.LEFT_WRIST_IDX = 15
        self.RIGHT_WRIST_IDX = 16

        self.LEFT_SHOULDER_IDX = 11
        self.RIGHT_SHOULDER_IDX = 12

    def __call__(self, data):
        frames = data["frames"]
        keypoints_all = data["keypoints"]

        left_hand_crops, right_hand_crops = [], []
        missing_wrists_left, missing_wrists_right = [], []
        for i in range(frames.shape[0]):
            frame_ = frames[i]
            keypoints = keypoints_all[i]
            shoulder_dist = (
                np.linalg.norm(
                    keypoints[self.LEFT_SHOULDER_IDX, 0:2]
                    - keypoints[self.RIGHT_SHOULDER_IDX, 0:2]
                )
                * self.SHOULDER_DIST_EPSILON
            )

            left_wrist_points = keypoints[self.LEFT_WRIST_IDX, 0:2]
            left_elbow_points = keypoints[self.LEFT_ELBOW_IDX, 0:2]
            left_hand_center = left_wrist_points + self.WRIST_DELTA * (
                left_wrist_points - left_elbow_points
            )
            left_hand_center_x = left_hand_center[0]
            left_hand_center_y = left_hand_center[1]
            left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
            left_hand_xmax = min(
                frame_.shape[1], int(left_hand_center_x + shoulder_dist // 2)
            )
            left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
            left_hand_ymax = min(
                frame_.shape[0], int(left_hand_center_y + shoulder_dist // 2)
            )

            if (
                not np.any(left_wrist_points)
                or not np.any(left_elbow_points)
                or left_hand_ymax - left_hand_ymin <= 0
                or left_hand_xmax - left_hand_xmin <= 0
            ):
                left_hand_crop = frame_
                missing_wrists_left.append(len(left_hand_crops) + 1)
            else:
                left_hand_crop = frame_[
                    left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :
                ]

            left_hand_crops.append(
                torchvision.transforms.functional.resize(
                    Image.fromarray(left_hand_crop), self.resize_dims
                )
            )

            right_wrist_points = keypoints[self.RIGHT_WRIST_IDX, 0:2]
            right_elbow_points = keypoints[self.RIGHT_ELBOW_IDX, 0:2]
            right_hand_center = right_wrist_points + self.WRIST_DELTA * (
                right_wrist_points - right_elbow_points
            )
            right_hand_center_x = right_hand_center[0]
            right_hand_center_y = right_hand_center[1]
            right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
            right_hand_xmax = min(
                frame_.shape[1], int(right_hand_center_x + shoulder_dist // 2)
            )
            right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
            right_hand_ymax = min(
                frame_.shape[0], int(right_hand_center_y + shoulder_dist // 2)
            )

            if (
                not np.any(right_wrist_points)
                or not np.any(right_elbow_points)
                or right_hand_ymax - right_hand_ymin <= 0
                or right_hand_xmax - right_hand_xmin <= 0
            ):
                right_hand_crop = frame_
                missing_wrists_left.append(len(right_hand_crops) + 1)
            else:
                right_hand_crop = frame_[
                    right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :
                ]

            right_hand_crops.append(
                torchvision.transforms.functional.resize(
                    Image.fromarray(right_hand_crop), self.resize_dims
                )
            )

        # Impute
        for clip_index in range(len(left_hand_crops)):
            # Left
            if clip_index in missing_wrists_left:
                replace_idx = -1
                min_distance = np.inf
                for ci in range(len(left_hand_crops)):
                    if ci not in missing_wrists_left:
                        dist = abs(ci - clip_index)
                        if dist < min_distance:
                            min_distance = dist
                            replace_idx = ci
                if replace_idx != -1:
                    left_hand_crop[clip_index] = left_hand_crop[replace_idx]

            # Right
            if clip_index in missing_wrists_right:
                replace_idx = -1
                min_distance = np.inf
                for ci in range(len(missing_wrists_right)):
                    if ci not in missing_wrists_right:
                        dist = abs(ci - clip_index)
                        if dist < min_distance:
                            min_distance = dist
                            replace_idx = ci
                if replace_idx != -1:
                    right_hand_crop[clip_index] = right_hand_crop[replace_idx]

        data["left_hand_crops"] = left_hand_crops
        data["right_hand_crops"] = right_hand_crops

        return data
