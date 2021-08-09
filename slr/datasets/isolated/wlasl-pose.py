import torch
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import numpy as np


class KeypointsDataset(torch.utils.data.Dataset):
    def __init__(
        self, index_file_path, split, pose_root, max_frames=300, test_index_file=None
    ):

        self.data = []
        self.test_index_file = test_index_file
        self.pose_root = pose_root
        self.read_index_file(index_file_path, split)
        self.hand_kps_to_select = list(range(0, 21, 2))
        self.body_kps_to_select = [1, 2, 3, 5, 6]
        self.max_frames = max_frames
        self.framename = "image_{}_keypoints.json"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]
        frames_to_sample = list(range(frame_start, frame_end + 1))

        all_skeletons = []
        for i in frames_to_sample:
            pose_path = os.path.join(
                self.pose_root, video_id, self.framename.format(str(i).zfill(5))
            )
            pose_keypoints = self.read_pose_file(pose_path)
            if pose_keypoints is None:
                continue
            x, y, c = pose_keypoints[:, 0], pose_keypoints[:, 1], pose_keypoints[:, 2]
            selected_skeleton = np.array([x, y, c])
            all_skeletons.append(selected_skeleton)

        all_skeletons = np.stack(all_skeletons, axis=0)
        all_skeletons = np.pad(
            all_skeletons,
            pad_width=((0, self.max_frames - all_skeletons.shape[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        all_skeletons = np.transpose(all_skeletons, [1, 0, 2])

        # Normalize
        all_skeletons[0, ...] /= 256
        all_skeletons[1, ...] /= 256

        skeletons = torch.FloatTensor(all_skeletons)
        return skeletons, torch.tensor(gloss_cat)

    def read_pose_file(self, filepath):
        try:
            content = json.load(open(filepath))["people"][0]
        except IndexError:
            return None

        body_kps = np.array(content["pose_keypoints_2d"])
        left_hand_kps = np.array(content["hand_left_keypoints_2d"])
        right_hand_kps = np.array(content["hand_right_keypoints_2d"])

        body_kps_selected = []
        for i in self.body_kps_to_select:
            body_kps_selected.append(body_kps[i * 3 : (i * 3 + 3)])

        left_hand_kps_selected = []
        right_hand_kps_selected = []
        for i in self.hand_kps_to_select:
            left_hand_kps_selected.append(left_hand_kps[i * 3 : (i * 3 + 3)])
            right_hand_kps_selected.append(right_hand_kps[i * 3 : (i * 3 + 3)])

        body_kps_selected = np.array(body_kps_selected)
        left_hand_kps_selected = np.array(left_hand_kps_selected)
        right_hand_kps_selected = np.array(right_hand_kps_selected)

        all_kps_selected = np.concatenate(
            [body_kps_selected, right_hand_kps_selected, left_hand_kps_selected]
        )
        return all_kps_selected

    def read_index_file(self, index_file_path, split):
        with open(index_file_path, "r") as f:
            content = json.load(f)

        glosses = sorted([gloss_entry["gloss"] for gloss_entry in content])
        label_encoder = LabelEncoder()
        label_encoder.fit(glosses)

        if self.test_index_file is not None:
            with open(self.test_index_file, "r") as f:
                content = json.load(f)

        for gloss_entry in content:
            gloss, instances = gloss_entry["gloss"], gloss_entry["instances"]
            gloss_cat = label_encoder.transform([gloss])[0]

            for instance in instances:
                if instance["split"] not in split:
                    continue

                frame_end = instance["frame_end"]
                frame_start = instance["frame_start"]
                video_id = instance["video_id"]

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.data.append(instance_entry)
