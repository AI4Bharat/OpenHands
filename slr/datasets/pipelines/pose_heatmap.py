import numpy as np


def select_posepoints(poses, confs):
    body_kps = np.concatenate(
        [poses[:, :15, :], poses[:, 501:, :]], axis=1
    )  # Removing leg and face mesh points
    confs = np.concatenate([confs[:, :15], confs[:, 501:]], axis=1)
    # TODO: Add support for filter by frames, keypoints
    return body_kps, confs


def resize_keypoints(data, shape=(64, 64)):
    img_w, img_h = data["img_shape"]
    scale_factor = (shape[0] / img_w, shape[1] / img_h)
    data["img_shape"] = shape
    data["keypoint"] = data["keypoint"] * scale_factor
    return data

class GeneratePoseHeatMap:
    """Generate pseudo heatmaps based on joint coordinates and confidence.
    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".
    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(
        self,
        skeletons,
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=False,
        left_kp=None,
        right_kp=None,
    ):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" ' 'and "with_kp" should be set as True.'
        )
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons

    def generate_heatmap_for_single_frame_single_keypoint(
        self, img_h, img_w, centers, sigma, max_values
    ):
        """Generate pseudo heatmap for one keypoint in one frame.
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
            patch = patch * max_value
            heatmap[st_y:ed_y, st_x:ed_x] = np.maximum(
                heatmap[st_y:ed_y, st_x:ed_x], patch
            )

        return heatmap

    def generate_limb_heatmap_for_single_frame(
        self, img_h, img_w, starts, ends, sigma, start_values, end_values
    ):
        """Generate pseudo heatmap for one limb in one frame.
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(
            starts, ends, start_values, end_values
        ):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2

            # distance to end keypoints
            d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2

            # the distance between start and end keypoints.
            d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2

            if d2_ab < 1:
                full_map = self.generate_heatmap_for_single_frame_single_keypoint(
                    img_h, img_w, [start], sigma, [start_value]
                )
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line
            )

            patch = np.exp(-d2_seg / 2.0 / sigma ** 2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch
            )

        return heatmap

    def generate_heatmap_for_single_frame(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                heatmap = self.generate_heatmap_for_single_frame_single_keypoint(
                    img_h, img_w, kps[:, i], sigma, max_values[:, i]
                )
                heatmaps.append(heatmap)

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_limb_heatmap_for_single_frame(
                    img_h, img_w, starts, ends, sigma, start_values, end_values
                )
                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def generate_heatmap(self, results):
        """Generate pseudo heatmaps for all frames.
        Args:
            results (dict): The dictionary that contains all info of a sample.
        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results["keypoint"]
        kp_shape = all_kps.shape

        if "keypoint_score" in results:
            all_kpscores = results["keypoint_score"]
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results["img_shape"]
        num_frame = kp_shape[1]

        imgs = []
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            hmap = self.generate_heatmap_for_single_frame(
                img_h, img_w, kps, sigma, max_values
            )
            imgs.append(hmap)

        return imgs

    def __call__(self, results):
        results["imgs"] = np.stack(self.generate_heatmap(results))
        return results


if __name__ == "__main__":
    path = "datasets/AUTSL/train_poses/signer0_sample1000_color.npy"
    poses = np.load(path)
    poses, confs = poses[:, :, :2], poses[:, :, 3]
    pose_selected = select_posepoints(poses, confs)
    sel_kps, sel_cfs = pose_selected
    data = {}
    data["keypoint"] = np.expand_dims(sel_kps, 0)
    data["keypoint_score"] = np.expand_dims(sel_cfs, 0)
    data["img_shape"] = (512, 512)
    resizer = resize_keypoints(data)
    skeleton_conn = (
        (
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 9),
            (0, 10),
            (10, 12),
            (9, 11),
            (12, 14),
            (14, 36),
            (36, 37),
            (36, 41),
            (36, 45),
            (36, 49),
            (36, 53),
            (37, 38),
            (38, 39),
            (39, 40),
            (41, 42),
            (42, 43),
            (43, 44),
            (45, 46),
            (46, 47),
            (47, 48),
            (49, 50),
            (50, 51),
            (51, 52),
            (53, 54),
            (54, 55),
            (55, 56),
            (11, 13),
            (13, 15),
            (15, 16),
            (15, 20),
            (15, 24),
            (15, 28),
            (15, 32),
            (16, 17),
            (17, 18),
            (18, 19),
            (20, 21),
            (21, 22),
            (22, 23),
            (24, 25),
            (25, 26),
            (26, 27),
            (28, 29),
            (29, 30),
            (30, 31),
            (32, 33),
            (33, 34),
            (34, 35),
        ),
    )
    generator = GeneratePoseHeatMap(skeletons=skeleton_conn, with_limb=True)
    out = generator(data)
