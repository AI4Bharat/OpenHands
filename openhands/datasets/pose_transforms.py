import torch
import random
import numpy as np


class Compose:
    """
    Compose a list of pose transforms
    
    Args:
        transforms (list): List of transforms to be applied.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: dict):
        """Applies the given list of transforms

        Args:
            x (dict): input data

        Returns:
            dict: data after the transforms
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class ScaleToVideoDimensions:
    """
    Scale the pose keypoints to the given width and height values.
    
    Args:
        width (int): Width of the frames
        height (int): Height of the frames 
        
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, data: dict):
        """
        Applies the scaling to the input keypoints data.

        Args:
            data (dict): given data

        Returns:
            dict: data after scaling
        """
        
        kps = data["frames"]

        kps[0, ...] *= self.width
        kps[1, ...] *= self.height

        data["frames"] = kps
        return data


class PoseRandomShift:
    """
    Randomly distribute the zero padding at the end of a video
    to initial and final positions
    """
    def __call__(self, data:dict):
        """
        Applies the random shift to the given input data

        Args:
            data (dict): input data

        Returns:
            dict: data after applying random shift
        """
        x = data["frames"]
        C, T, V = x.shape
        data_shifted = torch.zeros_like(x)
        valid_frame = ((x != 0).sum(dim=2).sum(dim=0) > 0).long()
        begin = valid_frame.argmax()
        end = len(valid_frame) - torch.flip(valid_frame, dims=[0]).argmax()

        size = end - begin
        bias = random.randint(0, T - size)
        data_shifted[:, bias : bias + size, :] = x[:, begin:end, :]

        data["frames"] = data_shifted
        return data


class PoseSelect:
    """
    Select the given index keypoints from all keypoints. 
    
    Args:
        preset (str | None, optional): can be used to specify existing presets - `mediapipe_holistic_minimal_27` or `mediapipe_holistic_top_body_59`
        If None, then the `pose_indexes` argument indexes will be used to select. Default: ``None``
        
        pose_indexes: List of indexes to select.
    """
    # fmt: off
    KEYPOINT_PRESETS = {
        "mediapipe_holistic_minimal_27": [ 0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74],
        "mediapipe_holistic_top_body_59": [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
    }
    # fmt: on
    
    def __init__(self, preset=None, pose_indexes: list=[]):
        if preset:
            self.pose_indexes = self.KEYPOINT_PRESETS[preset]
        elif pose_indexes:
            self.pose_indexes = pose_indexes
        else:
            raise ValueError("Either pass `pose_indexes` to select or `preset` name")

    def __call__(self, data:dict):
        """
        Apply selection of keypoints based on the given indexes.

        Args:
            data (dict): input data

        Returns:
            dict : transformed data
        """
        x = data["frames"]
        x = x[:, :, self.pose_indexes]
        data["frames"] = x
        return data


class PrependLangCodeOHE:
    """
    Prepend a one-hot encoded vector based on the language of the input video.
    Ideally, it should be used finally after all other normalizations/augmentations.

    Args:
        lang_codes: List of sign language codes.
    """

    def __init__(self, lang_codes: list):
        self.lang_codes = lang_codes
        self.lang_code_to_index = {lang_code: i for i, lang_code in enumerate(lang_codes)}
    
    def __call__(self, data: dict):
        """
        Preprend lang_code OHE dynamically.

        Args:
            data (dict): input data

        Returns:
            dict : transformed data
        """
        lang_index = self.lang_code_to_index[data["lang_code"]]
        x = data["frames"]
        x = x.permute(1, 2, 0) #CTV->TVC

        ohe = torch.zeros(1, x.shape[1], x.shape[2])
        ohe[0][lang_index] = 1

        x = torch.cat([ohe, x])
        data["frames"] = x.permute(2, 0, 1) #TVC->CTV
        return data

# Adopted from: https://github.com/AmitMY/pose-format/
class ShearTransform:
    """
    Applies `2D shear <https://en.wikipedia.org/wiki/Shear_matrix>`_ transformation
    
    Args:
        shear_std (float): std to use for shear transformation. Default: 0.2
    """
    def __init__(self, shear_std: float=0.2):
        self.shear_std = shear_std

    def __call__(self, data:dict):
        """
        Applies shear transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after shear transformation
        """
        
        x = data["frames"]
        assert x.shape[0] == 2, "Only 2 channels inputs supported for ShearTransform"
        x = x.permute(1, 2, 0) #CTV->TVC
        shear_matrix = torch.eye(2)
        shear_matrix[0][1] = torch.tensor(
            np.random.normal(loc=0, scale=self.shear_std, size=1)[0]
        )
        res = torch.matmul(x, shear_matrix)
        data["frames"] = res.permute(2, 0, 1) #TVC->CTV
        return data


class RotatationTransform:
    """
    Applies `2D rotation <https://en.wikipedia.org/wiki/Rotation_matrix>`_ transformation.
    
    Args:
        rotation_std (float): std to use for rotation transformation. Default: 0.2
    """
    def __init__(self, rotation_std: float=0.2):
        self.rotation_std = rotation_std

    def __call__(self, data):
        """
        Applies rotation transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after rotation transformation
        """
        x = data["frames"]
        assert x.shape[0] == 2, "Only 2 channels inputs supported for RotationTransform"
        x = x.permute(1, 2, 0) #CTV->TVC
        rotation_angle = torch.tensor(
            np.random.normal(loc=0, scale=self.rotation_std, size=1)[0]
        )
        rotation_cos = torch.cos(rotation_angle)
        rotation_sin = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor(
            [[rotation_cos, -rotation_sin], [rotation_sin, rotation_cos]],
            dtype=torch.float32,
        )
        res = torch.matmul(x, rotation_matrix)
        data["frames"] = res.permute(2, 0, 1) #TVC->CTV
        return data


class ScaleTransform:
    """
    Applies `Scaling <https://en.wikipedia.org/wiki/Scaling_(geometry)>`_ transformation

    Args:
        scale_std (float): std to use for Scaling transformation. Default: 0.2
    """
    def __init__(self, scale_std=0.2):
        self.scale_std = scale_std

    def __call__(self, data):
        """
        Applies scaling transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after scaling transformation
        """
        x = data["frames"]
        assert x.shape[0] == 2, "Only 2 channels inputs supported for ScaleTransform"

        x = x.permute(1, 2, 0) #CTV->TVC
        scale_matrix = torch.eye(2)
        scale_matrix[1][1] = torch.tensor(
            np.random.normal(loc=0, scale=self.scale_std, size=1)[0]
        )
        res = torch.matmul(x, scale_matrix)
        data["frames"] = res.permute(2, 0, 1) #TVC->CTV
        return data


class CenterAndScaleNormalize:
    """
    Centers and scales the keypoints based on the referent points given.

    Args:
        reference_points_preset (str | None, optional): can be used to specify existing presets - `mediapipe_holistic_minimal_27` or `mediapipe_holistic_top_body_59`
        reference_point_indexes (list): shape(p1, p2); point indexes to use if preset is not given then
        scale_factor (int): scaling factor. Default: 1
        frame_level (bool): Whether to center and normalize at frame level or clip level. Default: ``False``
    """
    REFERENCE_PRESETS = {
        "shoulder_mediapipe_holistic_minimal_27": [3, 4],
        "shoulder_mediapipe_holistic_top_body_59": [11, 12],
    }

    def __init__(
        self,
        reference_points_preset=None,
        reference_point_indexes=[],
        scale_factor=1,
        frame_level=False,
    ):

        if reference_points_preset:
            self.reference_point_indexes = CenterAndScaleNormalize.REFERENCE_PRESETS[
                reference_points_preset
            ]
        elif reference_point_indexes:
            self.reference_point_indexes = reference_point_indexes
        else:
            raise ValueError(
                "Mention the joint with respect to which the scaling & centering must be done"
            )
        self.scale_factor = scale_factor
        self.frame_level = frame_level

    def __call__(self, data):
        """
        Applies centering and scaling transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after centering normalization
        """
        x = data["frames"]
        C, T, V = x.shape
        x = x.permute(1, 2, 0) #CTV->TVC

        if self.frame_level:
            for ind in range(x.shape[0]):
                center, scale = self.calc_center_and_scale_for_one_skeleton(x[ind])
                x[ind] -= center
                x[ind] *= scale
        else:
            center, scale = self.calc_center_and_scale(x)
            x = x - center
            x = x * scale

        data["frames"] = x.permute(2, 0, 1) #TVC->CTV
        return data

    def calc_center_and_scale_for_one_skeleton(self, x):
        """
        Calculates the center and scale values for one skeleton.

        Args:
            x (torch.Tensor): Spatial keypoints at a timestep

        Returns:
            [float, float]: center and scale value to normalize for the skeleton
        """
        ind1, ind2 = self.reference_point_indexes
        point1, point2 = x[ind1], x[ind2]
        center = (point1 + point2) / 2
        dist = torch.sqrt(((point1 - point2) ** 2).sum(-1))
        scale = self.scale_factor / dist
        if torch.isinf(scale).any():
            return 0, 1  # Do not normalize
        return center, scale

    def calc_center_and_scale(self, x):
        """
        Calculates the center and scale value based on the sequence of skeletons.

        Args:
            x (torch.Tensor): all keypoints for the video clip.

        Returns:
            [float, float]: center and scale value to normalize
        """
        transposed_x = x.permute(1, 0, 2) # TVC -> VTC
        ind1, ind2 = self.reference_point_indexes
        points1 = transposed_x[ind1]
        points2 = transposed_x[ind2]

        points1 = points1.reshape(-1, points1.shape[-1])
        points2 = points2.reshape(-1, points2.shape[-1])

        center = torch.mean((points1 + points2) / 2, dim=0)
        mean_dist = torch.mean(torch.sqrt(((points1 - points2) ** 2).sum(-1)))
        scale = self.scale_factor / mean_dist
        if torch.isinf(scale).any():
            return 0, 1  # Do not normalize

        return center, scale


class RandomMove:
    """
    Moves all the keypoints randomly in a random direction.
    """
    def __init__(self, move_range=(-2.5, 2.5), move_step=0.5):
        self.move_range = torch.arange(*move_range, move_step)

    def __call__(self, data):
        x = data["frames"]  # C, T, V
        num_frames = x.shape[1]

        t_x = np.random.choice(self.move_range, 2)
        t_y = np.random.choice(self.move_range, 2)

        t_x = torch.linspace(t_x[0], t_x[1], num_frames)
        t_y = torch.linspace(t_y[0], t_y[1], num_frames)

        for i_frame in range(num_frames):
            x[0] += t_x[i_frame]
            x[1] += t_y[i_frame]

        data["frames"] = x
        return data


class PoseTemporalSubsample:
    """
    Randomly subsamples num_frames indices from the temporal dimension of the sequence of keypoints.
    If the num_frames if larger than the length of the sequence, then the remaining frames will be padded with zeros.
        
    Args:
        num_frames (int): Number of frames to subsample.
        temporal_dim(int): dimension of temporal to perform temporal subsample.
    """
    def __init__(self, num_frames, temporal_dim=1):
        self.num_frames = num_frames
        self.temporal_dim = temporal_dim

    def __call__(self, data):
        """
        performs random subsampling based on the number of frames needed.

        Args:
            data (dict): input data

        Returns:
            dict: data after subsampling
        """
        x = data["frames"]
        C, T, V = x.shape

        t = x.shape[self.temporal_dim]
        if t >= self.num_frames:
            start_index = random.randint(0, t - self.num_frames)
            indices = torch.arange(start_index, start_index + self.num_frames)
            data["frames"] = torch.index_select(x, self.temporal_dim, indices)

        else:
            # Padding
            pad_len = self.num_frames - t
            pad_tensor = torch.zeros(C, pad_len, V)
            data["frames"] = torch.cat((x, pad_tensor), dim=1)

        return data


class PoseUniformSubsampling:
    """
    Uniformly subsamples num_frames indices from the temporal dimension of the sequence of keypoints.
    If the num_frames is larger than the length of the sequence, then the remaining frames will be padded with zeros.
        
    Args:
        num_frames (int): Number of frames to subsample.
        randomize_start_index (int): While performing interleaved subsampling, select `start_index` from randint(0, step_size)
        temporal_dim (int): dimension of temporal to perform temporal subsample.
    """
    def __init__(self, num_frames, randomize_start_index=False, temporal_dim=1):
        self.num_frames = num_frames
        self.randomize_start_index = randomize_start_index
        self.temporal_dim = temporal_dim

    def __call__(self, data):
        """
        Performs uniform subsampling based on the number of frames needed.

        Args:
            data (dict): input data

        Returns:
            dict: data after subsampling
        """
        x = data["frames"]
        C, T, V = x.shape
        t = x.shape[self.temporal_dim]

        # Randomize start_index
        step_size = t / self.num_frames
        if self.randomize_start_index and step_size > 1:
            start_index = random.randint(0, int(step_size))
        else:
            start_index = 0

        indices = torch.linspace(start_index, t - 1, self.num_frames)
        indices = torch.clamp(indices, start_index, t - 1).long()
        x = torch.index_select(x, self.temporal_dim, indices)
        if x.shape[self.temporal_dim] < self.num_frames:
            # Pad
            pad_len = self.num_frames - x.shape[self.temporal_dim]
            pad_tensor = torch.zeros(C, pad_len, V)
            data["frames"] = torch.cat((x, pad_tensor), dim=self.temporal_dim)
        else:
            data["frames"] = x
        return data


class TemporalSample:
    """
    Randomly choose Uniform and Temporal subsample
        - If subsample_mode==2, randomly sub-sampling or uniform-sampling is done
        - If subsample_mode==0, only uniform-sampling (for test sets)
        - If subsample_mode==1, only sub-sampling (to reproduce results of some papers that use only subsampling)
        
    Args:
        num_frames (int): Number of frames to subsample.
        subsample_mode (int): Mode to choose.
    """

    def __init__(self, num_frames, subsample_mode=2):
        self.subsample_mode = subsample_mode
        self.num_frames = num_frames

        self.uniform_sampler = PoseUniformSubsampling(num_frames, randomize_start_index=subsample_mode==2)
        self.random_sampler = PoseTemporalSubsample(num_frames)

    def __call__(self, data):
        """
        performs subsampling based on the mode.

        Args:
            data (dict): input data

        Returns:
            dict: data after subsampling
        """
        if self.subsample_mode == 0:
            return self.uniform_sampler(data)
        elif self.subsample_mode == 1:
            return self.random_sampler(data)
        elif self.subsample_mode == 2:
            if random.random() > 0.5:
                return self.uniform_sampler(data)
            else:
                return self.random_sampler(data)


class FrameSkipping:
    """
    Skips the frame based on the jump range specified.
    
    Args:
        skip_range(int): The skip range.
    """
    def __init__(self, skip_range=1):
        self.skip_range = skip_range
        self.temporal_dim = 1

    def __call__(self, data):
        """
        performs frame skipping.

        Args:
            data (dict): input data

        Returns:
            dict: data after skipping frames based on the jump range.
        """
        x = data["frames"]
        t = x.shape[self.temporal_dim]
        indices = torch.arange(0, t, self.skip_range)
        data["frames"] = torch.index_select(x, self.temporal_dim, indices)
        return data


class AddClsToken:
    # Warning: Do not add any transforms after this
    def __call__(self, data):
        x = data["frames"]
        C, T, V = x.shape
        x = torch.cat([torch.ones(C, 1, V), x], dim=1)
        data["frames"] = x
        return data
