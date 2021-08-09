PKL_FOLDER = "ISH-News/corrected_pose_no_sub_one_min_splits"
output_hdf5 = PKL_FOLDER.rstrip("/") + ".h5"

# Get all pkl files
from glob import glob
from natsort import natsorted

pkl_files = natsorted(glob(PKL_FOLDER + "/**/*.pkl", recursive=True))
print(f"Found {len(pkl_files)} pkl files in {PKL_FOLDER}")

# Create output files

import os

os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
print(f"Writing output to {output_hdf5}")

import h5py

hf = h5py.File(output_hdf5, "w")
hf_keypoints = hf.create_group("keypoints")
hf_visibility = hf.create_group("visibility")

# Convert pkl to h5 datasets

import pickle
from tqdm import tqdm

dataset_names = set()
for pkl_file in tqdm(pkl_files, desc="Converting..."):
    # Read filename from path
    dataset_name = os.path.splitext(os.path.basename(pkl_file))[0]
    if dataset_name in dataset_names:
        raise RuntimeError(f"Filename {dataset_name} appears more than once")
    dataset_names.add(dataset_name)

    # Load data and save into h5
    pose_data = pickle.load(open(pkl_file, "rb"))
    hf_keypoints.create_dataset(dataset_name, data=pose_data["keypoints"])
    hf_visibility.create_dataset(dataset_name, data=pose_data["confidences"])

hf.close()
