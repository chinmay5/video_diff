import glob
import os
import pickle
import random

import nibabel as nib
import numpy as np
import torch.utils.data
from monai.transforms import Resize, NormalizeIntensity, Compose

from environment_setup import PROJECT_ROOT_DIR


def verify_splits(train_files, val_files, test_files):
    assert (len(set(train_files).intersection(set(val_files))) == 0 and
            len(set(val_files).intersection(set(test_files)))) == 0 and len(
        set(test_files).intersection(set(train_files))) == 0, "The splits have common element. Please correct."


def save_as_text_file(train_files, filename):
    with open(filename, 'w') as f:
        for img in train_files:
            f.write(img + "\n")


def save_pickle_files(split_name_files_dict):
    for filename, filelist in split_name_files_dict.items():
        with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', filename), 'wb') as file:
            pickle.dump(filelist, file)


def load_list_from_pickle_file(filename):
    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', filename), 'rb') as file:
        return pickle.load(file)


class TopCowMraDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform=None):
        super(TopCowMraDataset, self).__init__()
        self.split = split
        self.data_dir = data_dir
        self.bootstrap()
        self.mra_files = load_list_from_pickle_file(filename=f"{self.split}.pkl")
        self.transform = transform
        self.resize_transform = Compose([Resize((96, 96, 96)), NormalizeIntensity()])

    def __len__(self):
        # TODO: Parameterize this
        return min(8, len(self.mra_files))

    def __getitem__(self, idx):
        nii_img = nib.load(self.mra_files[idx]).get_fdata()[
            None, ...]  # Adding the channel dimension to work with monai
        # Let us put in the more common image format with c, l, h, w
        nii_img = np.transpose(nii_img, (0, 3, 1, 2))  # N, C, H, W
        nii_img = self.resize_transform(nii_img)
        if self.transform is not None:
            nii_img = self.transform(nii_img)
        # transpose the axes to put number of frames at the end.
        # nii_img = np.transpose(nii_img, (1, 0, 2, 3))  # N, C, H, W
        return nii_img

    def bootstrap(self):
        if os.path.exists(f'{self.split}.pkl'):
            print("Using existing splits")
        else:
            print("Generating new splits")
            # We have to create the new splits.
            all_files = glob.glob(f"{self.data_dir}/*.nii.gz")
            random.shuffle(all_files)
            proportions = (np.asarray([0.8, 0.9, 1.0]) * len(all_files)).astype(int)
            train_files = all_files[:proportions[0]]
            val_files = all_files[proportions[0]: proportions[1]]
            test_files = all_files[proportions[1]:]
            verify_splits(train_files, val_files, test_files)
            # Now we can save the different filenames.
            # We save it as txt file for ease in debugging
            save_as_text_file(train_files, f"train.txt")
            # save_as_text_file(val_files, "val.txt")
            # save_as_text_file(test_files, "test.txt")
            # However, the main work happens with the pickle files
            save_pickle_files(
                {"train.pkl": train_files,
                 "val.pkl": val_files,
                 "test.pkl": test_files}
            )


if __name__ == '__main__':
    data_dir = '/mnt/dog/bran/CoW_MRI/imagesTr_resampled'
    dataset = TopCowMraDataset(data_dir, transform=None, split="train")
    print(dataset[4].shape)
    print(dataset[4].max())
    print(dataset[4].min())
    print(dataset[4].mean())
