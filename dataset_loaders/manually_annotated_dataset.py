import re

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from pycocotools.coco import COCO

import os
from torchvision.io import read_image

DOWNLOAD_URL = 'https://drive.google.com/drive/folders/1-tls6KQyJnQQYwXv9Wmaoey3Bvvx3I3_?usp=drive_link'

IMAGE_DIR = 'images'
PCD_DIR = 'point_clouds'
JSON_PATH = 'annotations/surface_annotations.json'
LAST_NUMBER_PATTERN = re.compile('\d+')

DATASET_FOLDER_PATH = './datasets/manual_dataset/'


class ManuallyAnnotatedDataset(Dataset):
    def __init__(self, folder_path: str):
        if not os.path.isdir(DATASET_FOLDER_PATH):
            raise FileNotFoundError(
                f'Please manually download and move the manual dataset into {DATASET_FOLDER_PATH} from {DOWNLOAD_URL}')
        self.folder_path = folder_path
        self.coco = COCO(os.path.join(self.folder_path, JSON_PATH))

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx) -> tuple[Tensor, ndarray, dict]:
        mask_annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[idx]))
        image_file_name = self.coco.loadImgs(ids=[idx])[0]['file_name']
        file_number = find_last_int(image_file_name)
        img_path = os.path.join(self.folder_path, IMAGE_DIR, image_file_name)
        image = read_image(img_path)
        point_cloud_file_name = f'stereo_point_cloud_{str(file_number)}.npy'
        point_cloud_np_array = np.load(os.path.join(self.folder_path, PCD_DIR, point_cloud_file_name)).reshape((1024, 1280, 3))
        return image, point_cloud_np_array, mask_annotations

    def get_file_number_from_name(self):
        raise NotImplementedError()


def find_last_int(string: str) -> int:
    """Produces the last integer appearing in a string. Raises a ValueError if no integer is found."""
    re_matches = re.findall(LAST_NUMBER_PATTERN, string)
    if not re_matches:
        raise ValueError(f'No int found in string "{string}"')
    last_number = re_matches[-1]
    return int(last_number)


if __name__ == '__main__':
    # NOTE**: index starts at 1
    index = 1
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    image_1, point_cloud_np_array_1, mask_annotations_1 = manual_dataset[index]

    print('image_1: ')
    print(image_1)
    print()

    print('point_cloud_np_array_1: ')
    print(point_cloud_np_array_1)
    print()

    print('mask_annotations_1: ')
    print(mask_annotations_1)
    print()

