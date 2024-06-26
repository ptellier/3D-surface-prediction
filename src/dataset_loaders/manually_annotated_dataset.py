import re
import numpy as np
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from torchvision.io import read_image

from src.constants import MANUAL_DATASET_FOLDER_PATH

DOWNLOAD_URL = 'https://drive.google.com/drive/folders/1-tls6KQyJnQQYwXv9Wmaoey3Bvvx3I3_?usp=drive_link'

IMAGE_DIR = 'images'
PCD_DIR = 'point_clouds'
JSON_PATH = 'annotations/surface_annotations.json'
GT_COCO_DIR = 'annotations/surface_annotations.json'
CLUSTER_DISTANCES_DIR = 'cluster_distances'
NUMBER_OF_NEIGHBOURS_DIR = 'number_of_neighbours'
GT_LABELS_DIR = 'labels'
LAST_NUMBER_PATTERN = re.compile('\d+')


class ManuallyAnnotatedDataset(Dataset):
    def __init__(self, folder_path: str):
        if not os.path.isdir(MANUAL_DATASET_FOLDER_PATH):
            raise FileNotFoundError(
                f'Please manually download and move the manual dataset into {MANUAL_DATASET_FOLDER_PATH} from {DOWNLOAD_URL}')
        self.folder_path = folder_path
        self.coco = COCO(os.path.join(self.folder_path, JSON_PATH))

    def __len__(self):
        return len(self.coco.imgs)

    def get_clustering_data(self, index: int) -> tuple[Tensor, ndarray, dict]:
        gt_mask_annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[index]))
        image_file_name = self.coco.loadImgs(ids=[index])[0]['file_name']
        file_number = find_last_int(image_file_name)
        img_path = os.path.join(self.folder_path, IMAGE_DIR, image_file_name)
        image = read_image(img_path)

        point_cloud_np_array = self.load_npy_file(f'stereo_point_cloud_{str(file_number)}', PCD_DIR, shape=(1024, 1280, 3))

        return image, point_cloud_np_array, gt_mask_annotations

    def get_training_data(self, index: int) -> tuple[ndarray, ndarray, ndarray]:
        """
        Returns the training data to train an ML model for a particular index.

        Args:
            index: index for the training data to produce

        Returns:
            cluster_distances_np_array -- The average intra-cluster distances in point-wise clusterings of
                                          point neighbourhoods in the pointcloud.
            num_neighbours_np_array -- The number of neighbours for each point when doing point-wise clusterings.
            labels_np_array -- The ground-truth 3D surface class for each point in the pointcloud.
        """
        image_file_name = self.coco.loadImgs(ids=[index])[0]['file_name']
        file_number = find_last_int(image_file_name)
        cluster_distances_np_array = self.load_npy_file(f'cluster_similarity_{index}', CLUSTER_DISTANCES_DIR)
        num_neighbours_np_array = self.load_npy_file(f'neighbours_per_point_{index}', NUMBER_OF_NEIGHBOURS_DIR)
        labels_np_array = self.load_npy_file(f'gt_labels_downsampled_{index}', GT_LABELS_DIR)
        return cluster_distances_np_array, num_neighbours_np_array, labels_np_array

    def load_npy_file(self, file_name: str, folder_name: str, shape=None) -> ndarray:
        file_name = file_name + '.npy'
        np_array = np.load(os.path.join(self.folder_path, folder_name, file_name))
        if shape is not None:
            np_array = np_array.reshape(shape)
        return np_array


def find_last_int(string: str) -> int:
    """Produces the last integer appearing in a string. Raises a ValueError if no integer is found."""
    re_matches = re.findall(LAST_NUMBER_PATTERN, string)
    if not re_matches:
        raise ValueError(f'No int found in string "{string}"')
    last_number = re_matches[-1]
    return int(last_number)


def main():
    # NOTE**: index starts at 1
    index = 1
    manual_dataset = ManuallyAnnotatedDataset(folder_path=MANUAL_DATASET_FOLDER_PATH)
    image_1, point_cloud_np_array_1, mask_annotations_1, cluster_dist_1, num_neigh = manual_dataset[index]

    print('image_1: ')
    print(image_1)
    print()

    print('point_cloud_np_array_1: ')
    print(point_cloud_np_array_1)
    print()

    print('mask_annotations_1: ')
    print(mask_annotations_1)
    print()

    print('cluster_distances_1: ')
    print(cluster_dist_1)
    print()

    print('num_neighbours_1: ')
    print(num_neigh)
    print()


if __name__ == '__main__':
    main()
