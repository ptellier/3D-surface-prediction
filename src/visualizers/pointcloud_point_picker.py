import open3d as o3d
import open3d.visualization

from src.dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from nexera_packages.utilities.o3d_functions import pcd_img_to_o3d_pcd
from numpy import ndarray

from src.constants import MANUAL_DATASET_FOLDER_PATH
INDEX_TO_GRAB = 1


def run_pointcloud_point_picker(pcd_np_array: ndarray, rgb_img: ndarray = None) -> list[int]:
    pcd = pcd_img_to_o3d_pcd(pcd_np_array, rgb_img)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    return vis.get_picked_points()


def main():
    manual_dataset = ManuallyAnnotatedDataset(folder_path=MANUAL_DATASET_FOLDER_PATH)
    image, point_cloud_np_array, gt_mask_annotations = manual_dataset.get_clustering_data(INDEX_TO_GRAB)
    run_pointcloud_point_picker(point_cloud_np_array)


if __name__ == '__main__':
    main()
