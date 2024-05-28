import open3d as o3d
import open3d.visualization
from open3d.geometry import PointCloud

from src.dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from nexera_packages.utilities.o3d_functions import pcd_img_to_o3d_pcd

from src.constants import MANUAL_DATASET_FOLDER_PATH
INDEX_TO_GRAB = 1


def run_pointcloud_point_picker(pcd: PointCloud) -> list[int]:
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    return vis.get_picked_points()


def main():
    manual_dataset = ManuallyAnnotatedDataset(folder_path=MANUAL_DATASET_FOLDER_PATH)
    image, point_cloud_np_array, gt_mask_annotations = manual_dataset.get_clustering_data(INDEX_TO_GRAB)
    pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)
    run_pointcloud_point_picker(pcd)


if __name__ == '__main__':
    main()
