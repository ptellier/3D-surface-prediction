import numpy as np
from numpy import array
import open3d as o3d
from open3d.utility import Vector3dVector

from nexera_packages.utilities.o3d_functions import pcd_img_to_o3d_pcd
from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from src.clustering.cluster_normals import ClusterNormals
from src.constants import MANUAL_DATASET_FOLDER_PATH
from src.make_features import get_manual_dataset_gt_labels
from src.visualizers.pointcloud_point_picker import run_pointcloud_point_picker

INDEX_TO_GRAB = 1
GREEN = array([0.0, 0.6, 0.0])
BLACK = array([0.1, 0.1, 0.1])

def main():
    manual_dataset = ManuallyAnnotatedDataset(folder_path=MANUAL_DATASET_FOLDER_PATH)
    image, point_cloud_np_array, gt_mask_annotations = manual_dataset.get_clustering_data(INDEX_TO_GRAB)
    pcd = pcd_img_to_o3d_pcd(pcd=point_cloud_np_array)

    gt_labels = get_manual_dataset_gt_labels()

    cluster_normals = ClusterNormals(
        pcd=pcd,
        normal_estimation_radius=0.02,
        voxel_down_sample_size=0.01,
        orientation_ref=array([0.0, 0.0, 1.0]),
        gt_labels=gt_labels,
        image_id=INDEX_TO_GRAB,
    )

    point_indices = run_pointcloud_point_picker(cluster_normals.pcd_downsampled)

    num_points = len(cluster_normals.pcd_downsampled.points)
    point_colors = np.full((num_points, 3), BLACK)
    for point_index in point_indices:
        point_indices_in_radius = cluster_normals.find_points_in_radius(anchor=point_index, radius=0.02)
        point_colors[point_indices_in_radius] = GREEN
    cluster_normals.pcd_downsampled.colors = Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cluster_normals.pcd_downsampled)
    vis.run()


if __name__ == '__main__':
    main()
