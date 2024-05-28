import numpy as np
from numpy import array, ndarray
import open3d as o3d
from open3d.utility import Vector3dVector
from open3d.geometry import PointCloud, TriangleMesh

from nexera_packages.utilities.o3d_functions import pcd_img_to_o3d_pcd
from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from src.clustering.cluster_normals import ClusterNormals
from src.constants import MANUAL_DATASET_FOLDER_PATH
from src.make_features import get_manual_dataset_gt_labels
from src.utils.time_diff import TimeDiffPrinter
from src.visualizers.pointcloud_point_picker import run_pointcloud_point_picker

INDEX_TO_GRAB = 1
GREEN = array([0.0, 0.6, 0.0])
BLACK = array([0.1, 0.1, 0.1])


def visualize_normals(indices: ndarray, pcd: PointCloud):
    n = len(indices)
    print(f'visualizing {n} normals')
    normals = array(pcd.normals)[indices]
    normals_pcd = pcd_img_to_o3d_pcd(pcd=normals.reshape((n, 1, 3)))

    triangle_mesh = TriangleMesh()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(normals_pcd)
    vis.add_geometry(triangle_mesh.create_coordinate_frame())
    vis.run()


def main():
    time_printer = TimeDiffPrinter()
    time_printer.start()
    manual_dataset = ManuallyAnnotatedDataset(folder_path=MANUAL_DATASET_FOLDER_PATH)
    time_printer.print('Loaded manual dataset')

    image, point_cloud_np_array, gt_mask_annotations = manual_dataset.get_clustering_data(INDEX_TO_GRAB)
    time_printer.print('Retrieved a clustering datum')

    pcd = pcd_img_to_o3d_pcd(pcd=point_cloud_np_array)
    time_printer.print('Make numpy array and image array into a point cloud object')

    gt_labels = get_manual_dataset_gt_labels()
    time_printer.print('Get G.T. labels')

    cluster_normals = ClusterNormals(
        pcd=pcd,
        normal_estimation_radius=0.02,
        voxel_down_sample_size=0.005,
        orientation_ref=array([0.0, 0.0, 1.0]),
        gt_labels=gt_labels,
        image_id=INDEX_TO_GRAB,
    )
    time_printer.print('Estimate surface normals and construct kd-tree')

    point_indices = run_pointcloud_point_picker(cluster_normals.pcd_downsampled)
    time_printer.start()

    num_points = len(cluster_normals.pcd_downsampled.points)
    point_colors = np.full((num_points, 3), BLACK)
    radius_searches_indices = []
    for point_index in point_indices:
        point_indices_in_radius = cluster_normals.find_points_in_radius(anchor=point_index, radius=0.02)
        point_colors[point_indices_in_radius] = GREEN
        radius_searches_indices.append(point_indices_in_radius)
    cluster_normals.pcd_downsampled.colors = Vector3dVector(point_colors)
    time_printer.print('Color points in the point clouds by radius search')

    for radius_search_indices in radius_searches_indices:
        visualize_normals(indices=array(radius_search_indices), pcd=cluster_normals.pcd_downsampled)


if __name__ == '__main__':
    main()
