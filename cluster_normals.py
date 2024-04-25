import numpy as np
import open3d as o3d

from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from utils.surface_normals import estimate_surface_normals, pcd_img_to_o3d_pcd

DATASET_FOLDER_PATH = './datasets/manual_dataset/'
EXAMPLE_INDEX = 1

if __name__ == '__main__':
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    image, point_cloud_np_array, mask_annotations = manual_dataset[EXAMPLE_INDEX]

    pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)
    pcd_with_normals = estimate_surface_normals(
        pcd,
        voxel_down_sample_size=0.01,
        normal_estimation_radius=0.02,
        orientation_ref=np.array([0.0, 0.0, 1.0]),
        inplace=False
    )

    pcd_with_normals.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_kd_tree = o3d.geometry.KDTreeFlann(pcd_with_normals)

    num_points, idx, coordinates = pcd_kd_tree.search_radius_vector_3d(query=pcd_with_normals.points[1500], radius=0.05)
    np.asarray(pcd_with_normals.colors)[idx[1:], :] = [0, 0, 1]

    o3d.visualization.draw_geometries([pcd_with_normals])
