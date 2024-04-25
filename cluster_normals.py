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
        voxel_down_sample_size=None,
        normal_estimation_radius=1,
        orientation_ref=np.array([0.0, 0.0, 1.0]),
        inplace=False
    )

    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_kd_tree = o3d.geometry.KDTreeFlann(pcd)

    pcd_kd_tree.search_radius_vector_3d(query=pcd.points[1500], radius=0.2)
    # np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]


