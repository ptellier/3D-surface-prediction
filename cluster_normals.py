import numpy as np
import open3d as o3d

from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from utils.cluster_normals import ClusterNormals
from utils.surface_normals import estimate_surface_normals, pcd_img_to_o3d_pcd

DATASET_FOLDER_PATH = './datasets/manual_dataset/'
EXAMPLE_INDEX = 1

if __name__ == '__main__':
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    image, point_cloud_np_array, mask_annotations = manual_dataset[EXAMPLE_INDEX]

    pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)

    cluster_normals = ClusterNormals(
        pcd,
        lambda arr: 0.1,
        normal_estimation_radius=0.02,
        voxel_down_sample_size=0.01,
        orientation_ref=np.array([0.0, 0.0, 1.0])
    )

    cluster_normals.pcd.paint_uniform_color([0.5, 0.5, 0.5])

    cluster_normals.cluster_normals(radius=0.02, k=3)

    # o3d.visualization.draw_geometries([cluster_normals.pcd])

