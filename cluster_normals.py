import numpy as np
import open3d as o3d

from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from utils.NormalsClusterClassifier import NormalsClusterClassifier
from utils.cluster_normals import ClusterNormals
from utils.surface_normals import pcd_img_to_o3d_pcd

DATASET_FOLDER_PATH = './datasets/manual_dataset/'
EXAMPLE_INDEX = 1

TORCH_DEVICE = 'cuda'


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

    surface_classifier = NormalsClusterClassifier(
        n_inputs=6,
        max_iter=10_000,
        learning_rate=0.0001,
        weight_decay=0,
        init_scale=1,
        batch_size=1,
        device=TORCH_DEVICE,
    )

    # o3d.visualization.draw_geometries([cluster_normals.pcd])

