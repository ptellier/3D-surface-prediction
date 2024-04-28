import numpy as np
import open3d as o3d

from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from utils.cluster_normals import ClusterNormals
from utils.surface_normals import pcd_img_to_o3d_pcd
from neura_modules.coco_mask_iou_score.coco_mask_iou_score import CocoMaskIoUScore

DATASET_FOLDER_PATH = './datasets/manual_dataset/'
EXAMPLE_INDEX = 5
TORCH_DEVICE = 'cpu'

IMAGES_PATH = DATASET_FOLDER_PATH+'/images/'
GT_DATASET_PATH = DATASET_FOLDER_PATH+'/annotations/surface_annotations.json'
GT_INSTANCE_SEGMENTATIONS_PATH = DATASET_FOLDER_PATH+'/annotations/instance_segmentations.json'
SELECTED_IMG_IDS = [num for num in range(1,26)]
GT_CATEGORY_TO_DILATION = {
    'edge': 0,
    'corner': 20,
}
RESTRICT_MASKS = True
PLOT_MASKS = False


if __name__ == '__main__':
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    # image, point_cloud_np_array, gt_annotations, cluster_distances_np_array, num_neighbours_np_array = manual_dataset[EXAMPLE_INDEX]
    image, point_cloud_np_array, gt_annotations = manual_dataset.get_clustering_data(EXAMPLE_INDEX)
    coco_mask_iou_score = CocoMaskIoUScore(
        gt_dataset_path=GT_DATASET_PATH,
        gt_instance_segmentations_path=GT_INSTANCE_SEGMENTATIONS_PATH,
        images_path=IMAGES_PATH,
        selected_img_ids=SELECTED_IMG_IDS,
        dilations=GT_CATEGORY_TO_DILATION,
        restrict_masks=RESTRICT_MASKS,
        plot_masks=PLOT_MASKS
    )

    merged_category_masks = coco_mask_iou_score.get_merged_masks_across_categories(EXAMPLE_INDEX)
    gt_labels = (merged_category_masks.cpu().numpy().flatten())
    pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)


    cluster_normals = ClusterNormals(
        pcd,
        lambda arr: 0.1,
        normal_estimation_radius=0.02,
        voxel_down_sample_size=0.01,
        orientation_ref=np.array([0.0, 0.0, 1.0]),
        gt_labels=gt_labels,
        image_id = EXAMPLE_INDEX
    )

    cluster_normals.pcd.paint_uniform_color([0.5, 0.5, 0.5])
    k = [1,2,3]
    cluster_normals.cluster_normals(radius=0.02, k=k)
    print(cluster_normals.get_gt_labels().shape)

    # o3d.visualization.draw_geometries([cluster_normals.pcd])

