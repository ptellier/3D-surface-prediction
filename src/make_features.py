import numpy as np

from src.dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from src.clustering.cluster_normals import ClusterNormals
from src.utils.surface_normals import pcd_img_to_o3d_pcd
from neura_modules.coco_mask_iou_score import CocoMaskIoUScore, CocoMaskIoUScoreInputs

DATASET_FOLDER_PATH = '../datasets/manual_dataset/'
EXAMPLE_INDEX = 1

IMAGES_PATH = DATASET_FOLDER_PATH+'/images/'
DT_DATASET_PATH = TODO
GT_DATASET_PATH = DATASET_FOLDER_PATH+'/annotations/surface_annotations.json'
GT_INSTANCE_SEGMENTATIONS_PATH = DATASET_FOLDER_PATH+'/annotations/instance_segmentations.json'
SELECTED_DT_CATEGORIES = ['ingestible_shallow', 'ingestible_medium', 'flat', 'other']
SELECTED_IMG_IDS = [num for num in range(1, 26)]
DT_TO_GT_CATEGORIES = {
    'ingestible_shallow':  ['convex', 'corner', 'flat_edge', 'flat_corner', 'rod_like', 'convex_rim'],
    'ingestible_medium':   [],
    'ingestible_deep':     ['spherical_grasp'],
    'flat':                ['flat', 'bumpy'],
    'other':   ['edge', 'convex_edge']
}
GT_CATEGORY_TO_DILATION = {
    'edge': 0,
    'corner': 20,
}
RESTRICT_MASKS = True
PLOT_MASKS = False


if __name__ == '__main__':
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    image, point_cloud_np_array, gt_annotations = manual_dataset.get_clustering_data(EXAMPLE_INDEX)

    inputs: CocoMaskIoUScoreInputs = dict(
        dt_dataset_path=DT_DATASET_PATH,
        gt_dataset_path=GT_DATASET_PATH,
        gt_instance_segmentations_path=GT_INSTANCE_SEGMENTATIONS_PATH,
        dt_to_gt_categories=DT_TO_GT_CATEGORIES,
        images_path=IMAGES_PATH,
        selected_dt_categories=SELECTED_DT_CATEGORIES,
        selected_image_ids=SELECTED_IMG_IDS,
    )

    configs = dict(
        dilations=GT_CATEGORY_TO_DILATION,
        restrict_masks=RESTRICT_MASKS,
        plot_masks=PLOT_MASKS
    )

    coco_mask_iou_score = CocoMaskIoUScore()
    coco_mask_iou_score.set_configs(configs)
    outputs = coco_mask_iou_score.run_module(inputs)

    merged_category_masks = coco_mask_iou_score.gt_tensor_dict.get_merged_category_id_matrix(img_id=EXAMPLE_INDEX)
    gt_labels = (merged_category_masks.cpu().numpy().flatten())
    pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)

    cluster_normals = ClusterNormals(
        pcd,
        lambda arr: 0.1,
        normal_estimation_radius=0.02,
        voxel_down_sample_size=0.01,
        orientation_ref=np.array([0.0, 0.0, 1.0]),
        gt_labels=gt_labels,
        image_id=EXAMPLE_INDEX
    )

    cluster_normals.pcd.paint_uniform_color([0.5, 0.5, 0.5])
    k = [1, 2, 3]
    cluster_normals.cluster_normals(radius=0.02, k=k)
    cluster_normals.get_gt_labels()

    # o3d.visualization.draw_geometries([cluster_normals.pcd])

