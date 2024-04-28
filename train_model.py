from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from neura_modules.coco_mask_iou_score.coco_mask_iou_score import CocoMaskIoUScore
from utils.surface_normals import pcd_img_to_o3d_pcd

DATASET_FOLDER_PATH = './datasets/manual_dataset'
EXAMPLE_INDEX = 1
TORCH_DEVICE = 'cuda'

IMAGES_PATH = DATASET_FOLDER_PATH+'/images/'
GT_DATASET_PATH = DATASET_FOLDER_PATH+'/annotations/surface_annotations.json'
GT_INSTANCE_SEGMENTATIONS_PATH = DATASET_FOLDER_PATH+'/annotations/instance_segmentations.json'
SELECTED_IMG_IDS = [1]
DT_TO_GT_CATEGORY_NAMES = {
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


def transform_gt_to_tensor():
    pass


if __name__ == '__main__':
    # manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    # image, point_cloud_np_array, gt_annotations, cluster_distances_np_array, num_neighbours_np_array = manual_dataset[EXAMPLE_INDEX]
    # pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)

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


    print(
        merged_category_masks.cpu().numpy()
    )
