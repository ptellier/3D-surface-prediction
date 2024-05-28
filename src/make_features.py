import numpy as np
from numpy import ndarray

from src.constants import MANUAL_DATASET_FOLDER_PATH, MANUAL_DATASET_IMAGES_PATH, \
    MANUAL_DATASET_ANNOTATIONS_PATH, MANUAL_DATASET_INSTANCE_SEGMENTATIONS_PATH, \
    MANUAL_DATASET_DT_TO_GT_CATEGORIES, MANUAL_DATASET_GT_CATEGORY_TO_DILATION, \
    MANUAL_DATASET_GT_CATEGORY_PRIORITIES, DUMMY_DETECTED_DATASET_ANNOTATIONS_PATH
from src.dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from src.clustering.cluster_normals import ClusterNormals
from nexera_packages.utilities.o3d_functions import pcd_img_to_o3d_pcd
from neura_modules.coco_mask_iou_score import CocoMaskIoUScore, CocoMaskIoUScoreInputs
from src.utils.time_diff import TimeDiffPrinter

EXAMPLE_INDEX = 1

DT_DATASET_PATH = DUMMY_DETECTED_DATASET_ANNOTATIONS_PATH  # TODO: figure out how to handle this better
SELECTED_DT_CATEGORIES = ['ingestible_shallow', 'ingestible_medium', 'flat', 'other']
SELECTED_IMG_IDS = [num for num in range(1, 1+1)]
RESTRICT_MASKS = True
PLOT_MASKS = False


def get_manual_dataset_gt_labels() -> ndarray:
    inputs: CocoMaskIoUScoreInputs = dict(
        dt_dataset=DT_DATASET_PATH,
        gt_dataset_path=MANUAL_DATASET_ANNOTATIONS_PATH,
        gt_instance_segmentations_path=MANUAL_DATASET_INSTANCE_SEGMENTATIONS_PATH,
        dt_to_gt_categories=MANUAL_DATASET_DT_TO_GT_CATEGORIES,
        images_path=MANUAL_DATASET_IMAGES_PATH,
        selected_dt_categories=SELECTED_DT_CATEGORIES,
        selected_image_ids=SELECTED_IMG_IDS,
    )

    configs = dict(
        dilations=MANUAL_DATASET_GT_CATEGORY_TO_DILATION,
        restrict_masks=RESTRICT_MASKS,
        plot_masks=PLOT_MASKS,
        crop_box=None,
        category_priorities=MANUAL_DATASET_GT_CATEGORY_PRIORITIES,
    )

    coco_mask_iou_score = CocoMaskIoUScore()
    coco_mask_iou_score.set_configs(configs)
    outputs = coco_mask_iou_score.run_module(inputs)

    merged_category_masks = coco_mask_iou_score.gt_tensor_dict.get_merged_category_id_matrix(img_id=EXAMPLE_INDEX)
    gt_labels = merged_category_masks.cpu().numpy().flatten()
    return gt_labels


def main():
    time_printer = TimeDiffPrinter()
    time_printer.start()
    manual_dataset = ManuallyAnnotatedDataset(folder_path=MANUAL_DATASET_FOLDER_PATH)
    time_printer.print('Loaded manual dataset')

    _, point_cloud_np_array, _ = manual_dataset.get_clustering_data(EXAMPLE_INDEX)
    time_printer.print('Retrieved a clustering datum')

    gt_labels = get_manual_dataset_gt_labels()
    time_printer.print('Get G.T. labels')

    pcd = pcd_img_to_o3d_pcd(point_cloud_np_array)
    time_printer.print('Make numpy array and image array into a pointcloud object')

    cluster_normals = ClusterNormals(
        pcd,
        normal_estimation_radius=0.02,
        voxel_down_sample_size=0.01,
        orientation_ref=np.array([0.0, 0.0, 1.0]),
        gt_labels=gt_labels,
        image_id=EXAMPLE_INDEX
    )
    time_printer.print('Estimate surface normals and construct kd-tree')

    cluster_normals.pcd.paint_uniform_color([0.5, 0.5, 0.5])
    k = [1, 2, 3]
    cluster_normals.cluster_normals(radius=0.02, k=k)
    cluster_normals.get_downsampled_gt_labels()

    # o3d.visualization.draw_geometries([cluster_normals.pcd])


if __name__ == '__main__':
    main()
