import numpy as np
from dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from neura_modules.coco_mask_iou_score.coco_mask_iou_score import CocoMaskIoUScore
from utils.NormalsClusterClassifier import NormalsClusterClassifier

DATASET_FOLDER_PATH = './datasets/manual_dataset'
TORCH_DEVICE = 'cuda'

IMAGES_PATH = DATASET_FOLDER_PATH+'/images/'
GT_DATASET_PATH = DATASET_FOLDER_PATH+'/annotations/surface_annotations.json'
GT_INSTANCE_SEGMENTATIONS_PATH = DATASET_FOLDER_PATH+'/annotations/instance_segmentations.json'
TRAINING_IMAGE_IDS = [1, 2]
GT_CATEGORY_TO_DILATION = {
    'edge': 0,
    'corner': 20,
}
RESTRICT_MASKS = True
PLOT_MASKS = False

if __name__ == '__main__':
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)
    coco_mask_iou_score = CocoMaskIoUScore(
        gt_dataset_path=GT_DATASET_PATH,
        gt_instance_segmentations_path=GT_INSTANCE_SEGMENTATIONS_PATH,
        images_path=IMAGES_PATH,
        selected_img_ids=TRAINING_IMAGE_IDS,
        dilations=GT_CATEGORY_TO_DILATION,
        restrict_masks=RESTRICT_MASKS,
        plot_masks=PLOT_MASKS
    )

    all_cluster_distances = []
    all_num_neighbours = []
    all_labels = []
    for image_id in TRAINING_IMAGE_IDS:
        example = manual_dataset.get_training_data(idx=image_id)
        cluster_distances, num_neighbours, labels = example
        all_cluster_distances.append(cluster_distances)
        all_num_neighbours.append(num_neighbours)
        all_labels.append(labels)

    X_train = np.concatenate(all_cluster_distances)
    y_train = np.concatenate(all_labels)
    training_num_neighbours = np.concatenate(all_num_neighbours)

    surface_classifier = NormalsClusterClassifier(
        n_inputs=3,
        n_classes=4,
        max_iter=400,
        learning_rate=0.05,
        weight_decay=0.1,
        init_scale=1,
        batch_size=1000,
        device=TORCH_DEVICE,
    )

    surface_classifier.fit(X_train, y_train, training_num_neighbours)



