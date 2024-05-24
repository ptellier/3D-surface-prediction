import numpy as np
from numpy import ndarray

from src.dataset_loaders.manually_annotated_dataset import ManuallyAnnotatedDataset
from src.utils.NormalsClusterClassifier import NormalsClusterClassifier
from src.constants import TORCH_DEVICE

DATASET_FOLDER_PATH = '../datasets/manual_dataset'

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

def get_classifier_data(image_ids: list[int]) -> tuple[ndarray, ndarray, ndarray]:
    manual_dataset = ManuallyAnnotatedDataset(folder_path=DATASET_FOLDER_PATH)

    all_cluster_distances = []
    all_num_neighbours = []
    all_labels = []
    for image_id in image_ids:
        example = manual_dataset.get_training_data(index=image_id)
        cluster_distances, num_neighbours, labels = example
        all_cluster_distances.append(cluster_distances)
        all_num_neighbours.append(num_neighbours)
        all_labels.append(labels)

    X_train = np.concatenate(all_cluster_distances)
    y_train = np.concatenate(all_labels)
    num_neigh_train = np.concatenate(all_num_neighbours)

    has_label = y_train != -1
    X_train = X_train[has_label]
    y_train = y_train[has_label]
    num_neigh_train = num_neigh_train[has_label]

    y_train[y_train > 3] = 3  # janky but works for now

    return X_train, y_train, num_neigh_train


def train_surface_classifier(X: ndarray, y: ndarray, num_neigh: ndarray) -> NormalsClusterClassifier:
    surface_classifier = NormalsClusterClassifier(
        n_inputs=3,
        n_classes=4,
        max_iter=10000,
        learning_rate=0.001,
        weight_decay=0.1,
        init_scale=1,
        batch_size=1000,
        device=TORCH_DEVICE,
    )
    surface_classifier.fit(X, y, num_neigh)
    return surface_classifier


if __name__ == '__main__':
    X_train, y_train, num_neigh_train = get_classifier_data(TRAINING_IMAGE_IDS)
    n = len(y_train)
    trained_surface_classifier = train_surface_classifier(X_train, y_train, num_neigh_train)
    predictions = trained_surface_classifier.predict(X_train)
    mode_predictions = np.ones(n) * 1

    accuracy = np.sum(predictions == y_train) / n
    mode_dummy_accuracy = np.sum(mode_predictions == y_train) / n
    print(f'accuracy: {round(accuracy*100, 2)}%')
    print(f'dummy accuracy: {round(mode_dummy_accuracy * 100, 2)}%')
