import numpy as np
from numpy import ndarray
from torch import as_tensor

from utils.NormalsClusterClassifier import NormalsClusterClassifier

TORCH_DEVICE = 'cuda'

def make_dummy_class_data(d: int) -> tuple[ndarray, ndarray]:
    patches_x = [(0, 640), (0, 640), (640, 1280), (640, 1280)]
    patches_y = [(0, 512), (512, 1024), (0, 512), (512, 1024)]
    patches_fills = [np.random.rand(d)*2, np.random.rand(d)*2, np.random.rand(d)*2, np.random.rand(d)*2]
    patches_classes = [0, 1, 2, 3]

    features = np.zeros((1024, 1280, d))
    labels = np.zeros((1024, 1280))

    for patch_x, patch_y, patch_fill, patch_class in zip(patches_x, patches_y, patches_fills, patches_classes):
        x_0, x_1 = patch_x
        y_0, y_1 = patch_y
        features[y_0:y_1, x_0:x_1] = patch_fill
        labels[y_0:y_1, x_0:x_1] = patch_class
        noise = np.random.rand(1024, 1280, d)
        features = features + noise

    return features, labels


if __name__ == '__main__':
    dummy_feature_data, dummy_label_data = make_dummy_class_data(d=6)
    surface_classifier = NormalsClusterClassifier(
        n_inputs=6,
        n_classes=4,
        max_iter=400,
        learning_rate=0.05,
        weight_decay=0.1,
        init_scale=1,
        batch_size=1000,
        device=TORCH_DEVICE,
    )

    X, y = as_tensor(dummy_feature_data), as_tensor(dummy_label_data)
    surface_classifier.fit(X, y)
    mask_predictions = surface_classifier.predict(X)

