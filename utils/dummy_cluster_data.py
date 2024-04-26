import numpy as np
from numpy import ndarray


def make_dummy_class_data(d: int) -> tuple[ndarray, ndarray]:
    patches_x = [(0, 640), (0, 640), (640, 1280), (640, 1280)]
    patches_y = [(0, 512), (512, 1024), (0, 512), (512, 1024)]
    patches_fills = [np.random.rand(d)*2, np.random.rand(d)*2, np.random.rand(d)*2, np.random.rand(d)*2]
    patches_classes = [1, 2, 3, 4]

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
    what = 1+2

