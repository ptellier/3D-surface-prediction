import torch
import torchvision
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from numpy import array, ndarray
import torch
import matplotlib.pyplot as plt
import cv2
from time_diff import TimeDiffPrinter, time_func

SAM_CHECKPOINT_PATH = './sam/sam_vit_h_4b8939.pth'
SAM_MODEL_TYPE = 'vit_h'
TEST_IMAGE_PATH = 'datasets/validation-dataset-real/bin_x_plus/2024-02-22_12-18-59/RGBImage.png'
DEVICE = 'cuda'

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

stop_watch = TimeDiffPrinter()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1)) * color.reshape((1, 1, -1))
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plot_image(image) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()


@time_func()
def setup_sam() -> SamPredictor:
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


@time_func()
def sam_predict(predictor: SamPredictor, image,
                input_point: ndarray, input_label: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    predictor.set_image(image)
    return predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )


@time_func()
def sam_display_prediction(masks: ndarray, scores: ndarray, image, input_point: ndarray, input_label: ndarray):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score: .3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def main():
    image = cv2.imread(TEST_IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_point = array([[500, 375]])
    input_label = array([1])
    plot_image(image)
    predictor = setup_sam()
    masks, scores, logits = sam_predict(predictor, image, input_point, input_label)
    sam_display_prediction(masks, scores, image, input_point, input_label)


if __name__ == "__main__":
    main()
