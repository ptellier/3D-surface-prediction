from itertools import chain
import cv2 as cv
import torch
from matplotlib.figure import Figure
from pycocotools.coco import COCO
import numpy as np
from numpy import ndarray, array
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from torch import Tensor

from neura_modules.abstract_neura_module import NeuraModule

from neura_modules.coco_mask_iou_score.coco_tensor_dict import CocoTensorDict
from neura_modules.utils.data_type import DataType

CONFIGS_REQUIRED = [("plot_masks", DataType.BOOL), ("restrict_masks", DataType.BOOL), ("dilations", DataType.DICT_STRING_INT)]
INPUTS_REQUIRED = ['dt_dataset_path', 'gt_dataset_path', 'gt_instance_segmentations_path', 'images_path', 'dt_to_gt_categories',
                   'selected_dt_categories', 'selected_image_ids']
OUTPUTS_AVAILABLE = ['scores', 'figures']

TORCH_DEVICE = 'cuda'

INDEX_TO_COLOR = {  # RGB colors
    1:  array([0.411, 0.160, 0.768]),
    2:  array([0.066, 0.572, 0.909]),
    3:  array([0.000, 0.364, 0.364]),
    4:  array([0.623, 0.094, 0.325]),
    5:  array([0.980, 0.301, 0.337]),
    6:  array([0.341, 0.015, 0.031]),
    7:  array([0.098, 0.501, 0.219]),
    8:  array([0.000, 0.176, 0.611]),
    9:  array([0.933, 0.325, 0.545]),
    10: array([0.698, 0.525, 0.000]),
    11: array([0.000, 0.615, 0.603]),
    12: array([0.070, 0.215, 0.286]),
    13: array([0.541, 0.219, 0.000]),
    14: array([0.647, 0.431, 1.000])
}

INSTANCE_ID = 1
INSTANCE_NAME = 'object_instance'

class CocoMaskIoUScore(NeuraModule):
    """
    Attributes
    ----------
    coco_dt: COCO
        COCO dataset instance for detected masks.
    coco_gt: COCO
        COCO dataset instance for ground-truth masks.
    coco_gt_instance_segmentations: COCO
        TODO
    dt_dataset_path: str
        Path to the detected mask dataset.
    gt_dataset_path: str
        Path to the ground-truth mask dataset.
    gt_instance_segmentations_path: str
        TODO
    images_path: str
        Path to the folder containing all images for both datasets.
    selected_img_ids: list[int]
        IDs of selected COCO dataset images. Only these images and their masks are used when running the module.
    category_name_to_dilation_size: dict[str, list[int]]
        Mapping from a category name in the detected COCO dataset to the radius of a circular dilation
        to apply to the category's masks.
    plot_masks: bool
        Whether to create a matplotlib figure for each selected image.
    restrict_masks: bool
        TODO
    dt_to_gt_category_ids: dict[int, list[int]]
        Mapping from category IDs in the detected COCO dataset to the ground-truth COCO dataset.
    category_id_to_dilation_structuring_element: dict[int, cv.Mat]
        Mapping from category IDs in the detected COCO dataset to an OpenCV structuring element to perform dilations.
    """

    def __init__(self):
        super().__init__()
        self.configs_reqd = CONFIGS_REQUIRED
        self.inputs_reqd = INPUTS_REQUIRED
        self.outputs_available = OUTPUTS_AVAILABLE

    def setup_module(self, inputs: dict[str, object]) -> None:
        """
        Loads the ground-truth COCO mask datasets. Sets the module's fields for selected images and their file paths.
        Adds a "dilation" to (potentially) each mask in the ground-truth masks for selected images.

        Arguments:
            inputs: A dictionary containing all the inputs for the module.
                    The key is the variable name and the value is the variable's value.
        """
        self.gt_dataset_path: str = inputs['gt_dataset_path']
        self.dt_dataset_path: str = inputs['dt_dataset_path']
        self.gt_instance_segmentations_path: str = inputs['gt_instance_segmentations_path']
        self.images_path: str = inputs['images_path']

        self.selected_img_ids: list[int] = inputs['selected_image_ids']

        self.dt_to_gt_category_names: dict[str, list[str]] = inputs['dt_to_gt_categories']

        self.category_name_to_dilation_size = self.configs['dilations']
        self.restrict_masks: bool = self.configs['restrict_masks']
        self.plot_masks: bool = self.configs['plot_masks']

        self.coco_dt = COCO(self.dt_dataset_path)
        self.coco_gt = COCO(self.gt_dataset_path)
        self.coco_gt_instances = COCO(self.gt_instance_segmentations_path)

        selected_dt_category_names = inputs['selected_dt_categories']
        selected_gt_category_names = list(set(chain.from_iterable(self.dt_to_gt_category_names.values())))
        selected_dt_category_ids = [self.coco_dt.getCatIds(catNms=name)[0] for name in selected_dt_category_names]
        selected_gt_category_ids = [self.coco_gt.getCatIds(catNms=name)[0] for name in selected_gt_category_names]

        self.dt_tensor_dict: CocoTensorDict = CocoTensorDict(selected_dt_category_ids, selected_dt_category_names, self.selected_img_ids)
        self.gt_tensor_dict: CocoTensorDict = CocoTensorDict(selected_gt_category_ids, selected_gt_category_names, self.selected_img_ids)
        self.inst_tensor_dict: CocoTensorDict = CocoTensorDict([INSTANCE_ID], [INSTANCE_NAME], self.selected_img_ids)
        self.dt_tensor_dict.extract_tensors_from_coco(self.coco_dt)
        self.gt_tensor_dict.extract_tensors_from_coco(self.coco_gt)
        self.inst_tensor_dict.extract_tensors_from_coco(self.coco_gt_instances)

        self.category_id_to_dilation_structuring_element: dict[int, int] = {}
        if self.restrict_masks:
            self.gt_tensor_dict.set_instance_masks(self.inst_tensor_dict, INSTANCE_ID)
        self._setup_gt_mask_dilations()
        self.dilate_gt_masks(self.selected_img_ids)
        if self.restrict_masks:
            self.gt_tensor_dict.clip_by_instance_masks()

        self._setup_dt_to_gt_mappings()

    def run_module(self, inputs: dict[str, any]) -> dict[str, any]:
        """
        Runs the coco mask scoring module which computes the Intersection over Union (IoU) score
        for all mask categories and COCO dataset image IDs specified.
        This optionally returns a set of figures for each selected image comparing ground-truth
        to detected masks with IoU scores.
        This function assumes `setup_module()` has already been run.

        Arguments:
            inputs: A dictionary containing all the inputs for the module.
                    The key is the variable name and the value is the variable's value.

        Returns:
            A dictionary containing the results of the module. The key is the variable name
            and the value is the variable's value. All outputs must follow the IO standards
            (see README for details).
        """
        iou_scores = self.compare_coco_iou()
        figs = None
        if self.plot_masks:
            figs = []
            for img_id in self.selected_img_ids:
                fig_title = f'IoU Scores for img_id={img_id}'
                figs.append(self.make_image_with_masks_figure(img_id, iou_scores[img_id], fig_title))
        return {
            'scores': iou_scores,
            'figures': figs
        }

    def _setup_dt_to_gt_mappings(self):
        self.dt_to_gt_category_ids: dict[int, list[int]] = {}
        for dt_category_name in self.dt_tensor_dict.category_names:
            dt_category_id = self.dt_tensor_dict.category_id_from_name(dt_category_name)
            gt_category_names = self.dt_to_gt_category_names[dt_category_name]
            gt_category_ids = self.gt_tensor_dict.category_ids_from_names(gt_category_names)
            self.dt_to_gt_category_ids[dt_category_id] = gt_category_ids

        self.gt_to_dt_category_ids: dict[int, list[int]] = {}
        for dt_category_id in self.dt_to_gt_category_ids:
            for gt_category_id in self.dt_to_gt_category_ids[dt_category_id]:
                self.gt_to_dt_category_ids[gt_category_id] = [dt_category_id]

    def _setup_gt_mask_dilations(self) -> None:
        """
        Sets up the OpenCV structuring elements to dilate masks in the ground-truth COCO dataset
        by mask category. Resultant structuring elements depend on the `"dilations"` config
        which maps the COCO mask category names to dilation sizes. No dilation is added for category
        names that are not specified.
        """
        for category_name, dilation_size in self.category_name_to_dilation_size.items():
            get_ids_result = self.coco_gt.getCatIds(catNms=[category_name])
            if len(get_ids_result) == 0:
                raise ValueError(f'Category "{category_name}" in dilations config not found in ground-truth COCO dataset')
            category_id = get_ids_result[0]
            if dilation_size != 0:
                self.category_id_to_dilation_structuring_element[category_id] = create_dilation_structuring_element(dilation_size)

    def dilate_gt_masks(self, img_ids: list[int]) -> None:
        """
        TODO: UPDATE/CHECK DESCRIPTION
        Adds a "dilation" to (potentially) each mask in the ground-truth mask COCO dataset
        under the selected image IDs according to `category_id_to_dilation_structuring_element`.
        This maps the COCO mask category IDs to structuring elements.
        No dilation is added for category names that are not specified.

        Args:
            img_ids: IDs of images to dilate masks for in the ground-truth dataset.
        """
        for img_id in img_ids:
            for category_id in self.category_id_to_dilation_structuring_element:
                structuring_element = self.category_id_to_dilation_structuring_element[category_id]
                self.gt_tensor_dict.dilate(img_id, category_id, structuring_element)

    def compare_coco_iou(self) -> dict[int, dict[str, float]]:
        """
        Computes the "Intersection over Union" (IoU) score for selected categories
        and images in the detected COCO dataset.

        Returns:
            A nested dictionary of IoU scores
            by COCO image ID (int) and COCO mask category name (str).
        """
        img_scores_by_category = {}
        dt_category_ids = self.dt_tensor_dict.category_ids
        for img_id in self.selected_img_ids:
            img_scores_by_category[img_id] = {}
            for dt_category_id in dt_category_ids:
                dt_category_name = self.dt_tensor_dict.category_name_from_id(dt_category_id)
                gt_category_ids = self.dt_to_gt_category_ids[dt_category_id]
                if len(gt_category_ids) == 0:
                    img_scores_by_category[img_id][dt_category_name] = np.NaN
                else:
                    dt_merged_mask = merge_category_masks_in_image(self.dt_tensor_dict, [dt_category_id], img_id)
                    gt_merged_mask = merge_category_masks_in_image(self.gt_tensor_dict, gt_category_ids, img_id)
                    img_scores_by_category[img_id][dt_category_name] = torch_iou(dt_merged_mask, gt_merged_mask)
        return img_scores_by_category

    def make_image_with_masks_figure(self,
                                     img_id: int,
                                     category_name_to_score: dict[str, float],
                                     fig_title: str = '') -> Figure:
        """
        Produces a figure for an image that displays the masks for detected and ground-truth datasets,
        and their "scores" in a legend.

        Args:
            img_id: ID of COCO dataset image to display.
            category_name_to_score: Mapping from category names to their "scores".
            fig_title: Title to add to the figure.

        Returns:
            A Matplotlib Figure.
        """
        gt_category_ids = self.gt_tensor_dict.category_ids
        dt_category_ids = self.dt_tensor_dict.category_ids
        dt_color_map, gt_color_map = {}, {}

        for dt_category_id in dt_category_ids:
            dt_color_map[dt_category_id] = dt_category_id

        for gt_category_id in gt_category_ids:
            gt_color_map[gt_category_id] = self.gt_to_dt_category_ids[gt_category_id][0]  # assume unique mapping for now

        img = self.coco_gt.loadImgs(img_id)[0]
        np_img = io.imread(self.images_path + img['file_name'])
        fig, axes = plt.subplots(1, 2)
        fig.suptitle(fig_title)
        self._add_legend_to_plot(fig, dt_category_ids, category_name_to_score)
        self._add_masks_to_plot(self.dt_tensor_dict, np_img, img_id, dt_category_ids, dt_color_map, axes[0],
                                title="Predicted masks")
        self._add_masks_to_plot(self.gt_tensor_dict, np_img, img_id, gt_category_ids, gt_color_map, axes[1],
                                title="Ground-truth masks")
        return fig

    def _add_legend_to_plot(self, fig: Figure, dt_category_ids: list[int], category_name_to_score: dict[str, float]) -> None:
        """
        Update the Matplotlib Figure `fig` with a legend:
         - Show the name of each detected category with its "score".
         - Color legend entries by category according to colors from `CATEGORY_ID_TO_COLOR`.

        Args:
            fig: Matplotlib figure to modify.
            dt_category_ids: IDs of mask categories in detected COCO dataset to show in legend.
            category_name_to_score: A mapping from each mask category name in the detected COCO dataset
                                    to its score to display in the legend.
        """
        patches = []
        for dt_category_id in dt_category_ids:
            dt_category_name = self.dt_tensor_dict.category_name_from_id(dt_category_id)
            color = INDEX_TO_COLOR[dt_category_id].tolist()
            percent_score = round(category_name_to_score[dt_category_name] * 100, 1)
            label = f'{dt_category_name}, score={percent_score}%'
            patch = Patch(color=color, label=label)
            patches.append(patch)
        fig.legend(handles=patches, loc='lower center')

    def _add_masks_to_plot(self,
                           tensor_dict: CocoTensorDict,
                           np_img: ndarray,
                           img_id: int,
                           category_ids: list[int],
                           category_id_to_color_index: dict[int, int],
                           ax: Axes,
                           title: str) -> None:
        """
        Update the Matplotlib Axes `ax`
         - Plot the image `np_img`.
         - Add all the given masks in a loaded COCO dataset under the image id and category ids to `ax`.
         - Color masks by category according to `category_id_to_color_index`, getting colors from `CATEGORY_ID_TO_COLOR`.
         - Add a `title`.

        Args:
            coco: A COCO dataset instance. TODO: update parameters
            np_img: Numpy array representing an image.
            img_id: Image ID to select COCO masks from.
            category_ids: Category IDs to select COCO masks from.
            category_id_to_color_index: Mapping from category IDs to color indices to color.
            ax: Matplotlib Axes to plot the image and masks on.
            title: Title to add to Figure Axes.
        """
        ax.imshow(np_img)
        ax.set_title(title, fontsize=12, color='black')
        if len(category_ids) > 0:
            ax.set_autoscale_on(False)
            for category_id in category_ids:
                tensor_masks = tensor_dict[img_id, category_id]
                for tensor_mask in tensor_masks:
                    color_index = category_id_to_color_index[category_id]
                    color = INDEX_TO_COLOR[color_index]

                    np_mask = tensor_mask.cpu().detach().numpy()
                    img = np.ones((np_mask.shape[0], np_mask.shape[1], 3))
                    for i in range(3):
                        img[:, :, i] = color[i]
                    ax.imshow(np.dstack((img, np_mask * 0.5)))


def merge_category_masks_in_image(tensor_dict: CocoTensorDict, category_ids: list[int], img_id: int):
    """
    Returns all the COCO mask annotations for selected categories and image as one merged mask.

    Args:
        tensor_dict: A CocoTensorDict instance. TODO: update description
        category_ids: IDs of Categories to select masks from.
        img_id: ID of image to select COCO masks from.

    Returns:
        A single pytorch tensor with the same dimensions of the image with the given ID.
    """
    merged_masks = torch.zeros(tensor_dict.image_shape(img_id), dtype=torch.bool, device=TORCH_DEVICE)
    for category_id in category_ids:
        tensors_to_merge = tensor_dict[img_id, category_id]
        for tensor_mask in tensors_to_merge:
            merged_masks = torch.logical_or(merged_masks, tensor_mask)
    return merged_masks


def binary_mask_to_rle_np(binary_mask: ndarray) -> tuple[ndarray, list[int], int]:
    """
    Takes a boolean numpy array representing a segmentation mask and produces
    the `pycocotools` run-length encoding, image dimensions, the area of the mask.

    Arguments:
        binary_mask: A boolean numpy array representing a segmentation mask. Same dimensions as image it is for.

    Returns:
        A `pycocotools` run-length-encoding which is an array that specifies column wise how many consecutive pixels
        are a part of the mask, then how many are not, then how many are, etc. until the end of the image.
        Image-dimensions for the image the mask is for.
        The total area of the mask.
    """
    img_dimensions = list(binary_mask.shape)

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))
    area = int(np.sum(lengths[1::2]))

    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle = lengths.tolist()
    return rle, img_dimensions, area


def create_dilation_structuring_element(dilation_radius: int) -> cv.Mat:
    """
    Creates a circular OpenCV structuring element of `dilation_radius` in pixels.

    Args:
        dilation_radius: Radius of the desired circular structuring element in pixels.

    Returns:
        An OpenCV structuring element in the form of an OpenCV `Mat`.
    """
    kernel_size = (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    anchor = (dilation_radius, dilation_radius)
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size, anchor)


def torch_iou(mask1: Tensor, mask2: Tensor) -> float:
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return (intersection / union).item()

def torch_intersection(mask1: Tensor, masks: Tensor):
    return (mask1 * masks).sum()

############################
# TODO: REMOVE THESE BEFORE PR
# DEBUGGING FUNCTIONS

def to_np(tens: Tensor):
    return tens.cpu().detach().numpy()


def plt_mask(tensor_mask: Tensor):
    fig, ax = plt.subplots(1, 1)
    color_index = 1
    color = INDEX_TO_COLOR[color_index]
    np_mask = tensor_mask.cpu().detach().numpy()
    img = np.ones((np_mask.shape[0], np_mask.shape[1], 3))
    for i in range(3):
        img[:, :, i] = color[i]
    ax.imshow(np.dstack((img, np_mask * 0.5)))
    fig.show()