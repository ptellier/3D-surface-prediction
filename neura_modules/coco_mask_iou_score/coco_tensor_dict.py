from typing import Union
import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from torch import Tensor, tensor, Size

TORCH_DEVICE = 'cpu'

class CocoTensorDict:

    def __init__(self, category_ids: list[int], category_names: list[str], image_ids: list[int]):
        assert len(category_ids) == len(category_names), '`category_ids` and `category_names` should be the same length'
        self._tensor_dict: dict[int, dict[int, Union[list[Tensor], None]]] = {
            image_id: {category_id: None for category_id in category_ids} for image_id in image_ids
        }
        self._image_ids = image_ids
        self._category_ids = category_ids
        self._category_names = category_names
        self._category_id_to_name: dict[int, str] = {}
        self._category_name_to_id: dict[str, int] = {}
        for category_id, category_name in zip(category_ids, category_names):
            self._category_id_to_name[category_id] = category_name
            self._category_name_to_id[category_name] = category_id

        self._instances_mask_category_id = None
        self._instances_tensor_dict: CocoTensorDict = None
        self._instance_mask_indexes: dict[int, dict[int, Tensor]] = None

    def __getitem__(self, key) -> Union[list[Tensor], None]:
        image_id, category_id = key
        return self._tensor_dict[image_id][category_id]

    def __setitem__(self, key, new_value: list[Tensor]):
        image_id, category_id = key
        self._tensor_dict[image_id][category_id] = new_value

    def __delitem__(self, key):
        image_id, category_id = key
        del self._tensor_dict[image_id][category_id]

    def extract_tensors_from_coco(self, coco: COCO) -> None:
        for image_id in self._image_ids:
            for category_id in self._category_ids:
                annotations = coco.loadAnns(coco.getAnnIds(catIds=[category_id], imgIds=[image_id]))
                np_masks = map(coco.annToMask, annotations)
                tensor_masks = [tensor(np_mask, dtype=torch.bool, device=TORCH_DEVICE) for np_mask in np_masks]
                self[image_id, category_id] = tensor_masks

    def set_instance_masks(self, instances_tensor_dict: 'CocoTensorDict', instance_mask_category_id: int) -> None:
        self._instances_mask_category_id = instance_mask_category_id
        self._instances_tensor_dict = instances_tensor_dict
        self._instance_mask_indexes = {}
        for image_id in self._image_ids:
            tensor_instance_masks = torch.stack(self._instances_tensor_dict[image_id, instance_mask_category_id])
            self._instance_mask_indexes[image_id] = {}
            for category_id in self._category_ids:
                tensor_masks = self[image_id, category_id]
                list_indexes = torch.zeros(len(tensor_masks))
                for i, tensor_mask in enumerate(tensor_masks):
                    intersection_areas = torch.sum(torch.logical_and(tensor_mask, tensor_instance_masks), dim=(1, 2))
                    list_index = torch.argmax(intersection_areas)
                    list_indexes[i] = list_index
                self._instance_mask_indexes[image_id][category_id] = list_indexes

    def _get_instance_masks(self, image_id: int, category_id: int) -> list[Tensor]:
        mask_indexes = self._instance_mask_indexes[image_id][category_id]
        return [self._instances_tensor_dict[image_id, self._instances_mask_category_id][round(mask_index.item())] for mask_index in mask_indexes]

    def clip_by_instance_masks(self):
        for image_id in self._image_ids:
            for category_id in self._category_ids:
                masks = self._tensor_dict[image_id][category_id]
                instance_masks = self._get_instance_masks(image_id, category_id)
                for i in range(len(masks)):
                    self._tensor_dict[image_id][category_id][i] = torch.logical_and(masks[i], instance_masks[i])

    def dilate(self, image_id: int, category_id: int, dilation_structuring_element: cv2.Mat) -> None:
        tensor_masks = self[image_id, category_id]
        np_masks = [tensor_mask.cpu().detach().numpy().astype(np.uint8) for tensor_mask in tensor_masks]
        dilated_np_masks = [cv2.dilate(np_mask, dilation_structuring_element) for np_mask in np_masks]
        dilated_tensors = [tensor(np_mask, dtype=torch.bool, device=TORCH_DEVICE) for np_mask in dilated_np_masks]
        self[image_id, category_id] = dilated_tensors

    def get_merged_masks_across_categories(self, image_id: int, map_to_cat_id: dict[int, int]):
        merged_tensor = torch.zeros(self.image_shape(image_id), dtype=torch.int, device=TORCH_DEVICE)
        for category_id in self._category_ids:
            cat_masks = self[image_id, category_id]
            for cat_mask in cat_masks:
                merged_tensor = merged_tensor + map_to_cat_id[category_id]*cat_mask

        merged_tensor[merged_tensor == 0] = -1
        merged_tensor[merged_tensor > 3] = 3
        return merged_tensor

    @property
    def category_ids(self) -> list[int]:
        return self._category_ids

    @property
    def category_names(self) -> list[str]:
        return self._category_names

    @property
    def image_ids(self) -> list[int]:
        return self._image_ids

    def image_shape(self, image_id) -> Size:
        for category_id in self._category_ids:
            tensor_list = self[image_id, category_id]
            if len(tensor_list) > 0:
                return tensor_list[0].shape
        raise ValueError(f'No tensors with image ID {image_id}')

    def category_id_from_name(self, category_name: str) -> int:
        return self._category_name_to_id[category_name]

    def category_name_from_id(self, category_id: int) -> str:
        return self._category_id_to_name[category_id]

    def category_ids_from_names(self, category_names: list[str]) -> list[int]:
        return [self._category_name_to_id[category_name] for category_name in category_names]

    def category_names_from_ids(self, category_ids: list[int]) -> list[str]:
        return [self._category_id_to_name[category_id] for category_id in category_ids]
