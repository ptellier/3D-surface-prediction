import torch.cuda

TORCH_DEVICE = 'cpu'
if torch.backends.mps.is_available():
    TORCH_DEVICE = 'mps'
if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'


# Datasets
MANUAL_DATASET_FOLDER_PATH = './datasets/manual_dataset'
MANUAL_DATASET_IMAGES_PATH = MANUAL_DATASET_FOLDER_PATH+'/images/'
MANUAL_DATASET_ANNOTATIONS_PATH = MANUAL_DATASET_FOLDER_PATH+'/annotations/surface_annotations.json'
MANUAL_DATASET_INSTANCE_SEGMENTATIONS_PATH = MANUAL_DATASET_FOLDER_PATH+'/annotations/instance_segmentations.json'
MANUAL_DATASET_DT_TO_GT_CATEGORIES = {
    'ingestible_shallow':  ['convex', 'corner', 'flat_edge', 'flat_corner', 'rod_like', 'convex_rim'],
    'ingestible_medium':   [],
    'ingestible_deep':     ['spherical_grasp'],
    'flat':                ['flat', 'bumpy'],
    'other':   ['edge', 'convex_edge']
}
MANUAL_DATASET_GT_CATEGORY_TO_DILATION = {
    'edge': 0,
    'corner': 20,
}
MANUAL_DATASET_GT_CATEGORY_PRIORITIES = [
    'corner', 'flat_corner', 'spherical_grasp', 'rod_like', 'convex_rim', 'flat_edge', 'edge', 'convex_edge',
    'convex', 'flat', 'bumpy'
]

DUMMY_DETECTED_DATASET_ANNOTATIONS_PATH = './datasets/dummy_dt_dataset.json'
