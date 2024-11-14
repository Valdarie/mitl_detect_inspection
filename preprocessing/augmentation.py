import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path
from PIL import Image
from sahi.slicing import slice_coco
from sahi.utils.file import load_json, save_json
from tqdm import tqdm

coco_file_name = "cassette1_train"

original_file = f"{coco_file_name}_corrected_coco.json"
sliced_file = f"{coco_file_name}_sliced_coco.json"

DATA_DIR = os.path.join(".", "data", "coco")
ORG_ANNOTATION_PATH = os.path.join(DATA_DIR, original_file)
SLC_ANNOTATION_PATH  = os.path.join(DATA_DIR, sliced_file)
AUGMENTATION_PATH =  os.path.join("..", "data","augmentation")
IMAGE_DIR = os.path.join(DATA_DIR, "images") 
SLICED_IMAGE_DIR = os.path.join(DATA_DIR, "images_sliced",coco_file_name) 
VISUALIZATION_DIR = os.path.join(".", "data", "bbox_vis")
BBOX_VISUALISATION_DIR = os.path.join

# Ensure directories exist
os.makedirs(AUGMENTATION_PATH, exist_ok=True)

