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

coco_file_name = "cassette1_val"

original_file = f"{coco_file_name}_corrected_coco.json"
sliced_file = f"{coco_file_name}_sliced_coco.json"

DATA_COCO_DIR = os.path.join("...", "data", "coco")
DATA_DIR = os.path.join("...","data")

ORG_ANNOTATION_PATH = os.path.join(DATA_COCO_DIR, original_file)
SLC_ANNOTATION_PATH  = os.path.join(DATA_COCO_DIR, sliced_file)
AUGMENTATION_PATH =  os.path.join("...", "data","augmentation")
IMAGE_DIR = os.path.join(DATA_COCO_DIR, "images") 
SLICED_IMAGE_DIR = os.path.join(DATA_COCO_DIR, "images_sliced",coco_file_name) 
VISUALISATION_DIR = os.path.join("...", "data", "bbox_vis")
VISUALISATION_FILE = os.path.join(VISUALISATION_DIR, coco_file_name)

os.path.exists(DATA_DIR)
os.path.exists(AUGMENTATION_PATH)
os.path.exists(IMAGE_DIR)
os.path.exists(VISUALISATION_DIR)
os.path.exists(VISUALISATION_FILE)




