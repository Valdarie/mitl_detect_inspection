import os
import cv2
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sahi.utils.file import load_json, save_json
from tqdm import tqdm

# Configuration for COCO file and directories
coco_file_name = "cassette_val"

# Determine target split from coco_file_name
target_split = None
if "train" in coco_file_name.lower():
    target_split = "train"
elif "test" in coco_file_name.lower():
    target_split = "test"
elif "val" in coco_file_name.lower():
    target_split = "val"
if not target_split:
    raise ValueError("Unable to determine target split from coco_file_name.")

DATA_DIR = os.path.join(".", "testy")
COCO_DIR = os.path.join(DATA_DIR, "coco")

ORG_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_corrected_coco.json")
SLC_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_sliced_coco.json")

#IMAGE_DIR = os.path.join(COCO_DIR, "images")
IMAGE_DIR = os.path.join(COCO_DIR, "images")
SLICED_IMAGE_DIR = os.path.join(COCO_DIR, "images_sliced",coco_file_name)

BBOX_VISUALIZATION_DIR = os.path.join(DATA_DIR, "bbox_vis", coco_file_name)

os.path.exists(DATA_DIR)
os.path.exists(COCO_DIR)

os.path.exists(ORG_ANNOTATION_PATH)
os.path.exists(IMAGE_DIR)
os.path.exists(BBOX_VISUALIZATION_DIR)


# Check existence of essential paths
assert os.path.exists(DATA_DIR), f"Data directory {DATA_DIR} not found."
assert os.path.exists(COCO_DIR), f"COCO directory {COCO_DIR} not found."
assert os.path.exists(ORG_ANNOTATION_PATH), f"Original annotation file {ORG_ANNOTATION_PATH} not found."
assert os.path.exists(IMAGE_DIR), f"Image directory {IMAGE_DIR} not found."
assert os.path.exists(BBOX_VISUALIZATION_DIR), f"Bounding box visualization directory {BBOX_VISUALIZATION_DIR} not found."

# Load COCO annotations for original and sliced images
coco_dict = load_json(ORG_ANNOTATION_PATH)
slc_dict = load_json(SLC_ANNOTATION_PATH)

# Update file paths in the COCO JSON to use just the filename (without path) for consistency
for img in coco_dict["images"]:
    img["file_name"] = os.path.basename(img["file_name"])
save_json(coco_dict, save_path=ORG_ANNOTATION_PATH)

# Visualize bounding boxes on original images
for img in tqdm(coco_dict["images"], desc="Processing Original Images"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)

    # Load image in grayscale and convert to RGB
    mono_img = Image.open(os.path.join(IMAGE_DIR, img["file_name"])).convert("L")
    rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))

    # Draw bounding boxes
    for annotation in coco_dict["annotations"]:
        if annotation["image_id"] == img["id"]:
            xywh = annotation["bbox"]
            xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
            ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")

    # Display and save image with bounding boxes
    ax.axis("off")
    ax.imshow(rgb_img)
    fig.savefig(os.path.join(BBOX_VISUALIZATION_DIR, img["file_name"][:-4] + ".png"))
    plt.close(fig)

# Visualize bounding boxes on sliced images
for img in tqdm(slc_dict["images"], desc="Processing Sliced Images"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)

    # Load sliced image in grayscale and convert to RGB
    sliced_img_path = os.path.join(SLICED_IMAGE_DIR, img["file_name"])
    mono_img = Image.open(sliced_img_path).convert("L")
    rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))

    # Draw bounding boxes
    for annotation in slc_dict["annotations"]:
        if annotation["image_id"] == img["id"]:
            xywh = annotation["bbox"]
            xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
            ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")

    # Display and save image with bounding boxes
    ax.axis("off")
    ax.imshow(rgb_img)
    fig.savefig(os.path.join(BBOX_VISUALIZATION_DIR, img["file_name"][:-4] + "_sliced.png"))
    plt.close(fig)
