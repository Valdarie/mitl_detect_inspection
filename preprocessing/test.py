import os.path
import cv2
import json
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

import albumentations as A 
from sahi.utils.file import load_json, save_json
from tqdm import tqdm
coco_file_name  = "cassette_val"

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

print(target_split)

DATA_DIR = os.path.join("..", "testy")
COCO_DIR = os.path.join(DATA_DIR, "coco")

ORG_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_corrected_coco.json")
SLC_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_sliced_coco.json")
AUGMENTATION_PATH = os.path.join(COCO_DIR, "augmentated") # Folder for all augmentations ./data/coco/augmentated

print(ORG_ANNOTATION_PATH)

#IMAGE_DIR = os.path.join(COCO_DIR, "images")
IMAGE_DIR = os.path.join(COCO_DIR, "images")
SLICED_IMAGE_DIR = os.path.join(COCO_DIR, "images_sliced",coco_file_name)

BBOX_VISUALIZATION_DIR = os.path.join(DATA_DIR, "bbox_vis", coco_file_name)
BBOX_SAVE_DIR = os.path.join(BBOX_VISUALIZATION_DIR,"each") #Place to savea each of the before augmented bounding boxes

os.path.exists(DATA_DIR)
os.path.exists(COCO_DIR)

os.path.exists(ORG_ANNOTATION_PATH)
os.path.exists(IMAGE_DIR)
os.path.exists(BBOX_VISUALIZATION_DIR)

if not os.path.exists(BBOX_SAVE_DIR):
    os.makedirs(BBOX_SAVE_DIR)

if not os.path.exists(AUGMENTATION_PATH):
    os.makedirs(AUGMENTATION_PATH)

splits = ["train", "test", "val"]

# Create 'original' and 'sliced' directories within each split
for split in splits:
    base_path = os.path.join(AUGMENTATION_PATH, split)
    os.makedirs(base_path, exist_ok=True)
    
    # Create 'original' and 'sliced' subdirectories within each split
    os.makedirs(os.path.join(base_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "sliced"), exist_ok=True)

coco_dict = load_json(ORG_ANNOTATION_PATH)
[img.update({"file_name": img["file_name"].split("/")[-1]}) for img in coco_dict["images"]]
save_json(coco_dict, save_path=ORG_ANNOTATION_PATH)

coco_dict

for img in coco_dict["images"]:
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)

    mono_img = Image.open(os.path.join(IMAGE_DIR,img["file_name"])).convert("L")
    rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))

    # iterate over all annotations
    for ann_ind in range(len(coco_dict["annotations"])):
        
        if coco_dict["annotations"][ann_ind]["image_id"] == img["id"]:
            # convert coco bbox to pil bbox
            xywh = coco_dict["annotations"][ann_ind]["bbox"]
            xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]

            # visualize bbox over image
            ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")

    ax.axis("off")
    ax.imshow(rgb_img)

    slc_dict= load_json(SLC_ANNOTATION_PATH)

    for img in slc_dict["images"]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
        
        # Open the sliced image file in grayscale, convert it to RGB
        sliced_img_path = os.path.join(SLICED_IMAGE_DIR, img["file_name"])
        mono_img = Image.open(sliced_img_path).convert("L")
        rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))
        
        # Iterate over all annotations for this specific image
        for annotation in slc_dict["annotations"]:
            if annotation["image_id"] == img["id"]:
                # Extract and convert bounding box coordinates
                xywh = annotation["bbox"]
                xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
                
                # Draw the bounding box on the image
                ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")
        
        # Display and save the image with bounding boxes
        ax.axis("off")
        ax.imshow(rgb_img)
        fig.savefig(os.path.join(BBOX_SAVE_DIR, img["file_name"][:-4] + ".png")) #save each of the image to folder called Each
        plt.close()

def horizontal_flip(image_dir, coco_annotations, output_dir, aug_type="horizontal_flip"):
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=1.0)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))
    
    # Process each image in the COCO dataset
    for img in coco_annotations["images"]:
        file_name = img["file_name"]
        image_id = img["id"]

        # Full path to the image
        image_path = os.path.join(image_dir, file_name)
        
        # Load image and associated bounding boxes and labels
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented["image"]
        augmented_bboxes = augmented["bboxes"]

        # Convert back to BGR for saving
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        # Save the augmented image
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, f"{aug_type}_{file_name}")
        cv2.imwrite(output_image_path, augmented_image)

        # Update COCO JSON with augmented bounding boxes
        for ann, new_bbox in zip([ann for ann in coco_annotations["annotations"] if ann["image_id"] == image_id], augmented_bboxes):
            ann["bbox"] = new_bbox

    # Save the updated annotations with the augmentation type in the filename
    augmented_json_path = os.path.join(output_dir, f"augmented_annotations_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")
    

horizontal_flip(SLICED_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "horizontal_flip"))
horizontal_flip(SLICED_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "horizontal_flip"))