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


DATA_DIR = os.path.join("..", "testy")
COCO_DIR = os.path.join(DATA_DIR, "coco")

ORG_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_corrected_coco.json")
SLC_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_sliced_coco.json")

AUGMENTATION_PATH = os.path.join(COCO_DIR, "augmentated") # Folder for all augmentations ./data/coco/augmentated


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

os.makedirs(BBOX_SAVE_DIR, exist_ok=True)
os.makedirs(AUGMENTATION_PATH,exist_ok=True)

coco_dict = load_json(ORG_ANNOTATION_PATH)
[img.update({"file_name": img["file_name"].split("/")[-1]}) for img in coco_dict["images"]]
save_json(coco_dict, save_path=ORG_ANNOTATION_PATH)

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
    fig.savefig(os.path.join(BBOX_SAVE_DIR, img["file_name"][:-4] + ".png")) #save each of the image original

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

splits = ["train", "test", "val"]

# Create 'original' and 'sliced' directories within each split
for split in splits:
    base_path = os.path.join(AUGMENTATION_PATH, split)
    os.makedirs(base_path, exist_ok=True)
    
    # Create 'original' and 'sliced' subdirectories within each split
    os.makedirs(os.path.join(base_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "sliced"), exist_ok=True)

def horizontal_flip_with_bboxes(
    image_dir,
    coco_annotations,
    output_dir,
    aug_type="horizontal_flip",
    dataset_type="original",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0
):
    # Set AUG_SAVE_DIR and AUG_VIEW_DIR based on dataset_type (either "original" or "sliced")
    AUG_SAVE_DIR = os.path.join(AUGMENTATION_PATH, target_split, dataset_type, aug_type, "each")
    AUG_VIEW_DIR = os.path.join(AUGMENTATION_PATH, target_split, dataset_type, aug_type)

    os.makedirs(AUG_SAVE_DIR, exist_ok=True)
    os.makedirs(AUG_VIEW_DIR, exist_ok=True)
    
    # Augmentation pipeline with customizable normalization
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    for img in coco_annotations["images"]:
        file_name = img["file_name"]
        image_id = img["id"]

        # Load image and associated bounding boxes and labels
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for albumentations
        
        # Get bounding boxes and labels
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation with normalization
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented["image"]
        augmented_bboxes = augmented["bboxes"]

        # Re-scale normalized image back to [0, 255] for saving in RGB
        augmented_image_rescaled = (augmented_image * np.array(std) + np.array(mean)) * max_pixel_value
        augmented_image_rescaled = np.clip(augmented_image_rescaled, 0, 255).astype("uint8")
        
        # Save the plain (non-bounding-box) version in RGB format
        output_image_path = os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}")
        cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image_rescaled, cv2.COLOR_RGB2BGR))

        # Plot augmented image with bounding boxes for visualization (still in RGB)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(augmented_image_rescaled)  # Display in RGB
        
        # Draw each bounding box
        for bbox in augmented_bboxes:
            x, y, width, height = bbox
            rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

        ax.axis("off")
        # Save the augmented image with bounding boxes to AUG_VIEW_DIR in RGB
        vis_output_path = os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}")
        fig.savefig(vis_output_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        # Update COCO JSON with augmented bounding boxes
        for ann, new_bbox in zip([ann for ann in coco_annotations["annotations"] if ann["image_id"] == image_id], augmented_bboxes):
            ann["bbox"] = new_bbox

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")
    # Run the function for both original and sliced cases
#horizontal_flip_with_bboxes(IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "horizontal_flip"), dataset_type="original")
horizontal_flip_with_bboxes(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "horizontal_flip"), dataset_type="sliced")

