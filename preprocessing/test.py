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

DATA_DIR = os.path.join(".", "testy") #. for script, .. for notebook
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
    # Set directories based on dataset_type
    AUG_SAVE_DIR = os.path.join(AUGMENTATION_PATH, target_split, dataset_type, aug_type, "each")
    AUG_VIEW_DIR = os.path.join(AUGMENTATION_PATH, target_split, dataset_type, aug_type)

    os.makedirs(AUG_SAVE_DIR, exist_ok=True)
    os.makedirs(AUG_VIEW_DIR, exist_ok=True)
    
    # Augmentation pipeline with normalization
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    # Use tqdm to show progress
    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"]
        image_id = img["id"]

        # Load image as grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")

        # Convert grayscale image to numpy array for augmentation
        image_np = np.array(mono_img)

        # Get bounding boxes and labels
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation with normalization, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale and save the base monochrome image without bounding boxes
        augmented_image = (augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value
        augmented_image = np.clip(augmented_image, 0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Create an RGB image from the grayscale version to draw colored bounding boxes
        rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))
        draw = ImageDraw.Draw(rgb_img)
        
        # Draw each bounding box on the RGB image
        for bbox in augmented["bboxes"]:
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
        
        # Save the image with colored bounding boxes
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")

def run():
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

    # Run the function for both original and sliced cases
    # horizontal_flip_with_bboxes(IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "horizontal_flip"), dataset_type="original")
    horizontal_flip_with_bboxes(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "horizontal_flip"), dataset_type="sliced")


if __name__ == "__main__":
    run()
