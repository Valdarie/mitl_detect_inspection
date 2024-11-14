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
import random

import matplotlib.pyplot as plt

import albumentations as A 
from sahi.utils.file import load_json, save_json
from tqdm import tqdm

coco_file_name  = "cassette1_val"

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

DATA_DIR = os.path.join("..", "data")
COCO_DIR = os.path.join(DATA_DIR, "coco")

ORG_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_corrected_coco.json")
SLC_ANNOTATION_PATH = os.path.join(DATA_DIR, "coco",target_split, f"{coco_file_name}_sliced_coco.json")

AUGMENTATION_PATH = os.path.join(COCO_DIR, "augmentated") # Folder for all augmentations ./data/coco/augmentated

print(ORG_ANNOTATION_PATH)

#IMAGE_DIR = os.path.join(COCO_DIR, "images")
IMAGE_DIR = os.path.join(COCO_DIR, "images")
NEW_IMAGE_DIR = os.path.join(COCO_DIR, "images", "png") #for saving new original images (untouched)
SLICED_IMAGE_DIR = os.path.join(COCO_DIR, "images_sliced",coco_file_name)

BBOX_VISUALIZATION_DIR = os.path.join(DATA_DIR, "bbox_vis", coco_file_name)
BBOX_SAVE_DIR = os.path.join(BBOX_VISUALIZATION_DIR,"each") #Place to savea each of the before augmented bounding boxes

os.path.exists(DATA_DIR)
os.path.exists(COCO_DIR)

os.path.exists(ORG_ANNOTATION_PATH)
os.path.exists(IMAGE_DIR)
os.path.exists(BBOX_VISUALIZATION_DIR)

os.makedirs(BBOX_SAVE_DIR, exist_ok=True)
os.makedirs(NEW_IMAGE_DIR, exist_ok=True)
os.makedirs(AUGMENTATION_PATH,exist_ok=True)

def horizontal_flip(
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
        A.HorizontalFlip(p=1.0), # P is probability
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")  # Replace .bmp with .png
        image_id = img["id"]

        # Load and convert the image to grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")
        image_np = np.array(mono_img)

        # Collect bounding boxes and class labels for the image
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale augmented image back to 0–255 for viewing and saving
        augmented_image = ((augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value).clip(0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Convert the augmented image to RGB for bounding box visualization
        rgb_img = Image.merge("RGB", (Image.fromarray(augmented_image),) * 3)
        draw = ImageDraw.Draw(rgb_img)
        
        # Draw bounding boxes on the RGB image
        for bbox in augmented["bboxes"]:
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
        
        # Save the image with bounding boxes drawn
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")


def vertical_flip_with_bboxes(
    image_dir,
    coco_annotations,
    output_dir,
    aug_type="vertical_flip",
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
    
    # Augmentation pipeline with vertical flip and normalization
    augmentation_pipeline = A.Compose([
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")  # Replace .bmp with .png
        image_id = img["id"]

        # Load and convert the image to grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")
        image_np = np.array(mono_img)

        # Collect bounding boxes and class labels for the image
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale augmented image back to 0–255 for viewing and saving
        augmented_image = ((augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value).clip(0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Convert the augmented image to RGB for bounding box visualization
        rgb_img = Image.merge("RGB", (Image.fromarray(augmented_image),) * 3)
        draw = ImageDraw.Draw(rgb_img)
        
        # Draw bounding boxes on the RGB image
        for bbox in augmented["bboxes"]:
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
        
        # Save the image with bounding boxes drawn
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)

def random_flip(
    image_dir,
    coco_annotations,
    output_dir,
    dataset_type="original",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0
):
    # Set a fixed aug_type to ensure output is saved in the "random_flip" folder
    aug_type = "random_flip"

    # Randomly choose between horizontal and vertical flip
    if random.choice([True, False]):
        print("Applying horizontal flip")
        horizontal_flip(
            image_dir=image_dir,
            coco_annotations=coco_annotations,
            output_dir=output_dir,
            aug_type=aug_type,  # Use "random_flip" as aug_type
            dataset_type=dataset_type,
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value
        )
    else:
        print("Applying vertical flip")
        vertical_flip(
            image_dir=image_dir,
            coco_annotations=coco_annotations,
            output_dir=output_dir,
            aug_type=aug_type,  # Use "random_flip" as aug_type
            dataset_type=dataset_type,
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value
        )

def safe_rotate(
    image_dir,
    coco_annotations,
    output_dir,
    limit=(-90, 90),  # Range for random rotation angle
    interpolation=1,  # Default is cv2.INTER_LINEAR
    border_mode=4,  # Default is cv2.BORDER_REFLECT_101
    rotate_method="largest_box",  # How to handle bounding boxes
    aug_type="safe_rotate",
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
    
    # Augmentation pipeline with SafeRotate and normalization
    augmentation_pipeline = A.Compose([
        A.SafeRotate(
            limit=limit,  # Random rotation within specified range
            interpolation=interpolation,
            border_mode=border_mode,
            rotate_method=rotate_method,
            p=1.0
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    # Initialize a new list to store augmented annotations
    new_annotations = []

    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")  # Replace .bmp with .png
        image_id = img["id"]

        # Load and convert the image to grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")
        image_np = np.array(mono_img)

        # Collect bounding boxes and class labels for the image
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale augmented image back to 0–255 for viewing and saving
        augmented_image = ((augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value).clip(0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Convert the augmented image to RGB for bounding box visualization
        rgb_img = Image.merge("RGB", (Image.fromarray(augmented_image),) * 3)
        draw = ImageDraw.Draw(rgb_img)

        # Update new annotations with augmented bounding boxes directly
        for bbox, label in zip(augmented["bboxes"], class_labels):
            # Visualize the bounding box
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
            
            # Save the updated annotation
            new_annotations.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": len(new_annotations) + 1  # Unique ID for each annotation
            })

        # Save the image with bounding boxes drawn
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Replace the original annotations with the new, augmented ones
    coco_annotations["annotations"] = new_annotations

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")


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
import random

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
NEW_IMAGE_DIR = os.path.join(COCO_DIR, "images", "png") #for saving new original images (untouched)
SLICED_IMAGE_DIR = os.path.join(COCO_DIR, "images_sliced",coco_file_name)

BBOX_VISUALIZATION_DIR = os.path.join(DATA_DIR, "bbox_vis", coco_file_name)
BBOX_SAVE_DIR = os.path.join(BBOX_VISUALIZATION_DIR,"each") #Place to savea each of the before augmented bounding boxes

os.path.exists(DATA_DIR)
os.path.exists(COCO_DIR)

os.path.exists(ORG_ANNOTATION_PATH)
os.path.exists(IMAGE_DIR)
os.path.exists(BBOX_VISUALIZATION_DIR)

os.makedirs(BBOX_SAVE_DIR, exist_ok=True)
os.makedirs(NEW_IMAGE_DIR, exist_ok=True)
os.makedirs(AUGMENTATION_PATH,exist_ok=True)
coco_dict = load_json(ORG_ANNOTATION_PATH)
[img.update({"file_name": img["file_name"].split("/")[-1]}) for img in coco_dict["images"]]
save_json(coco_dict, save_path=ORG_ANNOTATION_PATH)

coco_dict
# Process images with a progress bar
for img in tqdm(coco_dict["images"], desc="Processing original images"):
    # Open and convert the image to grayscale
    mono_img = Image.open(os.path.join(IMAGE_DIR, img["file_name"])).convert("L")
    
    # Save the grayscale image as .png
    png_file_name = img["file_name"].replace(".bmp", ".png")
    mono_img.save(os.path.join(NEW_IMAGE_DIR, png_file_name), format="PNG")
    
    # Convert grayscale image to RGB for bounding box visualization
    rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))

    # Visualize bounding boxes on the RGB image
    for ann in coco_dict["annotations"]:
        if ann["image_id"] == img["id"]:
            xywh = ann["bbox"]
            xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
            ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")

    # Display and save the image with bounding boxes as .png
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
    ax.axis("off")
    ax.imshow(rgb_img)
    fig.savefig(os.path.join(BBOX_SAVE_DIR, png_file_name))
slc_dict= load_json(SLC_ANNOTATION_PATH)
print(SLC_ANNOTATION_PATH)

slc_dict

for img in tqdm(slc_dict["images"], desc="Processing sliced images"):
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
    fig.savefig(os.path.join(BBOX_SAVE_DIR, img["file_name"][:-4] + ".png"))  # Save each image to the specified folder
splits = ["train", "test", "val"]

# Create 'original' and 'sliced' directories within each split
for split in splits:
    base_path = os.path.join(AUGMENTATION_PATH, split)
    os.makedirs(base_path, exist_ok=True)
    
    # Create 'original' and 'sliced' subdirectories within each split
    os.makedirs(os.path.join(base_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "sliced"), exist_ok=True)
## Spatial AUGMENTATION
### horizontal flip
def horizontal_flip(
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
        A.HorizontalFlip(p=1.0), # P is probability
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")  # Replace .bmp with .png
        image_id = img["id"]

        # Load and convert the image to grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")
        image_np = np.array(mono_img)

        # Collect bounding boxes and class labels for the image
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale augmented image back to 0–255 for viewing and saving
        augmented_image = ((augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value).clip(0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Convert the augmented image to RGB for bounding box visualization
        rgb_img = Image.merge("RGB", (Image.fromarray(augmented_image),) * 3)
        draw = ImageDraw.Draw(rgb_img)
        
        # Draw bounding boxes on the RGB image
        for bbox in augmented["bboxes"]:
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
        
        # Save the image with bounding boxes drawn
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")

# Run the function for both original and sliced cases
horizontal_flip(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "horizontal_flip"), dataset_type="original")
horizontal_flip(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "horizontal_flip"), dataset_type="sliced")
### vertical flip
def vertical_flip(
    image_dir,
    coco_annotations,
    output_dir,
    aug_type="vertical_flip",
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
    
    # Augmentation pipeline with vertical flip and normalization
    augmentation_pipeline = A.Compose([
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")  # Replace .bmp with .png
        image_id = img["id"]

        # Load and convert the image to grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")
        image_np = np.array(mono_img)

        # Collect bounding boxes and class labels for the image
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale augmented image back to 0–255 for viewing and saving
        augmented_image = ((augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value).clip(0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Convert the augmented image to RGB for bounding box visualization
        rgb_img = Image.merge("RGB", (Image.fromarray(augmented_image),) * 3)
        draw = ImageDraw.Draw(rgb_img)
        
        # Draw bounding boxes on the RGB image
        for bbox in augmented["bboxes"]:
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
        
        # Save the image with bounding boxes drawn
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")

# Run the function for both original and sliced cases
vertical_flip(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "vertical_flip"), dataset_type="original")
vertical_flip(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "vertical_flip"), dataset_type="sliced")
### Random Flip

def random_flip(
    image_dir,
    coco_annotations,
    output_dir,
    dataset_type="original",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0
):
    # Set a fixed aug_type to ensure output is saved in the "random_flip" folder
    aug_type = "random_flip"

    # Randomly choose between horizontal and vertical flip
    if random.choice([True, False]):
        print("Applying horizontal flip")
        horizontal_flip(
            image_dir=image_dir,
            coco_annotations=coco_annotations,
            output_dir=output_dir,
            aug_type=aug_type,  # Use "random_flip" as aug_type
            dataset_type=dataset_type,
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value
        )
    else:
        print("Applying vertical flip")
        vertical_flip(
            image_dir=image_dir,
            coco_annotations=coco_annotations,
            output_dir=output_dir,
            aug_type=aug_type,  # Use "random_flip" as aug_type
            dataset_type=dataset_type,
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value
        )

# Run the random_flip function for both original and sliced cases
random_flip(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "random_flip"), dataset_type="original")
random_flip(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "random_flip"), dataset_type="sliced")

### Safe Rotate
def safe_rotate(
    image_dir,
    coco_annotations,
    output_dir,
    limit=(-90, 90),  # Range for random rotation angle
    interpolation=1,  # Default is cv2.INTER_LINEAR
    border_mode=4,  # Default is cv2.BORDER_REFLECT_101
    rotate_method="largest_box",  # How to handle bounding boxes
    aug_type="safe_rotate",
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
    
    # Augmentation pipeline with SafeRotate and normalization
    augmentation_pipeline = A.Compose([
        A.SafeRotate(
            limit=limit,  # Random rotation within specified range
            interpolation=interpolation,
            border_mode=border_mode,
            rotate_method=rotate_method,
            p=1.0
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    # Initialize a new list to store augmented annotations
    new_annotations = []

    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")  # Replace .bmp with .png
        image_id = img["id"]

        # Load and convert the image to grayscale
        image_path = os.path.join(image_dir, file_name)
        mono_img = Image.open(image_path).convert("L")
        image_np = np.array(mono_img)

        # Collect bounding boxes and class labels for the image
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply augmentation, including class_labels
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, class_labels=class_labels)
        
        # Rescale augmented image back to 0–255 for viewing and saving
        augmented_image = ((augmented["image"] * np.array(std[0]) + mean[0]) * max_pixel_value).clip(0, 255).astype("uint8")
        Image.fromarray(augmented_image).save(os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}"))

        # Convert the augmented image to RGB for bounding box visualization
        rgb_img = Image.merge("RGB", (Image.fromarray(augmented_image),) * 3)
        draw = ImageDraw.Draw(rgb_img)

        # Update new annotations with augmented bounding boxes directly
        for bbox, label in zip(augmented["bboxes"], class_labels):
            # Visualize the bounding box
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw.rectangle(xyxy, outline="lime", width=5)
            
            # Save the updated annotation
            new_annotations.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": len(new_annotations) + 1  # Unique ID for each annotation
            })

        # Save the image with bounding boxes drawn
        rgb_img.save(os.path.join(AUG_VIEW_DIR, f"{aug_type}_{file_name}"))

    # Replace the original annotations with the new, augmented ones
    coco_annotations["annotations"] = new_annotations

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")

# Run the safe_rotate function for both original and sliced cases
safe_rotate(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "safe_rotate"), dataset_type="original")
safe_rotate(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "safe_rotate"), dataset_type="sliced")
### Optical Distortion
def optical_distortion(
    image_dir,
    coco_annotations,
    output_dir,
    dataset_type="original",
    distort_limit=0.5,
    shift_limit=0.05,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0
):
    # Set a fixed aug_type to ensure output is saved in the "optical_distortion" folder
    aug_type = "optical_distortion"

    # Create directories for saving augmented images and annotations
    AUG_SAVE_DIR = os.path.join(output_dir, dataset_type, aug_type, "each")
    os.makedirs(AUG_SAVE_DIR, exist_ok=True)

    # Define the augmentation pipeline with OpticalDistortion and normalization
    transform = A.Compose([
        A.OpticalDistortion(
            distort_limit=distort_limit,
            shift_limit=shift_limit,
            interpolation=interpolation,
            border_mode=border_mode,
            p=1.0
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    # Process each image in coco_annotations
    new_annotations = []
    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")
        image_id = img["id"]

        # Load the image
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)

        # Collect bounding boxes and class labels
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply the transformation
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        # Save the augmented image
        augmented_image = (transformed["image"] * max_pixel_value).clip(0, 255).astype("uint8")
        augmented_image_path = os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}")
        cv2.imwrite(augmented_image_path, augmented_image)

        # Update the new annotations with transformed bounding boxes
        for bbox, label in zip(transformed["bboxes"], class_labels):
            new_annotations.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": len(new_annotations) + 1
            })

    # Update the COCO annotations with augmented bounding boxes
    coco_annotations["annotations"] = new_annotations

    # Save the updated COCO JSON annotations
    augmented_json_path = os.path.join(output_dir, f"{coco_file_name}_{aug_type}.json")
    save_json(coco_annotations, augmented_json_path)
    print(f"Augmented annotations saved to {augmented_json_path}")

def optical_distortion(
    image_dir,
    coco_annotations,
    output_dir,
    dataset_type="original",
    distort_limit=0.5,
    shift_limit=0.05,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0
):
    # Set a fixed aug_type to ensure output is saved in the "optical_distortion" folder
    aug_type = "optical_distortion"

    # Create directories for saving augmented images and annotations
    AUG_SAVE_DIR = os.path.join(output_dir, dataset_type, aug_type, "each")
    os.makedirs(AUG_SAVE_DIR, exist_ok=True)

    # Define the augmentation pipeline with OpticalDistortion and normalization
    transform = A.Compose([
        A.OpticalDistortion(
            distort_limit=distort_limit,
            shift_limit=shift_limit,
            interpolation=interpolation,
            border_mode=border_mode,
            p=1.0
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    # Process each image in coco_annotations
    new_annotations = []
    for img in tqdm(coco_annotations["images"], desc=f"Processing {dataset_type} images"):
        file_name = img["file_name"].replace(".bmp", ".png")
        image_id = img["id"]

        # Load the image
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)

        # Collect bounding boxes and class labels
        bboxes = [ann["bbox"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
        class_labels = [ann["category_id"] for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]

        # Apply the transformation
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        # Save the augmented image
        augmented_image = (transformed["image"] * max_pixel_value).clip(0, 255).astype("uint8")
        augmented_image_path = os.path.join(AUG_SAVE_DIR, f"{aug_type}_{file_name}")
        cv2.imwrite(augmented_image_path, augmented_image)

        # Update the new annotations with transformed bounding boxes
        for bbox, label in zip(transformed["bboxes"], class_labels):
            new_annotations.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": len(new_annotations) + 1
            })

    # Update the COCO annotations with augmented bounding boxes
    coco_annotations["annotations"] = new_annotations

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
        plt.close(fig)

    slc_dict= load_json(SLC_ANNOTATION_PATH)

    for img in tqdm(slc_dict["images"], desc="Processing sliced images"):
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
        fig.savefig(os.path.join(BBOX_SAVE_DIR, img["file_name"][:-4] + ".png"))  # Save each image to the specified folder #save each of the image to folder called Each
        plt.close(fig)

    splits = ["train", "test", "val"]

    # Create 'original' and 'sliced' directories within each split
    for split in splits:
        base_path = os.path.join(AUGMENTATION_PATH, split)
        os.makedirs(base_path, exist_ok=True)
        
        # Create 'original' and 'sliced' subdirectories within each split
        os.makedirs(os.path.join(base_path, "original"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "sliced"), exist_ok=True)


    '''
    AUGMENTATION begins
    '''
    horizontal_flip(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "horizontal_flip"), dataset_type="original")
    horizontal_flip(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "horizontal_flip"), dataset_type="sliced")
    # Run the function for both original and sliced cases
    vertical_flip_with_bboxes(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "vertical_flip"), dataset_type="original")
    vertical_flip_with_bboxes(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "vertical_flip"), dataset_type="sliced")

    safe_rotate(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "safe_rotate"), dataset_type="original")
    safe_rotate(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "safe_rotate"), dataset_type="sliced")

    horizontal_flip(NEW_IMAGE_DIR, coco_dict, os.path.join(AUGMENTATION_PATH, target_split, "original", "horizontal_flip"), dataset_type="original")
    horizontal_flip(SLICED_IMAGE_DIR, slc_dict, os.path.join(AUGMENTATION_PATH, target_split, "sliced", "horizontal_flip"), dataset_type="sliced")

    


if __name__ == "__main__":
    run()
