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

# Define paths
DATA_DIR = os.path.join(".", "data")
IMAGE_DIR = os.path.join(DATA_DIR, "cassette1_bboxes_vis")
IMAGE_PATH = os.path.join(IMAGE_DIR, "01BE01.png")
AUGMENTATION_PATH = os.path.join(DATA_DIR, "coco_json_files/coco_sliced_coco.json")
SLICED_COCOS_DIR = os.path.join(DATA_DIR, "coco_json_files", "sliced")
AUGMENTED_IMAGES_DIR = os.path.join(DATA_DIR, "bbox_visualization")

# Ensure directories exist
os.makedirs(SLICED_COCOS_DIR, exist_ok=True)
os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

# Load the COCO JSON file
coco_dict = load_json(AUGMENTATION_PATH)

# Tile the images and save the new COCO annotations
sliced_coco_path = os.path.join(SLICED_COCOS_DIR, "cassette1_train_sliced.json")
slice_coco(
    coco_annotation_file_path=AUGMENTATION_PATH,
    image_dir=IMAGE_DIR,
    output_coco_annotation_file_name=sliced_coco_path,
    output_dir=IMAGE_DIR,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    ignore_negative_samples=False
)

# Load the sliced COCO JSON
sliced_coco_dict = load_json(sliced_coco_path)

# Augment and visualize each sliced image with bounding boxes
for img in tqdm(sliced_coco_dict["images"], desc="Augmenting sliced images"):
    sliced_image_path = os.path.join(IMAGE_DIR, img["file_name"])

    # Check if image file exists
    if not os.path.exists(sliced_image_path):
        print(f"Warning: Image {img['file_name']} not found in directory.")
        continue

    # Load the sliced image
    image = cv2.imread(sliced_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Get bounding boxes and class labels for this sliced image
    bboxes = []
    class_labels = []
    for annotation in sliced_coco_dict["annotations"]:
        if annotation["image_id"] == img["id"]:
            x, y, w, h = annotation["bbox"]
            bboxes.append([x, y, w, h])
            class_labels.append(annotation["category_id"])

    # Apply augmentation
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']

    # Display the augmented image with bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(augmented_image.permute(1, 2, 0))  # CHW to HWC for plotting

    # Plot bounding boxes
    for bbox in augmented_bboxes:
        x_min, y_min, width, height = bbox
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Save visualization of augmented image with bounding boxes
    augmented_image_path = os.path.join(AUGMENTED_IMAGES_DIR, f"{img['file_name'][:-4]}_augmented.png")
    plt.savefig(augmented_image_path)
    plt.close(fig)

print("Augmentation and visualization for tiled images completed.")
