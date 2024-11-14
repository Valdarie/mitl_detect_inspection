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

split = "train"  # or "val" as required
coco_file_name = "cassette1_" + split
original_file = f"{coco_file_name}_corrected_coco.json"
sliced_file = f"{coco_file_name}_sliced_coco.json"

# Path setup
DATA_DIR = os.path.join("..", "data")
DATA_COCO_DIR = os.path.join(DATA_DIR, "coco")

AUGMENTATION_PATH = os.path.join(DATA_DIR, "augmentation")
IMAGE_DIR = os.path.join(DATA_COCO_DIR, "images")
SLICED_IMAGE_DIR = os.path.join(DATA_COCO_DIR, "images_sliced", coco_file_name)
VISUALISATION_DIR = os.path.join(DATA_DIR, "bbox_vis")
VISUALISATION_FILE = os.path.join(VISUALISATION_DIR, coco_file_name)

# Update paths for original and sliced annotations based on split
ORG_ANNOTATION_PATH = os.path.join(DATA_COCO_DIR, split, original_file)
SLC_ANNOTATION_PATH = os.path.join(DATA_COCO_DIR, split, sliced_file)

os.path.exists(DATA_DIR)
os.path.exists(ORG_ANNOTATION_PATH)

class OriginalImageAugmentor:
    def __init__(self, coco_file_name, augmentation_config):
        # Define directory paths based on specified structure
        DATA_DIR = os.path.join("..", "data", "coco")
        self.org_annotation_path = os.path.join(DATA_DIR, f"{coco_file_name}_corrected_coco.json")
        self.image_dir = os.path.join(DATA_DIR, "images")
        
        AUGMENTATION_PATH = os.path.join("..", "data", "augmentation")
        VISUALIZATION_DIR = os.path.join("..", "data", "bbox_vis")
        
        self.output_dir = os.path.join(AUGMENTATION_PATH, "original_images")
        self.bbox_visualization_dir = os.path.join(VISUALIZATION_DIR, coco_file_name)
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.bbox_visualization_dir, exist_ok=True)
        
        # Define augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline(**augmentation_config)
        
        # Load COCO annotations
        self.coco_dict = load_json(self.org_annotation_path)
    
    def _create_augmentation_pipeline(self, horizontal_flip, brightness_contrast, rotate, blur, normalize):
        # Define an augmentation pipeline based on provided configuration
        return A.Compose([
            A.HorizontalFlip(p=horizontal_flip),
            A.RandomBrightnessContrast(p=brightness_contrast),
            A.ShiftScaleRotate(rotate_limit=rotate, p=0.5),
            A.Blur(blur_limit=blur, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def augment_image(self, image, bboxes, class_labels):
        # Apply augmentation to an image
        return self.augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

    def process_images(self):
        # Build a dictionary to quickly access annotations by image_id
        annotations_by_image = {}
        for annotation in self.coco_dict["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)

        # Process each image in the COCO JSON file
        for img in self.coco_dict["images"]:
            image_id = img["id"]
            image_path = os.path.join(self.image_dir, img["file_name"])
            
            # Check if the image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {img['file_name']} not found.")
                continue
            
            # Load and prepare the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get bounding boxes and labels for this image
            bboxes, class_labels = [], []
            if image_id in annotations_by_image:
                for annotation in annotations_by_image[image_id]:
                    x, y, w, h = annotation["bbox"]
                    bboxes.append([x, y, w, h])
                    class_labels.append(annotation["category_id"])

            # Apply augmentation
            augmented = self.augment_image(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']

            # Save augmented image and visualize bounding boxes
            self.save_augmented_image(augmented_image, img["file_name"])
            self.visualize_bboxes(augmented_image, augmented_bboxes, img["file_name"])

    def save_augmented_image(self, image, filename):
        # Save the augmented image
        output_path = os.path.join(self.output_dir, filename)
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC format
        image = (image * 255).astype('uint8')
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def visualize_bboxes(self, image, bboxes, filename):
        # Visualize and save bounding boxes on the augmented image
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))  # CHW to HWC for plotting

        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Save visualization
        output_path = os.path.join(self.bbox_visualization_dir, f"{filename[:-4]}_augmented.png")
        plt.savefig(output_path)
        plt.close(fig)

augmentation_config = {
    'horizontal_flip': 0.5,
    'brightness_contrast': 0.5,
    'rotate': 30,
    'blur': 3,
    'normalize': True
}

coco_file_name = "cassette1_train"
augmentor = OriginalImageAugmentor(coco_file_name=coco_file_name, augmentation_config=augmentation_config)
augmentor.process_images()