'''
Mainly for COCO files. Not catered for YOLO
'''

import os
import cv2
import json
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from sahi.utils.file import load_json
from PIL import Image, ImageDraw

# Define the main COCO file name
coco_file_name = "cassette1_train"

class ImageAugmentor:
    def __init__(self, coco_file_name, config, split="train", image_type="original"):
        self.split = split
        self.coco_file_name = coco_file_name
        self.config = config
        self.image_type = image_type

        self.DATA_DIR = config.get("DATA_DIR", "./data")
        self.AUGMENTATION_PATH = config.get("AUGMENTATION_PATH", os.path.join(self.DATA_DIR, "augmentation"))
        self.VISUALIZATION_DIR = config.get("VISUALIZATION_DIR", os.path.join(self.DATA_DIR, "bbox_vis"))

        image_subdir = "original_images" if self.image_type == "original" else "sliced_images"
        self.output_dir = os.path.join(self.AUGMENTATION_PATH, image_subdir, f"{split}_images")
        self.bbox_vis_dir = os.path.join(self.VISUALIZATION_DIR, coco_file_name, image_subdir)

        self.org_annotation_path = os.path.join(self.DATA_DIR, "coco", split, f"{coco_file_name}_corrected_coco.json")
        self.image_dir = os.path.join(self.DATA_DIR, "coco", "images_sliced" if self.image_type == "sliced" else "images")

        self._create_directories([self.output_dir, self.bbox_vis_dir])

        # Load COCO annotations and initialize augmented annotations structure
        self.coco_dict = load_json(self.org_annotation_path)
        self.augmented_annotations = {
            "images": [],
            "annotations": [],
            "categories": self.coco_dict["categories"]
        }
        self.annotation_id = 1  # Initialize annotation ID counter

    def _create_directories(self, directories):
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def process_images(self):
        annotations_by_image = self._organize_annotations_by_image_id()
        
        for img in tqdm(self.coco_dict["images"], desc="Processing Images"):
            image_path = os.path.join(self.image_dir, img["file_name"])
            if not os.path.exists(image_path):
                print(f"Warning: Image {img['file_name']} not found.")
                continue
            
            image = self._load_image(image_path)
            bboxes, class_labels = self._extract_bboxes_and_labels(img["id"], annotations_by_image)
            
            # Apply and save each augmentation type separately
            self._apply_and_save(image, bboxes, class_labels, img["file_name"], "flip")
            self._apply_and_save(image, bboxes, class_labels, img["file_name"], "contrast")
            self._apply_and_save(image, bboxes, class_labels, img["file_name"], "noise")

    def _apply_and_save(self, image, bboxes, class_labels, filename, aug_type):
        if aug_type == "flip":
            augmentation = A.Compose([A.HorizontalFlip(p=1)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        elif aug_type == "contrast":
            augmentation = A.Compose([A.RandomBrightnessContrast(p=1)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        elif aug_type == "noise":
            augmentation = A.Compose([A.GaussNoise(var_limit=(10, 50), p=1)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']

        # Save augmented image and bbox visualization
        save_filename = f"{filename[:-4]}_{aug_type}.png"
        self._save_augmented_image(augmented_image, save_filename)
        self._visualize_augmented_with_bboxes(augmented_image, augmented_bboxes, save_filename, aug_type)

        # Save augmented annotation
        self._save_augmented_annotation(save_filename, augmented_bboxes, class_labels, img_width=augmented_image.shape[1], img_height=augmented_image.shape[0])

    def _save_augmented_annotation(self, filename, bboxes, class_labels, img_width, img_height):
        image_entry = {
            "id": len(self.augmented_annotations["images"]) + 1,
            "file_name": filename,
            "width": img_width,
            "height": img_height
        }
        self.augmented_annotations["images"].append(image_entry)

        for bbox, label in zip(bboxes, class_labels):
            x, y, w, h = bbox
            annotation_entry = {
                "id": self.annotation_id,
                "image_id": image_entry["id"],
                "category_id": label,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            self.augmented_annotations["annotations"].append(annotation_entry)
            self.annotation_id += 1

    def save_augmented_coco_json(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.augmented_annotations, f)

            

    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def _save_augmented_image(self, image, filename):
        output_path = os.path.join(self.output_dir, filename)
        image = np.transpose(image, (1, 0, 2)) if image.ndim == 3 else image
        image = (image * 255).astype('uint8')
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def _visualize_augmented_with_bboxes(self, image, bboxes, filename, aug_type):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
        ax.imshow(image.permute(1, 2, 0))

        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f"Augmentation: {aug_type}")
        save_path = os.path.join(self.bbox_vis_dir, f"{filename[:-4]}_{aug_type}_bbox.png")
        plt.savefig(save_path)
        plt.close(fig)

    def _organize_annotations_by_image_id(self):
        annotations_by_image = {}
        for annotation in self.coco_dict["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        return annotations_by_image

    def _extract_bboxes_and_labels(self, image_id, annotations_by_image):
        bboxes, class_labels = [], []
        if image_id in annotations_by_image:
            for annotation in annotations_by_image[image_id]:
                x, y, w, h = annotation["bbox"]
                bboxes.append([x, y, w, h])
                class_labels.append(annotation["category_id"])
        return bboxes, class_labels


def load_coco_as_dataframe(coco_path):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    data = []
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        data.append({
            "image_id": image_id,
            "category_id": category_id,
            "class_name": categories[category_id],
            "bbox_x": ann["bbox"][0],
            "bbox_y": ann["bbox"][1],
            "bbox_width": ann["bbox"][2],
            "bbox_height": ann["bbox"][3]
        })
    
    return pd.DataFrame(data)


class AugmentationAnalyzer:
    def __init__(self, original_data, augmented_data):
        self.original_data = original_data
        self.augmented_data = augmented_data

    def plot_class_distribution(self):
        original_counts = self.original_data['class_name'].value_counts()
        augmented_counts = self.augmented_data['class_name'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(original_counts.index, original_counts.values, alpha=0.7, label="Original")
        plt.bar(augmented_counts.index, augmented_counts.values, alpha=0.7, label="Augmented")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class Distribution Comparison")
        plt.legend()
        plt.show()


# Configuration for grayscale-friendly augmentations
config = {
    "DATA_DIR": "./data",
    "AUGMENTATION_PATH": "./data/augmentation",
    "VISUALIZATION_DIR": "./data/bbox_vis",
    "augmentation_params": {
        "horizontal_flip": 0.5,
        "brightness_contrast": 0.5,
        "blur": 3,
        "noise": 0.3,
        "normalize": True
    }
}

def run():
    # Verify if the original COCO file exists
    if not os.path.exists(config["DATA_DIR"] + "/coco/train/cassette1_train_corrected_coco.json"):
        print("Error: Original COCO file not found.")
        return
    
    # Augmentation
    augmentor_original = ImageAugmentor(coco_file_name=coco_file_name, config=config, split="train", image_type="original")
    augmentor_sliced = ImageAugmentor(coco_file_name=coco_file_name, config=config, split="train", image_type="sliced")
    augmentor_original.process_images()
    augmentor_sliced.process_images()

    # Save augmented COCO JSON files
    augmentor_original.save_augmented_coco_json("./data/augmentation/original_images/train_annotations_augmented_original.json")
    augmentor_sliced.save_augmented_coco_json("./data/augmentation/sliced_images/train_annotations_augmented_sliced.json")

    # Load original and augmented data for analysis
    original_data = load_coco_as_dataframe("./data/coco/train/cassette1_train_corrected_coco.json")
    augmented_data_path = "./data/augmentation/original_images/train_annotations_augmented_original.json"
    
    if not os.path.exists(augmented_data_path):
        print(f"Error: Augmented data not found at {augmented_data_path}.")
        return

    augmented_data = load_coco_as_dataframe(augmented_data_path)

    # Analysis
    analyzer = AugmentationAnalyzer(original_data, augmented_data)
    analyzer.plot_class_distribution()

if __name__ == "__main__":
    run()
