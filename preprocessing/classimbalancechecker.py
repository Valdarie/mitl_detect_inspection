# class_imbalance_check.py
# Do this right after image slicing
import json
from collections import Counter, defaultdict
import os.path
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

coco_file_name = 'cassette2_train'  # Change file name of original coco JSON file for training

def calculate_class_counts(coco_obj):
    category_ids = coco_obj.getCatIds()
    bbox_counts = Counter()
    for cat_id in category_ids:
        ann_ids = coco_obj.getAnnIds(catIds=[cat_id])
        bbox_counts[cat_id] = len(ann_ids)
    return bbox_counts

def calculate_area(bbox):
    return bbox[2] * bbox[3]

def categorize_by_dynamic_size(area, small_threshold, medium_threshold):
    if area > medium_threshold:
        return "Large"
    elif area > small_threshold:
        return "Medium"
    else:
        return "Small"

def create_bbox_size_data(annotations, category_mapping, small_threshold, medium_threshold):
    bbox_data = []
    for ann in annotations['annotations']:
        defect_type = category_mapping[ann['category_id']]
        area = calculate_area(ann['bbox'])
        size_category = categorize_by_dynamic_size(area, small_threshold, medium_threshold)
        bbox_data.append({"Defect Type": defect_type, "Bounding Box Area": area, "Size Category": size_category})
    return bbox_data

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area, boxB_area = boxA[2] * boxA[3], boxB[2] * boxB[3]
    return inter_area / float(boxA_area + boxB_area - inter_area)

def remove_duplicate_bboxes(annotations, iou_threshold=0.7):
    unique_annotations, removed_ids = [], set()
    category_groups = defaultdict(list)
    for ann in annotations:
        category_groups[ann['category_id']].append(ann)

    for cat_id, bboxes in category_groups.items():
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if bboxes[i]['id'] in removed_ids or bboxes[j]['id'] in removed_ids:
                    continue
                if calculate_iou(bboxes[i]['bbox'], bboxes[j]['bbox']) > iou_threshold:
                    removed_ids.add(bboxes[j]['id'])

    unique_annotations = [ann for ann in annotations if ann['id'] not in removed_ids]
    return unique_annotations, removed_ids

def calculate_class_counts_deduped(annotations, category_mapping):
    bbox_counts = Counter(ann['category_id'] for ann in annotations)
    return {category_mapping[cat_id]: count for cat_id, count in bbox_counts.items()}

def calculate_median_bbox_size(annotations, category_mapping):
    bbox_sizes = defaultdict(list)
    for ann in annotations['annotations']:
        defect_type = category_mapping[ann['category_id']]
        bbox_area = calculate_area(ann['bbox'])
        bbox_sizes[defect_type].append(bbox_area)
    
    # Include all categories, even those without annotations
    return {defect_type: np.median(sizes) if sizes else 0 for defect_type, sizes in bbox_sizes.items()}

def classify_defect_scale(median_bbox_sizes, category_mapping):
    size_threshold = np.median([size for size in median_bbox_sizes.values() if size > 0])  # Dynamic threshold
    
    # Ensure all defect types are included in the classification
    return {
        defect_type: "Large Scale" if median_bbox_sizes.get(defect_type, 0) > size_threshold else "Small Scale"
        for defect_type in category_mapping.values()
    }

def run():
    # Load original and sliced annotation files
    with open(f"./data/coco/{coco_file_name}.json") as f:
        original_annotations = json.load(f)
    with open(f"./data/coco/{coco_file_name}_sliced.json") as f:
        sliced_annotations = json.load(f)

    coco_original = COCO(f"./data/coco/{coco_file_name}.json")
    coco_sliced = COCO(f"./data/coco/{coco_file_name}_sliced.json")
    category_mapping = {cat['id']: cat['name'] for cat in coco_original.loadCats(coco_original.getCatIds())}

    # Calculate median bounding box sizes for all categories
    median_bbox_sizes_original = calculate_median_bbox_size(original_annotations, category_mapping)
    scale_classification_original = classify_defect_scale(median_bbox_sizes_original, category_mapping)

    median_bbox_sizes_sliced = calculate_median_bbox_size(sliced_annotations, category_mapping)
    scale_classification_sliced = classify_defect_scale(median_bbox_sizes_sliced, category_mapping)

    # Prepare data for display, ensuring all defect types are included
    scale_data_original = [
        {"Defect Type": defect_type, "Median Bounding Box Size": median_bbox_sizes_original.get(defect_type, 0), "Scale Classification": scale_classification_original[defect_type]}
        for defect_type in category_mapping.values()
    ]

    scale_data_sliced = [
        {"Defect Type": defect_type, "Median Bounding Box Size": median_bbox_sizes_sliced.get(defect_type, 0), "Scale Classification": scale_classification_sliced[defect_type]}
        for defect_type in category_mapping.values()
    ]

    df_original = pd.DataFrame(scale_data_original).sort_values(by="Median Bounding Box Size", ascending=False).reset_index(drop=True)
    df_sliced = pd.DataFrame(scale_data_sliced).sort_values(by="Median Bounding Box Size", ascending=False).reset_index(drop=True)

    print("\nDefect Type Scale Classification (Original)\n", df_original)
    print("\nDefect Type Scale Classification (Sliced)\n", df_sliced)

if __name__ == "__main__":
    run()