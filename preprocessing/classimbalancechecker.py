# class_imbalance_check.py
import json
from collections import Counter, defaultdict
import os.path
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

coco_file_name = 'cassette2'  # Change file name of original coco JSON file

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

def run():
    # Load original and sliced annotation files
    with open(f"./data/coco_json_files/{coco_file_name}.json") as f:
        original_annotations = json.load(f)
    with open(f"./data/coco_json_files/{coco_file_name}_sliced_coco.json") as f:
        sliced_annotations = json.load(f)

    # Initialize COCO objects
    coco_original = COCO(f"./data/coco_json_files/{coco_file_name}.json")
    coco_sliced = COCO(f"./data/coco_json_files/{coco_file_name}_sliced_coco.json")

    # Map category IDs to names for readability
    category_mapping = {cat['id']: cat['name'] for cat in coco_original.loadCats(coco_original.getCatIds())}

    # Calculate counts for original and sliced datasets
    original_counts = calculate_class_counts(coco_original)
    sliced_counts = calculate_class_counts(coco_sliced)

    # Calculate dynamic thresholds for bounding box size categorization
    all_areas = [calculate_area(ann['bbox']) for ann in original_annotations['annotations']]
    small_threshold, medium_threshold = np.percentile(all_areas, 33), np.percentile(all_areas, 66)

    # Create bounding box data for original and sliced annotations
    original_bbox_data = create_bbox_size_data(original_annotations, category_mapping, small_threshold, medium_threshold)
    sliced_bbox_data = create_bbox_size_data(sliced_annotations, category_mapping, small_threshold, medium_threshold)

    # Remove duplicates from sliced dataset
    unique_annotations, removed_ids = remove_duplicate_bboxes(sliced_annotations['annotations'], iou_threshold=0.7)
    sliced_annotations['annotations'] = unique_annotations  # Overwrite with deduplicated annotations

    # Calculate class counts after deduplication
    deduped_counts = calculate_class_counts_deduped(unique_annotations, category_mapping)

    # Prepare data for DataFrame display
    data = {
        "Defect Type": [category_mapping[cat_id] for cat_id in original_counts.keys()],
        "Original Count": [original_counts[cat_id] for cat_id in original_counts.keys()],
        "Sliced Count": [sliced_counts.get(cat_id, 0) for cat_id in original_counts.keys()],
        "Deduplicated Sliced Count": [deduped_counts.get(category_mapping[cat_id], 0) for cat_id in original_counts.keys()]
    }

    # Create and display DataFrame
    df = pd.DataFrame(data).sort_values(by="Original Count", ascending=False).reset_index(drop=True)
    print("Class Imbalance Comparison\n", df)

    # Display bounding box data for original and sliced images
    #print("\nBounding Box Data for Original Dataset:", original_bbox_data)
    #print("\nBounding Box Data for Sliced Dataset (After Deduplication):", sliced_bbox_data)

    # Optional: Save deduplicated annotations back to the original file
    with open(f"./data/coco_json_files/{coco_file_name}_sliced_coco.json", "w") as f:
        json.dump(sliced_annotations, f)

if __name__ == "__main__":
    run()
