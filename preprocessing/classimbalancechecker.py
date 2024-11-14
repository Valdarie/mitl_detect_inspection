# class_imbalance_check.py
import json
from collections import Counter, defaultdict
import os.path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pycocotools.coco import COCO

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

def calculate_size_category_counts(bbox_data):
    size_category_counts = defaultdict(lambda: defaultdict(int))
    for data in bbox_data:
        defect_type = data["Defect Type"]
        size_category = data["Size Category"]
        size_category_counts[defect_type][size_category] += 1
    return size_category_counts   

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
    return {defect_type: np.median(sizes) if sizes else 0 for defect_type, sizes in bbox_sizes.items()}


def classify_defect_scale(median_bbox_sizes, category_mapping):
    # Set a dynamic threshold using the median of non-zero median sizes
    size_threshold = np.median([size for size in median_bbox_sizes.values() if size > 0])
    return {
        defect_type: "Large Scale" if median_size > size_threshold else "Small Scale"
        for defect_type, median_size in median_bbox_sizes.items()
    }


def run():
    coco_file_name = 'cassette1_train' # Change file name of original coco json file

    # Load the original and sliced annotation files
    with open(f"./data/coco/{coco_file_name}_corrected_coco.json") as f:  # . in script, .. in notebook
        original_annotations = json.load(f)

    with open(f"./data/coco/{coco_file_name}_sliced_coco.json") as f:  # . in script, .. in notebook
        sliced_annotations = json.load(f)

    coco_original = COCO(f"./data/coco/{coco_file_name}_corrected_coco.json")
    coco_sliced = COCO(f"./data/coco/{coco_file_name}_sliced_coco.json")

    # Map category IDs to names for readability
    category_mapping = {cat['id']: cat['name'] for cat in coco_original.loadCats(coco_original.getCatIds())}

    original_counts = calculate_class_counts(coco_original)
    sliced_counts = calculate_class_counts(coco_sliced)

    all_areas = [calculate_area(ann['bbox']) for ann in original_annotations['annotations']]
    small_threshold, medium_threshold = np.percentile(all_areas, 33), np.percentile(all_areas, 66)

    original_bbox_data = create_bbox_size_data(original_annotations, category_mapping, small_threshold, medium_threshold)
    sliced_bbox_data = create_bbox_size_data(sliced_annotations, category_mapping, small_threshold, medium_threshold)

    # Remove duplicates from sliced dataset
    unique_annotations, removed_ids = remove_duplicate_bboxes(sliced_annotations['annotations'], iou_threshold=0.7)
    sliced_annotations['annotations'] = unique_annotations  # Overwrite with deduplicated annotations
    
    # Map category IDs to names for readability
    category_mapping = {cat['id']: cat['name'] for cat in coco_original.loadCats(original_counts.keys())}
    original_counts_named = {category_mapping[k]: v for k, v in original_counts.items()}
    sliced_counts_named = {category_mapping.get(k, 'Unknown'): v for k, v in sliced_counts.items()}

    # Calculate the total number of bounding boxes across all defect types
    total_original_bboxes = sum(original_counts.values())
    total_sliced_bboxes = sum(sliced_counts.values())

    original_size_counts = calculate_size_category_counts(original_bbox_data)
    sliced_size_counts = calculate_size_category_counts(sliced_bbox_data)
    deduped_size_counts = calculate_size_category_counts(create_bbox_size_data(sliced_annotations, category_mapping, small_threshold, medium_threshold))

    size_data = []
    for defect_type in category_mapping.values():  # Iterate over all known defect types
        for size_category in ["Small", "Medium", "Large"]:
            size_data.append({
                "Defect Type": defect_type,
                "Size Category": size_category,
                "Original Count": original_size_counts[defect_type].get(size_category, 0),
                "Sliced Count": sliced_size_counts[defect_type].get(size_category, 0),
                "Deduplicated Sliced Count": deduped_size_counts[defect_type].get(size_category, 0)
            })

    # Create and display DataFrame
    df = pd.DataFrame(size_data).sort_values(by=["Defect Type", "Size Category"]).reset_index(drop=True)
    print("Class Imbalance by Size Category\n", df)

    with open(f"./data/coco/{coco_file_name}_sliced_coco.json", "w") as f:
        json.dump(sliced_annotations, f) #deduplicated version

    bbox_sizes = defaultdict(list)

    for ann in original_annotations['annotations']:
        defect_type = category_mapping[ann['category_id']]
        bbox_area = calculate_area(ann['bbox'])
        bbox_sizes[defect_type].append(bbox_area)

    # Calculate the median size for each defect type, including all categories in the mapping
    median_bbox_sizes = {defect_type: np.median(sizes) if sizes else 0 for defect_type, sizes in bbox_sizes.items()}
    median_bbox_sizes = {defect_type: median_bbox_sizes.get(defect_type, 0) for defect_type in category_mapping.values()}

    # Determine a threshold for "Small Scale" vs "Large Scale" using the median of non-zero medians
    size_threshold = np.median([size for size in median_bbox_sizes.values() if size > 0])

    # Classify each defect type based on the threshold
    scale_classification = {
        defect_type: "Large Scale" if median_size > size_threshold else "Small Scale"
        for defect_type, median_size in median_bbox_sizes.items()
    }

   # Remove duplicates from sliced dataset
    unique_annotations, removed_ids = remove_duplicate_bboxes(sliced_annotations['annotations'], iou_threshold=0.7)
    sliced_annotations['annotations'] = unique_annotations  # Overwrite with deduplicated annotations

    median_bbox_sizes_original = calculate_median_bbox_size(original_annotations, category_mapping)
    scale_classification_original = classify_defect_scale(median_bbox_sizes_original, category_mapping)

    median_bbox_sizes_sliced = calculate_median_bbox_size(sliced_annotations, category_mapping)
    scale_classification_sliced = classify_defect_scale(median_bbox_sizes_sliced, category_mapping)

    # Prepare data for display
    scale_data_original = [
        {"Defect Type": defect_type, "Median Bounding Box Size": median_size, "Scale Classification": scale_classification_original[defect_type]}
        for defect_type, median_size in median_bbox_sizes_original.items()
    ]

    scale_data_sliced = [
        {"Defect Type": defect_type, "Median Bounding Box Size": median_size, "Scale Classification": scale_classification_sliced[defect_type]}
        for defect_type, median_size in median_bbox_sizes_sliced.items()
    ]

    # Create and display DataFrames
    df_original = pd.DataFrame(scale_data_original).sort_values(by="Median Bounding Box Size", ascending=False).reset_index(drop=True)
    df_sliced = pd.DataFrame(scale_data_sliced).sort_values(by="Median Bounding Box Size", ascending=False).reset_index(drop=True)

    print("\nDefect Type Scale Classification (Original)\n", df_original)
    print("\nDefect Type Scale Classification (Sliced)\n", df_sliced)

if __name__ == "__main__":
    run()

