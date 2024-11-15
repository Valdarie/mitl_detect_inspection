import json
import os
import shutil
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm

coco_file_name = 'cassette2_train'
base_path = "./data/coco/"

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

# Define paths for JSON and CSV output
json_output_path = os.path.join(base_path, target_split)
csv_output_path = json_output_path
os.makedirs(csv_output_path, exist_ok=True)

def calculate_class_counts(coco_obj):
    return Counter(cat_id for cat_id in coco_obj.getCatIds() for _ in coco_obj.getAnnIds(catIds=[cat_id]))

def calculate_area(bbox):
    return bbox[2] * bbox[3]

def categorize_by_dynamic_size(area, small_threshold, medium_threshold):
    if area > medium_threshold:
        return "Large"
    elif area > small_threshold:
        return "Medium"
    return "Small"

def create_bbox_size_data(annotations, category_mapping, small_threshold, medium_threshold):
    return [
        {
            "Defect Type": category_mapping[ann['category_id']],
            "Bounding Box Area": calculate_area(ann['bbox']),
            "Size Category": categorize_by_dynamic_size(calculate_area(ann['bbox']), small_threshold, medium_threshold),
        }
        for ann in tqdm(annotations['annotations'], desc="Processing Bounding Boxes")
    ]

def calculate_size_category_counts(bbox_data):
    size_counts = defaultdict(lambda: defaultdict(int))
    for data in bbox_data:
        size_counts[data["Defect Type"]][data["Size Category"]] += 1
    return size_counts

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    return inter_area / float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter_area)

def remove_duplicate_bboxes(annotations, iou_threshold=0.7):
    removed_ids = set()
    category_groups = defaultdict(list)

    for ann in annotations:
        category_groups[ann['category_id']].append(ann)

    for bboxes in category_groups.values():
        for i in range(len(bboxes)):
            for j in tqdm(range(i + 1, len(bboxes)), desc="Checking Duplicates"):
                if bboxes[i]['id'] in removed_ids or bboxes[j]['id'] in removed_ids:
                    continue
                if calculate_iou(bboxes[i]['bbox'], bboxes[j]['bbox']) > iou_threshold:
                    removed_ids.add(bboxes[j]['id'])

    return [ann for ann in annotations if ann['id'] not in removed_ids]

def calculate_median_bbox_size(annotations, category_mapping):
    bbox_sizes = defaultdict(list)
    for ann in annotations['annotations']:
        bbox_sizes[category_mapping[ann['category_id']]].append(calculate_area(ann['bbox']))
    return {defect_type: np.median(sizes) if sizes else 0 for defect_type, sizes in bbox_sizes.items()}

def classify_defect_scale(median_bbox_sizes):
    size_threshold = np.median([size for size in median_bbox_sizes.values() if size > 0])
    return {defect_type: "Large Scale" if size > size_threshold else "Small Scale" for defect_type, size in median_bbox_sizes.items()}

def move_coco_file():
    for file_type in ["sliced", "corrected"]:
        source_path = os.path.join(base_path, f"{coco_file_name}_{file_type}_coco.json")
        destination_path = os.path.join(json_output_path, f"{coco_file_name}_{file_type}_coco.json")

        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved {source_path} to {destination_path}")
        else:
            print(f"Source file {source_path} does not exist.")

def run():
    # Load annotations
    original_path = os.path.join(base_path, f"{coco_file_name}_corrected_coco.json")
    sliced_path = os.path.join(base_path, f"{coco_file_name}_sliced_coco.json")

    with open(original_path) as f:
        original_annotations = json.load(f)

    with open(sliced_path) as f:
        sliced_annotations = json.load(f)

    coco_original = COCO(original_path)
    category_mapping = {cat['id']: cat['name'] for cat in coco_original.loadCats(coco_original.getCatIds())}

    # Calculate thresholds
    all_areas = [calculate_area(ann['bbox']) for ann in original_annotations['annotations']]
    small_threshold, medium_threshold = np.percentile(all_areas, 33), np.percentile(all_areas, 66)

    # Generate data
    original_bbox_data = create_bbox_size_data(original_annotations, category_mapping, small_threshold, medium_threshold)
    sliced_bbox_data = create_bbox_size_data(sliced_annotations, category_mapping, small_threshold, medium_threshold)

    unique_annotations = remove_duplicate_bboxes(sliced_annotations['annotations'])
    sliced_annotations['annotations'] = unique_annotations

    # Calculate counts
    original_size_counts = calculate_size_category_counts(original_bbox_data)
    sliced_size_counts = calculate_size_category_counts(sliced_bbox_data)
    deduped_size_counts = calculate_size_category_counts(create_bbox_size_data(sliced_annotations, category_mapping, small_threshold, medium_threshold))

    size_data = [
        {
            "Defect Type": defect_type,
            "Size Category": size_category,
            "Original Count": original_size_counts[defect_type].get(size_category, 0),
            "Sliced Count": sliced_size_counts[defect_type].get(size_category, 0),
            "Deduplicated Sliced Count": deduped_size_counts[defect_type].get(size_category, 0),
        }
        for defect_type in category_mapping.values()
        for size_category in ["Small", "Medium", "Large"]
    ]

    df = pd.DataFrame(size_data).sort_values(by=["Defect Type", "Size Category"]).reset_index(drop=True)
    print("Class Imbalance by Size Category\n", df)

    with open(sliced_path, "w") as f:
        json.dump(sliced_annotations, f)

    # Calculate and classify scales
    median_bbox_sizes_original = calculate_median_bbox_size(original_annotations, category_mapping)
    median_bbox_sizes_sliced = calculate_median_bbox_size(sliced_annotations, category_mapping)

    scale_classification_original = classify_defect_scale(median_bbox_sizes_original)
    scale_classification_sliced = classify_defect_scale(median_bbox_sizes_sliced)

    scale_data_original = [
        {"Defect Type": defect_type, "Median Bounding Box Size": median_size, "Scale Classification": scale_classification_original[defect_type]}
        for defect_type, median_size in median_bbox_sizes_original.items()
    ]

    scale_data_sliced = [
        {"Defect Type": defect_type, "Median Bounding Box Size": median_size, "Scale Classification": scale_classification_sliced[defect_type]}
        for defect_type, median_size in median_bbox_sizes_sliced.items()
    ]

    df_original = pd.DataFrame(scale_data_original).sort_values(by="Median Bounding Box Size", ascending=False).reset_index(drop=True)
    df_sliced = pd.DataFrame(scale_data_sliced).sort_values(by="Median Bounding Box Size", ascending=False).reset_index(drop=True)

    print("\nDefect Type Scale Classification (Original)\n", df_original)
    print("\nDefect Type Scale Classification (Sliced)\n", df_sliced)

    # Save DataFrames to CSV
    df.to_csv(os.path.join(csv_output_path, f'{coco_file_name}_imbalance.csv'), index=False)
    df_original.to_csv(os.path.join(csv_output_path, f'{coco_file_name}_original.csv'), index=False)
    df_sliced.to_csv(os.path.join(csv_output_path, f'{coco_file_name}_sliced.csv'), index=False)

    move_coco_file()

if __name__ == "__main__":
    run()
