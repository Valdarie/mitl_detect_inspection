import json
import pandas as pd
import numpy as np
from collections import Counter

coco_file_name = 'coco'  # Change file name of original COCO JSON file

# Load the original and sliced annotation files with exception handling
try:
    with open(f"./data/coco_json_files/{coco_file_name}.json") as f:
        original_annotations = json.load(f)
except FileNotFoundError:
    print(f"Error: File {coco_file_name}.json not found.")
    raise

try:
    with open(f"./data/coco_json_files/{coco_file_name}_sliced_coco.json") as f:
        sliced_annotations = json.load(f)
except FileNotFoundError:
    print(f"Error: File {coco_file_name}_sliced_coco.json not found.")
    raise

# Map category IDs to names with a check for missing IDs
category_mapping = {cat['id']: cat['name'] for cat in original_annotations['categories']}
for ann in original_annotations['annotations']:
    if ann['category_id'] not in category_mapping:
        print(f"Warning: Category ID {ann['category_id']} not found in categories.")

# Calculate bounding box area
def calculate_area(bbox):
    return bbox[2] * bbox[3]  # width * height

# Collect all bounding box areas from the original dataset
all_areas = [calculate_area(ann['bbox']) for ann in original_annotations['annotations']]

# Calculate dynamic thresholds based on percentiles
small_threshold = np.percentile(all_areas, 33)
medium_threshold = np.percentile(all_areas, 66)

# Function to categorize by dynamic size thresholds
def categorize_by_dynamic_size(area, small_threshold, medium_threshold):
    if area > medium_threshold:
        return "Large"
    elif area > small_threshold:
        return "Medium"
    else:
        return "Small"

# Prepare data for bounding box size analysis with dynamic thresholds
def create_bbox_size_data(annotations, category_mapping, small_threshold, medium_threshold):
    bbox_data = []
    for ann in annotations['annotations']:
        defect_type = category_mapping[ann['category_id']]
        area = calculate_area(ann['bbox'])
        size_category = categorize_by_dynamic_size(area, small_threshold, medium_threshold)
        bbox_data.append({
            "Defect Type": defect_type,
            "Bounding Box Area": area,
            "Size Category": size_category
        })
    return bbox_data

# Create bounding box data for original and sliced annotations
original_bbox_data = create_bbox_size_data(original_annotations, category_mapping, small_threshold, medium_threshold)
sliced_bbox_data = create_bbox_size_data(sliced_annotations, category_mapping, small_threshold, medium_threshold)

# Convert bounding box data to DataFrames
original_bbox_df = pd.DataFrame(original_bbox_data)
sliced_bbox_df = pd.DataFrame(sliced_bbox_data)

# Group by defect type and size category, counting occurrences
grouped_original_bbox_df = original_bbox_df.groupby(['Defect Type', 'Size Category']).size().reset_index(name='Original Count')
grouped_sliced_bbox_df = sliced_bbox_df.groupby(['Defect Type', 'Size Category']).size().reset_index(name='Sliced Count')

# Merge original and sliced data on defect type and size category
merged_bbox_df = pd.merge(grouped_original_bbox_df, grouped_sliced_bbox_df, on=['Defect Type', 'Size Category'], how='outer').fillna(0)

# Convert counts to integers
merged_bbox_df['Original Count'] = merged_bbox_df['Original Count'].astype(int)
merged_bbox_df['Sliced Count'] = merged_bbox_df['Sliced Count'].astype(int)

print("\nGrouped by Dynamic Bounding Box Size Category")
print(merged_bbox_df.to_string(index=False))
