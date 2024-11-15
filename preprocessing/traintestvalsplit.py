# train_test_split.py
import json
import os.path
import random
from collections import Counter
from pycocotools.coco import COCO

# Set up file names for consistency
coco_file_name = 'cassette3'
data_dir = os.path.join(".","data","coco")
original_coco_path = os.path.join(data_dir, f"{coco_file_name}.json")

# Set random seed for reproducibility
random.seed(42)

# Load the original COCO annotations
with open(original_coco_path, 'r') as f:
    original_annotations = json.load(f)

# Define the split ratios
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Split the images into train, validation, and test sets by file name
file_names = [img['file_name'] for img in original_annotations['images']]
random.shuffle(file_names)

# Calculate split points
num_images = len(file_names)
train_end = int(num_images * train_ratio)
val_end = train_end + int(num_images * val_ratio)

# Assign file names to each split
train_filenames = set(file_names[:train_end])
val_filenames = set(file_names[train_end:val_end])
test_filenames = set(file_names[val_end:])

# Function to split annotations based on file names
def split_annotations(images, annotations, filenames_set):
    images_split = [img for img in images if img['file_name'] in filenames_set]
    image_ids_split = {img['id'] for img in images_split}  # Use image ids to link with annotations
    annotations_split = [ann for ann in annotations if ann['image_id'] in image_ids_split]
    return images_split, annotations_split

# Additional function to count annotations per category
def count_annotations_per_category(annotations, category_mapping):
    category_counts = Counter([ann['category_id'] for ann in annotations])
    return {category_mapping[cat_id]: count for cat_id, count in category_counts.items()}

def run():
    # Train, validation, and test splits
    train_images, train_annotations = split_annotations(original_annotations['images'], original_annotations['annotations'], train_filenames)
    val_images, val_annotations = split_annotations(original_annotations['images'], original_annotations['annotations'], val_filenames)
    test_images, test_annotations = split_annotations(original_annotations['images'], original_annotations['annotations'], test_filenames)

    # Ensure no overlap between splits
    assert not (train_filenames & val_filenames), "Train and validation sets have overlapping files!"
    assert not (train_filenames & test_filenames), "Train and test sets have overlapping files!"
    assert not (val_filenames & test_filenames), "Validation and test sets have overlapping files!"

    # Initialize COCO for category mapping
    coco_original = COCO(original_coco_path)
    category_mapping = {cat['id']: cat['name'] for cat in coco_original.loadCats(coco_original.getCatIds())}

    # Prepare data for JSON files
    train_data = {"images": train_images, "annotations": train_annotations, "categories": original_annotations['categories']}
    val_data = {"images": val_images, "annotations": val_annotations, "categories": original_annotations['categories']}
    test_data = {"images": test_images, "annotations": test_annotations, "categories": original_annotations['categories']}

    # Save each split as a JSON file
    with open(os.path.join(data_dir, f"{coco_file_name}_train.json"), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(data_dir, f"{coco_file_name}_val.json"), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(data_dir, f"{coco_file_name}_test.json"), 'w') as f:
        json.dump(test_data, f)

    # Print summary information
    print("Total images:", num_images)
    print("Images assigned to train:", len(train_images))
    print("Images assigned to test:", len(test_images))
    print("Images assigned to validation:", len(val_images))

    # Count annotations per category for each split
    train_counts = count_annotations_per_category(train_annotations, category_mapping)
    val_counts = count_annotations_per_category(val_annotations, category_mapping)
    test_counts = count_annotations_per_category(test_annotations, category_mapping)

    print("\nAnnotation counts per category:")
    print("Train:", train_counts)
    print("Validation:", val_counts)
    print("Test:", test_counts)

    # Re-read JSONs to verify saved splits (Optional)
    for split_name in ['train', 'val', 'test']:
        with open(os.path.join(data_dir, f"{coco_file_name}_{split_name}.json"), 'r') as f:
            split_data = json.load(f)
            print(f"\n{split_name.capitalize()} set verification:")
            print(" - Total images:", len(split_data['images']))
            print(" - Total annotations:", len(split_data['annotations']))

    print("Train, validation, and test splits completed based on file names. Check the coco_json_files directory for the output files.")

if __name__ == "__main__":
    run()