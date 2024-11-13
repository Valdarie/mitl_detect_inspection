import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directory paths
data_dir = './data'
sliced_images_dir = os.path.join(data_dir, 'images_sliced')
coco_sliced_path = os.path.join(data_dir, 'coco_sliced_coco.json')
adjusted_annotations_path = os.path.join(data_dir, 'adjusted_annotations.json')

# Load COCO sliced data
with open(coco_sliced_path, 'r') as f:
    coco_sliced_data = json.load(f)

# Helper function to read image dimensions
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

# Adjust bounding boxes to each slice based on actual slice dimensions
def adjust_bboxes_to_slices(coco_data, sliced_images_dir):
    adjusted_annotations = {
        "images": coco_data["images"],
        "annotations": [],
        "categories": coco_data["categories"]
    }

    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        image_filename = image_info["file_name"]
        image_path = os.path.join(sliced_images_dir, image_filename)
        
        # Get actual dimensions of the sliced image
        slice_width, slice_height = get_image_dimensions(image_path)
        
        # Define boundaries of the current slice
        x_min_slice, y_min_slice = 0, 0
        x_max_slice, y_max_slice = slice_width, slice_height
        
        # Loop through annotations and check for intersections
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] != image_id:
                continue
            
            x_min_bbox, y_min_bbox, bbox_width, bbox_height = annotation["bbox"]
            x_max_bbox, y_max_bbox = x_min_bbox + bbox_width, y_min_bbox + bbox_height
            
            # Check if bounding box intersects with slice
            intersects = not (x_min_bbox > x_max_slice or x_max_bbox < x_min_slice or
                              y_min_bbox > y_max_slice or y_max_bbox < y_min_slice)
            
            if intersects:
                # Adjust the bbox coordinates for the slice
                adjusted_bbox = [
                    max(x_min_bbox - x_min_slice, 0),
                    max(y_min_bbox - y_min_slice, 0),
                    min(bbox_width, x_max_slice - x_min_bbox),
                    min(bbox_height, y_max_slice - y_min_bbox)
                ]
                
                # Update annotation with adjusted bbox
                adjusted_annotation = {
                    "id": annotation["id"],
                    "image_id": image_id,
                    "category_id": annotation["category_id"],
                    "segmentation": annotation.get("segmentation", []),
                    "bbox": adjusted_bbox,
                    "ignore": annotation.get("ignore", 0),
                    "iscrowd": annotation.get("iscrowd", 0),
                    "area": adjusted_bbox[2] * adjusted_bbox[3]
                }
                adjusted_annotations["annotations"].append(adjusted_annotation)

    return adjusted_annotations

# Adjust the annotations based on the actual image slice dimensions
adjusted_annotations = adjust_bboxes_to_slices(coco_sliced_data, sliced_images_dir)

# Save adjusted annotations
with open(adjusted_annotations_path, 'w') as f:
    json.dump(adjusted_annotations, f)
print(f"Adjusted annotations saved to {adjusted_annotations_path}")

# Visualization of all adjusted bounding boxes
def visualize_all_annotations(adjusted_annotations, sliced_images_dir):
    for image_info in adjusted_annotations["images"]:
        image_filename = image_info["file_name"]
        image_path = os.path.join(sliced_images_dir, image_filename)
        
        # Ensure the image file exists
        if not os.path.exists(image_path):
            print(f"Image file {image_filename} not found. Skipping.")
            continue

        # Get annotations for the current slice
        image_id = image_info["id"]
        image_annotations = [ann for ann in adjusted_annotations["annotations"] if ann["image_id"] == image_id]

        # Print whether bounding boxes were found or not
        if image_annotations:
            print(f"Found {len(image_annotations)} bounding box(es) for {os.path.basename(image_path)}")
        else:
            print(f"No bounding boxes found for {os.path.basename(image_path)}")

        # Plot image and annotations
        fig, ax = plt.subplots(1)
        img = Image.open(image_path)
        ax.imshow(img)
        
        # Draw each bounding box if present
        for annotation in image_annotations:
            x, y, w, h = annotation["bbox"]
            print(f"Drawing bbox at (x={x}, y={y}, width={w}, height={h})")  # Debugging information
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        # Show the plot for each image
        plt.show()

# Call the visualization function for all annotations
visualize_all_annotations(adjusted_annotations, sliced_images_dir)
