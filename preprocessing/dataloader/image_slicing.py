'''
IGNORE
'''

import os.path
import re
import shutil
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from sahi.slicing import slice_coco
from sahi.utils.file import load_json, save_json
from tqdm import tqdm
import argparse


def plot_mono_bboxes_coco(annotation: dict, root_dir: str, img_dir: str, save_dir: str):

    # read image
    for img in tqdm(annotation["images"],  desc="Plotting unsliced coco dataset bounding boxes"):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)

        mono_img = Image.open(os.path.join(root_dir, img_dir, img["file_name"])).convert("L")
        rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))

        # iterate over all annotations
        for ann_ind in range(len(annotation["annotations"])):
            
            if annotation["annotations"][ann_ind]["image_id"] == img["id"]:
                # convert coco bbox to pil bbox
                xywh = annotation["annotations"][ann_ind]["bbox"]
                xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]

                # visualize bbox over image
                ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")

        ax.axis("off")
        ax.imshow(rgb_img)

        fig.savefig(os.path.join(root_dir, save_dir, img["file_name"][:-4] + ".png"))
        plt.close()


def plot_sliced_mono_bboxes_coco(annotations: dict, root_dir: str, img_dir: str, save_dir: str):

    # get number of slices, row-wise and col-wise
    num_rows = len(set([img["file_name"].split("_")[-1] for img in annotations["images"]]))
    num_cols = len(set([img["file_name"].split("_")[-2] for img in annotations["images"]]))
    
    for img in tqdm(annotations["images"][::num_rows * num_cols], desc="Plotting sliced coco dataset bounding boxes"):
        img_idx = img["id"] - 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9), constrained_layout=True)
        # fig.tight_layout()
        # fig.subplots_adjust(wspace=0.1, hspace=0.1)

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):

                # read image
                mono_img = Image.open(os.path.join(root_dir, img_dir, annotations["images"][img_idx]["file_name"])).convert("L")
                rgb_img = Image.merge("RGB", (mono_img, mono_img, mono_img))

                # iterate over all annotations
                for ann_ind in range(len(annotations["annotations"])):

                    # find annotations that belong the selected image
                    if annotations["annotations"][ann_ind]["image_id"] == annotations["images"][img_idx]["id"]:
                        # convert coco bbox to pil bbox
                        xywh = annotations["annotations"][ann_ind]["bbox"]
                        xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]

                        # visualize bbox over image
                        ImageDraw.Draw(rgb_img).rectangle(xyxy, width=5, outline="lime")

                axes[row_idx, col_idx].imshow(rgb_img)
                axes[row_idx, col_idx].axis("off")

                img_idx += 1

        # save and close plot
        fig.savefig(os.path.join(root_dir, save_dir, img["file_name"].split("_")[0] + "_sliced.png"))
        plt.close()


def slice_images(root_dir: str, anno_fname: str, img_dir: str, slice_size: int, overlap_ratio: float):
    coco_dict = load_json(os.path.join(root_dir, anno_fname))
    dataset_name = re.split('_|\.', anno_fname)[0]

    # slice images and annotations
    sliced_coco_dict, _ = slice_coco(
        coco_annotation_file_path=os.path.join(root_dir, anno_fname),
        image_dir=os.path.join(root_dir, img_dir),
        output_coco_annotation_file_name=f"../{dataset_name}_sliced",
        ignore_negative_samples=False,
        output_dir=os.path.join(root_dir, f"{img_dir}_sliced"),
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio=0.1,
        verbose=False
    )

    # create save dir, delete save dir if already exists
    save_dir = f"{dataset_name}_bboxes_vis"
    save_path = os.path.join(root_dir, save_dir)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)


    # visualize bboxes on unsliced and sliced images
    plot_mono_bboxes_coco(
        annotation=coco_dict,
        root_dir=root_dir,
        img_dir=img_dir,
        save_dir=save_dir
    )
    plot_sliced_mono_bboxes_coco(
        annotations=sliced_coco_dict,
        root_dir=root_dir,
        img_dir=f"{img_dir}_sliced",
        save_dir=save_dir
    )

    return coco_dict, sliced_coco_dict


def run():
    DATA_DIR = os.path.join(".", "data")
    ANNOTATION_FNAME = "coco_json_files/cassette1_train.json"  # change to train_coco.json or train.json etc
    ANNOTATION_PATH = os.path.join(DATA_DIR, ANNOTATION_FNAME)
    CORRECTED_ANNOTATION_FNAME = "coco_json_files/cassette1_train_sliced.json"  # change to train_coco.json or train.json etc
    CORRECTED_ANNOTATION_PATH = os.path.join(DATA_DIR, CORRECTED_ANNOTATION_FNAME)
    IMAGE_DIR = "images" 

    SLICE_SIZE = 640
    OVERLAP_RATIO = 0.2

    # correct image paths in annotations
    coco_dict = load_json(ANNOTATION_PATH)
    [img.update({"file_name": img["file_name"].split("/")[-1]}) for img in coco_dict["images"]]
    save_json(coco_dict, save_path=CORRECTED_ANNOTATION_PATH)

    # slice and save annotations and images
    coco, sliced_coco = slice_images(
        root_dir=DATA_DIR, 
        anno_fname=CORRECTED_ANNOTATION_FNAME, 
        img_dir=IMAGE_DIR,
        slice_size=SLICE_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )


if __name__ == "__main__":
    run()