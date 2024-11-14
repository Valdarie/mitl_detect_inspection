import copy
import json
import os
import shutil

from PIL import Image
from tqdm import tqdm
import yaml


def merge_dataset_coco(base: dict, others: list[dict], save_path: str):
    # check classes are same across dataset
    if any(dataset["categories"] != base["categories"] for dataset in others):
        raise ValueError("All datasets must have the same set of labels")
    
    n_imgs = len(base["images"])
    n_annos = len(base["annotations"])
    print(f"Number of images in base dataset: {n_imgs}")
    print(f"Number of annotations in base dataset: {n_annos}\n")

    for idx, coco in enumerate(others):
        # avoid issue with referencing same dictionary
        temp = copy.deepcopy(coco)

        for img in tqdm(temp["images"], desc=f"Updating image indices from dataset {idx}"):
            img["id"] += n_imgs 

        for anno in tqdm(temp["annotations"], desc=f"Updating annotation indices from dataset {idx}"):
            anno["image_id"] += n_imgs
            anno["id"] += n_annos

        # update number of images and annotations
        n_imgs += len(temp["images"])
        n_annos += len(temp["annotations"])

        base["images"].extend(temp["images"])
        base["annotations"].extend(temp["annotations"])

    # get all updated image and annotation indices
    img_indices = [img["id"] for img in base["images"]]
    anno_indices = [anno["id"] for anno in base["annotations"]]

    # save merged coco if no duplicated indices
    if len(img_indices) == len(set(img_indices)) and len(anno_indices) == len(set(anno_indices)):
        print(f"\nNumber of images in updated dataset: {len(img_indices)}")
        print(f"Number of annotations in updated dataset: {len(anno_indices)}")

        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(base, f, ensure_ascii=False, indent=4)
    else:
        raise ValueError("Duplicated indices found")


def run():
    DATA_DIR = os.path.join("..", "data", "coco")
    SAVE_PATH = os.path.join(DATA_DIR, "merged_coco.json")
    ANNO_PATHS = {
        "train": os.path.join(DATA_DIR, "train_example.json"),
        "val": os.path.join(DATA_DIR, "val_example.json"),
    }

    with open(ANNO_PATHS["train"]) as f:
        base_coco = json.load(f)

    with open(ANNO_PATHS["val"]) as f:
        other_coco = json.load(f)

    merge_dataset_coco(base=base_coco, others=[other_coco, other_coco], save_path=SAVE_PATH)


if __name__ == "__main__":
    run()
