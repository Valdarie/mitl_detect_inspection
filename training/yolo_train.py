import os
import torch
from ultralytics import YOLO


def run():
    # configurations
    MODEL_PATH = os.path.join("..", "models", "yolov10n.pt")
    DATA_DIR = os.path.join("..", "data", "yolo")
    SAVE_DIR = os.path.join("..", "results", "yolo")
    IMG_SIZE = 640
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    EPOCHS = 300
    BATCH_SIZE = 16
    OPTIMIZER = "AdamW"
    LR = 0.01
    EARLY_STOP = 50
    AUGMENT = False

    augmentations = {
        "hsv_h": 0.0,  # hue
        "hsv_s": 0.0,  # saturation
        "hsv_v": 0.0,  # value
        "degrees": 0.0,  # rotation
        "translate": 0.0,  # translate
        "scale": 0.0,  # scale
        "shear": 0.0,  # shear
        "perspective": 0.0,  # perspective
        "flipud": 0.0,  # flip up-down
        "fliplr": 0.0,  # flip left-right
        "mosaic": 0.0,  # mosaic
        "mixup": 0.0  # mixup
    }

    # load a pretrained model
    model = YOLO(MODEL_PATH) 

    # train model with parameters
    results = model.train(
        data=os.path.join(DATA_DIR, "data.yaml"), 
        # resume=False,
        project=SAVE_DIR,
        exist_ok=True,
        plots=True,
        imgsz=IMG_SIZE, 
        device=DEVICE,
        epochs=EPOCHS, 
        batch=BATCH_SIZE,
        optimizer=OPTIMIZER,
        lr0=LR,
        patience=EARLY_STOP,
        **augmentations if AUGMENT else {}
    )


if __name__ == "__main__":
    run()