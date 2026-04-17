# ---------------------------------------------------------------------------- #
# SETUP
# ---------------------------------------------------------------------------- #
import os

import torch
import torchvision
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from deepforest import main, utilities

print(f"VISIBLE CUDA DEVICES: {torch.cuda.device_count()}")

def load_model(checkpoint_dict_path: str = None):
    '''
    Load the prtraind deepforest model from either Hugging face or from the PT file.
    If finetuning the modeel on an air-gappe HPC cluster we need to load the model from the PT file.
    If the compute platform is connected to the internet, we can load the model from Hugging face.
    '''

    if checkpoint_dict_path is not None:
        # Load the model from the PT file
        # ── 1. Load bundled config (local only, no network) ───────────────────────────
        cfg = utilities.load_config()

        # ── 2. Build architecture with ZERO downloads ─────────────────────────────────
        #   weights=None      → no COCO RetinaNet download
        #   weights_backbone=None → no ResNet-50 ImageNet download  ← the missing piece
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=cfg.num_classes,
            score_thresh=cfg.score_thresh,
            nms_thresh=cfg.nms_thresh,
        )

        # ── 3. Load NEON.pt weights ───────────────────────────────────────────────────
        checkpoint = torch.load(checkpoint_dict_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip legacy "model." prefix if present (same logic as RetinaNetHub's hook)
        state_dict = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()

        # ── 4. Set attributes deepforest's predict_tile expects on self.model ─────────
        model.label_dict = dict(cfg.label_dict)
        model.num_classes = cfg.num_classes
        model.nms_thresh  = cfg.nms_thresh
        model.score_thresh = cfg.score_thresh
        
        # ── 5. Wrap in deepforest (model != None → skips create_model() entirely) ─────
        m = main.deepforest(model=model, config_args={"model": {"name": None}})
        m.label_dict = dict(cfg.label_dict)
        m.numeric_to_label_dict = {v: k for k, v in cfg.label_dict.items()}

        return m
    
    else:
        # Load the model from Hugging face
        f = main.deepforest()
        f.load_model("weecology/deepforest-tree")
        return f


def run_training():
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    PT_FILE = "../resources/NEON.pt"
    f = load_model(PT_FILE)
    # f = main.deepforest()
    # f.load_model("weecology/deepforest-tree")

    TRAIN_CSV = "../sample_data/annotations/master_train.csv"
    VAL_CSV = "../sample_data/annotations/master_val.csv"
    IMAGE_ROOT_DIR = "../all_chips"
    OUT_DIR = "../lightning_logs"

    # Training config
    f.config.train.csv_file = TRAIN_CSV
    f.config.train.root_dir = IMAGE_ROOT_DIR

    f.config.train.epochs = 100
    f.config.train.lr = 1e-4

    # Data loading / performance
    f.config.batch_size = 32
    f.config.workers = 8
    f.config.devices = 1  # per-rank; each srun task owns exactly 1 GPU

    # Simple augmentations
    f.config.train.augmentations = ["HorizontalFlip", "VerticalFlip"]

    # Validation config
    f.config.validation.csv_file = VAL_CSV
    f.config.validation.root_dir = IMAGE_ROOT_DIR
    f.config.validation.val_accuracy_interval = 1
    f.config.validation.iou_threshold = 0.4

    logger = TensorBoardLogger(save_dir=OUT_DIR, name="lightning_logs")

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(OUT_DIR, "checkpoints"),
        filename="deepforest-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Increase the detection threshold to avoid ignored detections in dense images
    if hasattr(f, "mAP_metric"):
        f.mAP_metric.max_detection_thresholds = [1, 10, 100, 1000]
        f.mAP_metric.warn_on_many_detections = False

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_cb],
        accelerator="gpu",
        devices=2,          # 1 GPU per srun task (CUDA_VISIBLE_DEVICES=0)
        num_nodes=1,
        strategy="ddp",
        max_epochs=f.config.train.epochs,
        enable_progress_bar=True,
    )

    # Attach the trainer so DeepForest's internal references still work
    f.trainer = trainer
    trainer.fit(f)

if __name__ == '__main__':
    run_training()
