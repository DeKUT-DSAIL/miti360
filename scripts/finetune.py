# ---------------------------------------------------------------------------- #
# SETUP
# ---------------------------------------------------------------------------- #
import os

# Pin this process to its own GPU *before* importing torch/CUDA.
# srun sets SLURM_LOCALID=0 for every task when --gpus-per-task=1, so each
# process will see exactly GPU 0 through its own CUDA_VISIBLE_DEVICES.
# _local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
# os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import torch
import torchvision
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from deepforest import main, utilities

print(#f"RANK={os.environ.get('RANK','?')} LOCAL_RANK={_local_rank} "
      #f"WORLD_SIZE={os.environ.get('WORLD_SIZE','?')} "
      f"VISIBLE CUDA DEVICES: {torch.cuda.device_count()}")


def load_model(checkpoint_dict_path):
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


def run_training():
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    PT_FILE = "../resources/NEON.pt"
    f = load_model(PT_FILE)
    # f = main.deepforest()
    # f.load_model("weecology/deepforest-tree")

    TRAIN_CSV = "../training_dataset/master_train.csv"
    VAL_CSV = "../training_dataset/master_val.csv"
    TRAIN_IMAGE_ROOT_DIR = "../training_dataset/train_chips"
    VAL_IMAGE_ROOT_DIR = "../training_dataset/val_chips"
    OUT_DIR = "../lightning_logs"

    # Training config
    f.config.train.csv_file = TRAIN_CSV
    f.config.train.root_dir = TRAIN_IMAGE_ROOT_DIR

    f.config.train.epochs = 100
    f.config.train.lr = 1e-4

    # Data loading / performance
    # Adjusted to 4 to prevent CUDA OOM errors; scale up if VRAM permits.
    f.config.batch_size = 32
    f.config.workers = 2
    f.config.devices = 1  # per-rank; each srun task owns exactly 1 GPU

    # Simple augmentations
    f.config.train.augmentations = ["HorizontalFlip", "VerticalFlip"]

    # Validation config
    f.config.validation.csv_file = VAL_CSV
    f.config.validation.root_dir = VAL_IMAGE_ROOT_DIR
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

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        min_delta=0.001,
        verbose=True,
    )

    # Increase the detection threshold to avoid ignored detections in dense images
    if hasattr(f, "mAP_metric"):
        f.mAP_metric.max_detection_thresholds = [1, 10, 100, 1000]
        f.mAP_metric.warn_on_many_detections = False

    # Build pl.Trainer directly instead of using f.create_trainer().
    # DeepForest's create_trainer can silently override strategy/devices
    # with its own internal config reads, which is what caused the original
    # "requested gpu: [0,1,2,3] but machine only has: [0]" error.
    #
    # srun spawns one process per GPU; each process declares devices=1.
    # SLURMEnvironment reads RANK/LOCAL_RANK/WORLD_SIZE from the env vars
    # set in the sbatch, so DDP coordination works across all 4 ranks.
    slurm_env = SLURMEnvironment()
    ddp = DDPStrategy(cluster_environment=slurm_env, find_unused_parameters=False)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_cb, early_stop_cb],
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
    # mp.set_start_method('spawn', force=True) is generally not required
    # when using standard DDP inside the __main__ block, but can be explicitly
    # called here if specific multiprocessing hangups occur with the dataloaders.

    run_training()
