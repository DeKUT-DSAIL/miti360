# ---------------------------------------------------------------------------- #
# SETUP
# ---------------------------------------------------------------------------- #
import os
import warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="pytorch_lightning.utilities._pytree"
)
warnings.filterwarnings(
    "ignore",
    message="Converting predictions to GeoDataFrame.*",
    category=UserWarning,
    module="deepforest",
)
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries.*",
    category=FutureWarning,
    module="deepforest",
)
warnings.filterwarnings(
    "ignore",
    message="Got processor for bboxes, but no transform to process it.*",
    category=UserWarning,
    module="albumentations",
)

# ---- Defensive DDP rank bootstrap (must happen BEFORE pytorch_lightning import) ----
# lightning_fabric reads os.environ["RANK"] at import time and calls int() on it.
# If RANK is set to "" (e.g. from a typo in the sbatch: SLURM_PROC_ID vs SLURM_PROCID),
# the import crashes with: ValueError: invalid literal for int() with base 10: ''
# This block ensures RANK/LOCAL_RANK/WORLD_SIZE are always valid integers before
# any lightning import fires.
def _slurm_int(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    return int(val) if val else default

if not os.environ.get("RANK", "").strip():
    os.environ["RANK"]       = str(_slurm_int("SLURM_PROCID",   0))

# OVERWRITE LOCAL_RANK unconditionally here!
# The sbatch script exports LOCAL_RANK=$SLURM_LOCALID (which ranges 0-3),
# but because srun uses --gpus-per-task=1, CUDA_VISIBLE_DEVICES limits each
# python process to seeing exactly 1 GPU (index 0). Lightning's parallel_devices
# list will have length 1, so local_rank MUST be 0 to avoid an IndexError.
os.environ["LOCAL_RANK"] = "0"

if not os.environ.get("WORLD_SIZE", "").strip():
    os.environ["WORLD_SIZE"] = str(_slurm_int("SLURM_NTASKS",   1))

# ---- Prevent Lightning's auto-detected SLURMEnvironment from firing its
# per-node task count validator (slurm.py:163) which would require devices==4
# while each srun task physically only has 1 GPU visible.
#
# Lightning auto-instantiates SLURMEnvironment whenever SLURM_NTASKS is present.
# Its validate_settings() then reads SLURM_NTASKS_PER_NODE to enforce:
#   devices == ntasks_per_node
# By unsetting those two vars BEFORE any Lightning import we hide the per-node
# count from the validator.  RANK / LOCAL_RANK / WORLD_SIZE (set above from the
# real SLURM values) are what Lightning DDP actually uses for coordination, so
# removing SLURM_NTASKS_PER_NODE has no effect on correctness.
# Lightning's _validate_srun_variables() (slurm.py:221) explicitly rejects
# SLURM_NTASKS (global task flag) and raises:
#   "You set --ntasks=4 ... HINT: Use --ntasks-per-node instead"
# It also rejects SLURM_NTASKS_PER_NODE in validate_settings() when devices!=ntasks.
# Unset ALL raw SLURM task-count vars. WORLD_SIZE (set above) is what Lightning
# DDP uses for actual coordination — these vars are no longer needed.
for _var in (
    "SLURM_NTASKS",           # triggers _validate_srun_variables RuntimeError
    "SLURM_NPROCS",           # alias for SLURM_NTASKS on some clusters
    "SLURM_NTASKS_PER_NODE",  # triggers validate_settings ValueError
    "SLURM_STEP_NUM_TASKS",   # step-level alias
    "NCCL_NET",               # prevents 'Failed to initialize any NET plugin' for single node
):
    os.environ.pop(_var, None)
# -------------------------------------------------------------------------------------

import torch
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

import pandas as pd
import numpy as np
import time

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import ClusterEnvironment
from deepforest import main

# ---- Custom Cluster Environment ----
# We implement our own ClusterEnvironment to forcefully stop Lightning from
# guessing if it needs to launch subprocesses (which causes the EADDRINUSE error
# when multiple processes double-spawn and attempt to act as RANK 0).
class CustomSlurmEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        # Crucial: DO NOT let Lightning launch subprocesses. srun handles it.
        return True

    @property
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    def main_port(self) -> int:
        return int(os.environ.get("MASTER_PORT", 29500))

    @staticmethod
    def detect() -> bool:
        return True

    def set_global_rank(self, rank: int) -> None: pass
    def set_world_size(self, size: int) -> None: pass

    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", "1"))

    def global_rank(self) -> int:
        return int(os.environ.get("RANK", "0"))

    def local_rank(self) -> int:
        # We forced this to "0" in the bootstrap above because CUDA_VISIBLE_DEVICES
        # hides all other GPUs, leaving only index 0.
        return 0

    def node_rank(self) -> int:
        # DDP coordination only strictly needs global_rank and local_rank.
        return 0

print(f"VISIBLE CUDA DEVICES: {torch.cuda.device_count()}")

def run_training():
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    f = main.deepforest()
    f.load_model("weecology/deepforest-tree")

    TRAIN_CSV = "master_train.csv"
    VAL_CSV = "master_val.csv"
    IMAGE_ROOT_DIR = "all_chips"
    OUT_DIR = "deepforest_finetune"

    # Training config
    f.config.train.csv_file = TRAIN_CSV
    f.config.train.root_dir = IMAGE_ROOT_DIR

    f.config.train.epochs = 10
    f.config.train.lr = 1e-4

    # Data loading / performance
    # Adjusted to 4 to prevent CUDA OOM errors; scale up if VRAM permits.
    f.config.batch_size = 128
    f.config.workers = 8
    f.config.devices = 1  # each srun task is bound to 1 GPU via --gpus-per-task=1

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
    early_stop_cb = EarlyStopping(
        monitor="val_loss",   # monitor validation loss, not train loss
        patience=15,          # stop if no improvement for 15 epochs
        min_delta=0.005,      # minimum change to count as improvement
        mode="min",
        verbose=True,
    )


    # Increase the detection threshold to avoid ignored detections in dense images
    if hasattr(f, "mAP_metric"):
        f.mAP_metric.max_detection_thresholds = [1, 10, 100, 1000]
        f.mAP_metric.warn_on_many_detections = False

    # Initialize trainer with our extremely strict CustomSlurmEnvironment.
    # By passing CustomSlurmEnvironment:
    #   1. creates_processes_externally=True prevents the EADDRINUSE double-spawning bug.
    #   2. The precise RANK / LOCAL_RANK bindings are rigidly enforced.
    
    custom_env = CustomSlurmEnvironment()
    ddp = DDPStrategy(cluster_environment=custom_env, find_unused_parameters=False)

    # To satisfy Lightning's algebraic requirement (num_nodes * devices == trainer.world_size)
    # while keeping _AcceleratorConnector from violently protesting about missing local GPUs,
    # we trick Lightning into treating each individual process as an abstract 1-GPU "node".
    num_nodes_mock = int(os.environ.get("WORLD_SIZE", "1"))

    f.create_trainer(
        logger=logger,
        callbacks=[ckpt_cb, early_stop_cb],
        accelerator="gpu",
        devices=1,        # 1 GPU physically visible to each process via CUDA_VISIBLE_DEVICES
        num_nodes=num_nodes_mock, # Satisfies the trainer.world_size algebra
        strategy=ddp,
    )

    # Only rank 0 reports timing to avoid 4x duplicate output
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        print(f"[Rank {rank}] Training started...", flush=True)

    t_start = time.perf_counter()
    f.trainer.fit(f)
    t_end = time.perf_counter()

    if rank == 0:
        elapsed = t_end - t_start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"[Rank {rank}] Training complete. "
            f"Total time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s",
            flush=True,
        )


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True) is generally not required
    # when using standard DDP inside the __main__ block, but can be explicitly
    # called here if specific multiprocessing hangups occur with the dataloaders.

    run_training()