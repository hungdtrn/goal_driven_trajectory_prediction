
import os
from pathlib import Path


PROJECT_PATH = Path(Path(__file__).absolute()).parent.parent

LOG_PATH = os.path.join(PROJECT_PATH, "reports", "train_logs")
TENSORBOARD_PATH = os.path.join(PROJECT_PATH, "reports", "tensorboards")
CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints")

DATA_PATH = os.path.join(PROJECT_PATH, "data")
TOOLPATH = os.path.join(PROJECT_PATH,
                        "src", "tools", "segmentation")
