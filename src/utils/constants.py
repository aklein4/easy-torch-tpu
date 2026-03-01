
import os
import dotenv

try:
    import torch_xla.runtime as xr
    XLA_AVAILABLE = True # whether torch-xla can be accessed
except ImportError:
    print("Warning: torch_xla not found", flush=True)
    XLA_AVAILABLE = False

# the path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)
# path of the repo (easy-torch-tpu)
REPO_PATH = os.path.dirname(BASE_PATH)

# load environment variables from .env file
dotenv.load_dotenv(os.path.join(REPO_PATH, ".env"))

# id of the current device
PROCESS_INDEX = lambda: xr.process_index()

# whether this is the main process on its device
PROCESS_IS_MAIN = lambda: xr.process_index() == 0

# total number of processes/devices
PROCESS_COUNT = lambda: xr.process_count()

# path to local data folder
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# path to checkpoint folder
CHECKPOINTS_PATH = os.path.join(LOCAL_DATA_PATH, "checkpoints")

# modules for dynamic importing
MODEL_MODULE = "models"
TRAINER_MODULE = "trainers"
COLLATOR_MODULE = "collators"
OPTIMIZER_MODULE = "optimizers"

# huggingface login id
HF_ID = os.getenv("HF_ID", None)
