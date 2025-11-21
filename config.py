import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# data path configuration
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, "_cache")

# make sure the directory exists
# for dir_path in [DATA_DIR, CACHE_DIR]:
#     os.makedirs(dir_path, exist_ok=True)

# training parameters configuration
BATCH_SIZE = 16 
EPOCH = int
LEARNING_RATE = float
VALIDATION_INTERVAL = int

# data loader configuration
NUM_WORKERS = 4  # from 16 reduce to 4, lower CPU context switch overhead
PREFETCH_FACTOR = 2  # from 32 reduce to 2, lower memory footprint
PERSISTENT_WORKERS = True  # avoid worker recreation

# model parameters configuration
NUMBER_WORDS = 4097
NUMBER_POS = 70
EMBEDDING_DIM = 768
CNN_KERNEL_SIZE = 40
POOL_KERNEL_SIZE = 20
OUT_CHANNELS = 64

# sequence length configuration
MAX_ENHANCER_LENGTH = 1000  # fixed enhancer sequence length
MAX_PROMOTER_LENGTH = 4000  # fixed promoter sequence length

# feature dimension configuration
ENHANCER_FEATURE_DIM = (5, 3)  # 5 features, each feature 3 dimensions
PROMOTER_FEATURE_DIM = (5, 4)  # 5 features, each feature 4 dimensions


# file path configuration
EMBEDDING_MATRIX_PATH = os.path.join(PROJECT_ROOT, "save_model/dnabert.npy")

# cell line configuration
TRAIN_CELL_LINE = "ALL"  # select a single cell line or all cell lines
TEST_CELL_LINES = ["GM12878", "IMR90", "HeLa-S3", "HUVEC", "K562", "NHEK"]

# visualization configuration
TSNE_PERPLEXITY = 10
TSNE_RANDOM_STATE = 42

# device configuration
DEVICE = "cuda"  # prioritize GPU

# debug and logging configuration
DEBUG_MODE = False
SAVE_ATTENTION_OUTPUTS = False

# placeholder
