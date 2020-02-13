from yacs.config import CfgNode
from calamari_ocr.proto import DataPreprocessorParams
import calamari_ocr

_C = CfgNode()

_C.VERSION = calamari_ocr.__version__
# Seed for random operations. If negative or zero a 'random' seed is used"
_C.SEED = 16
# The number of threads to use for all operation
_C.NUM_THREADS = 1

# Default directory where to store checkpoints and models
_C.OUTPUT_DIR = "run_0"
# Prefix for storing checkpoints and models
_C.OUTPUT_MODEL_PREFIX = 'model_'
# The prefix of the best model using early stopping
_C.EARLY_STOPPING_BEST_MODEL_PREFIX = 'best'
# Path where to store the best model. Default is output_dir
_C.EARLY_STOPPING_BEST_MODEL_OUTPUT_DIR = False

# Frequency of how often an output shall occur during training. If 0 < display <= 1 
# the display is in units of epochs
_C.DISPLAY = 100
# Average this many iterations for computing an average loss, label error rate and 
# training time
_C.STATS_SIZE = 100
# Do not show any progress bars
_C.NO_PROGRESS_BAR = False

_C.INPUT = CfgNode()
# The line height
_C.INPUT.LINE_HEIGHT = 48
# Padding (left right) of the line
_C.INPUT.PAD = 16

_C.INPUT.N_AUGMENT= 0
# Text regularization to apply.
_C.INPUT.TEXT_REGULARIZATION = ["extended"]
# Unicode text normalization to apply. Defaults to NFC
_C.INPUT.TEXT_NORMALIZATION = "NFC"
# Data Processing
_C.INPUT.DATA_PREPROCESSING = [DataPreprocessorParams.DEFAULT_NORMALIZER]
# Bidirectional Text
_C.INPUT.BIDI_DIR = False

_C.DATASET = CfgNode()
# List all image files that shall be processed. Ground truth fils with the same 
# base name but with '.gt.txt' as extension are required at the same location
_C.DATASET.TRAIN = CfgNode()
_C.DATASET.TRAIN.PATH = []
# Optional list of GT files if they are in other directory
_C.DATASET.TRAIN.TEXT_FILES = False
# Default extension of the gt files (expected to exist in same dir)
_C.DATASET.TRAIN.GT_EXTENSION = False
# Type of dataset
_C.DATASET.TRAIN.TYPE = 1

_C.DATASET.VALID = CfgNode()
# Validation line files used for early stopping
_C.DATASET.VALID.PATH = []
# Optional list of GT files if they are in other directory
_C.DATASET.VALID.TEXT_FILES = False
# Default extension of the gt files (expected to exist in same dir)
_C.DATASET.VALID.GT_EXTENSION = False
# Type of dataset
_C.DATASET.VALID.TYPE = 1


_C.DATALOADER = CfgNode()
# Instead of preloading all data during the training, load the data on the fly. 
# This is slower, but might be required for limited RAM or large datasets
_C.DATALOADER.TRAIN_ON_THE_FLY = False
# Instead of preloading all data during the training, load the data on the fly. 
# This is slower, but might be required for limited RAM or large datasets
_C.DATALOADER.VALID_ON_THE_FLY = False
# Do no skip invalid gt, instead raise an exception.
_C.DATALOADER.NO_SKIP_INVALID_GT = True
# Amount of data augmentation per line (done before training). If this number is < 1 
# the amount is relative.
_C.DATALOADER.ONLY_TRAIN_ON_AUGMENTED = False
# Number of examples in the shuffle buffer for training (default 1000). A higher number 
# required more memory. If set to 0, the buffer size equates an epoch i.e. the full dataset
_C.DATALOADER.SHUFFLE_BUFFER_SIZE = 1000


_C.MODEL = CfgNode()
# The network structure
_C.MODEL.NETWORK = "cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5"
# Load network weights from the given file.
_C.MODEL.WEIGHTS = ""
_C.MODEL.CODEX = CfgNode()
# Do not compute the codec automatically. See also whitelist
_C.MODEL.CODEX.SEE_WHITELIST = False
# Whitelist of txt files that may not be removed on restoring a model
_C.MODEL.CODEX.WHITELIST_FILES = []
# Whitelist of characters that may not be removed on restoring a model. 
# For large datasets you can use this to skip the automatic codec computation
# (see --no_auto_compute_codec)
_C.MODEL.CODEX.WHITELIST = []
# Fully include the codec of the loaded model to the new codec
_C.MODEL.CODEX.KEEP_LOADED_CODEC = False

_C.SOLVER = CfgNode()
#  Learning rate
_C.SOLVER.LR = 0.001
# The batch size to use for training
_C.SOLVER.BATCH_SIZE = 5
# The number of iterations for training. 
# If using early stopping, this is the maximum number of iterations
_C.SOLVER.MAX_ITER = 1000000
# Stop training if the early stopping accuracy reaches this value
_C.SOLVER.EARLY_STOPPING_AT_ACC = 0
# The frequency how often to write checkpoints during training. If 0 < value <= 1 the 
#  unit is in epochs, thus relative to the number of training examples."
# If -1, the early_stopping_frequency will be used
_C.SOLVER.CHECKPOINT_FREQ = -1
# Clipping constant of the norm of the gradients.
_C.SOLVER.GRADIENT_CLIPPING_NORM = 5
# The frequency of early stopping. By default the checkpoint frequency uses the early 
# stopping frequency. By default (negative value) the early stopping frequency is 
# approximated as a half epoch time (if the batch size is 1). 
# If 0 < value <= 1 the frequency has the unit of an epoch (relative to the 
# number of training data).
_C.SOLVER.EARLY_STOPPING_FREQ = -1
# The number of models that must be worse than the current best model to stop
_C.SOLVER.EARLY_STOPPING_NBEST = 5

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    return _C.clone()