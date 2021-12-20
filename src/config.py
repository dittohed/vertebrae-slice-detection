import os

# paths
DATA_PATH = os.path.join(os.path.curdir, 'data')
OUTPUT_PATH = os.path.join(os.path.curdir, 'output')
CHECKPOINT_PATH = os.path.join(os.path.curdir, 'checkpoints')
FIGURES_PATH = os.path.join(os.path.curdir, 'figures')
LOGS_PATH = os.path.join(os.path.curdir, 'logs')

# image processing
V_LEVEL = 'T12'
USE_FRONT = True
HU_LOWER = 100
HU_UPPER = 1500
RGB = False # whether to triple gray channel (for EFficientNet)
USE_OVERSAMPLING = False

# training
EXP_NAME = '' # TODO: set
INPUT_SHAPE = (256, 384, 1)
MODEL_NAME = 'Kanavati'
BATCH_SIZE = 8
ANYWHERE_RATE = 1 # 0.5 -> 1.0; prawdopodobieństwo, że wycinek może nie zawierać kręgu
OPTIMIZER = 'adam'
NUM_EPOCHS = 50
USE_PRETRAINED = False

# inference
THRESHOLD = 0.5

