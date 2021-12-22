import os

# training
EXP_NAME = 'kanavati_l3_sagittal_bce_1' # TODO: set
LOSS = 'BCE' # 'BCE' or 'Focal'
MODEL_NAME = 'Kanavati' # 'Kanavati' or 'Efficient' or 'Own'
INPUT_SHAPE = (256, 384, 1)
BATCH_SIZE = 8
ANYWHERE_RATE = 1 # 0.5 -> 1.0; prawdopodobieństwo, że wycinek może nie zawierać kręgu
OPTIMIZER = 'adam'
NUM_EPOCHS = 80

# image processing
V_LEVEL = 'L3'
USE_FRONT = False
RGB = False # whether to triple gray channel (for EFficientNet)
USE_OVERSAMPLING = False
HU_LOWER = 100
HU_UPPER = 1500

# inference
THRESHOLD = 0.5

# paths
DATA_PATH = os.path.join(os.path.curdir, 'data')
OUTPUT_PATH = os.path.join(os.path.curdir, 'output')
CHECKPOINT_PATH = os.path.join(os.path.curdir, 'checkpoints')
FIGURES_PATH = os.path.join(os.path.curdir, 'figures')
LOGS_PATH = os.path.join(os.path.curdir, 'logs')