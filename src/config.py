import os

# paths
DATA_PATH = os.path.join(os.path.curdir, 'data')
OUTPUT_PATH = os.path.join(os.path.curdir, 'output')
CHECKPOINT_PATH = os.path.join(os.path.curdir, 'checkpoints')
FIGURES_PATH = os.path.join(os.path.curdir, 'figures')
LOGS_PATH = os.path.join(os.path.curdir, 'logs')

# image processing
USE_FRONT = False
HU_LOWER = 100
HU_UPPER = 1500
RGB = True # whether to triple gray channel (for EFficientNet)

# training
MODEL_NAME = 'Efficient'
MAX_SIGMA = 10
MIN_SIGMA = 1.5 # było 3, ale zmieniłem na takie jak w publikacji
BATCH_SIZE = 8
ANYWHERE_RATE = 1 # 0.5 -> 1.0; prawdopodobieństwo, że wycinek może nie zawierać kręgu
X_DIST = 10 # maksymalna odległość środka wycinka od środka obrazu
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy' # 'binary_crossentropy' or 'categorical_crossentropy'
ACTIVATION = 'sigmoid' # 'softmax' or 'sigmoid'
NUM_EPOCHS = 50

# other
INPUT_SHAPE = (256, 384, 1)
THRESHOLD = 0.5

