import os

# paths
DATA_PATH = os.path.join(os.path.curdir, 'data')
OUTPUT_PATH = os.path.join(os.path.curdir, 'output')
CHECKPOINT_PATH = os.path.join(os.path.curdir, 'checkpoint')
FIGURES_PATH = os.path.join(os.path.curdir, 'figures')
LOGS_PATH = os.path.join(os.path.curdir, 'logs')

USE_FRONT = False

# image processing
HU_LOWER = 100
HU_UPPER = 1500

# training
MAX_SIGMA = 10
MIN_SIGMA = 1.5 # było 3, ale zmieniłem na takie jak w publikacji
BATCH_SIZE = 8
ANYWHERE_RATE = 0.5 # prawdopodobieństwo, że wycinek na pewno bedzię zawierał krąg
X_DIST = 10 # maksymalna odległość środka wycinka od środka obrazu
OPTIMIZER = 'adam'
NUM_EPOCHS = 5 # change to 50!

INPUT_SHAPE = (256, 384, 1)

