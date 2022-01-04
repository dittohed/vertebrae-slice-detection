"""
For sole inference, set INF_V_LEVEL, INF_DS_NAME and VISUALIZE. 
"""

import os

# inference
INF_V_LEVEL = 'T12' # 'L3' or 'T12' - set for making inference
INF_DS_NAME = '' # .npz archive name with data for inference, stored in data/ directory  
VISUALIZE = True # whether to visualize predictions (stored in output)

# image processing
V_LEVEL = 'T12' # used by loaders.py
USE_FRONT = True # whether to use 'imgs_s' or 'imgs_f' arrays as input
RGB = False # whether to triple gray channel (for EFficientNet only)
HU_LOWER = 100
HU_UPPER = 1500

# training
EXP_NAME = 'example_training' # set for each training experiment!
LOSS = 'BCE' # 'BCE' or 'Focal'
MODEL_NAME = 'Kanavati' # 'Kanavati' or 'Efficient' or 'Own'
USE_IMAGENET = False # whether to use EfficientNet with ImageNet weights
LR = 1e-3
INPUT_SHAPE = (256, 384, 1) # not tested for different input shapes
BATCH_SIZE = 8
ANYWHERE_RATE = 1 # 0.5 -> 1.0; probability of crop not containing V_LEVEL vertebrae (negative sample)
NUM_EPOCHS = 80

# paths
DATA_PATH = os.path.join(os.path.curdir, 'data')
OUTPUT_PATH = os.path.join(os.path.curdir, 'output')
CHECKPOINT_PATH = os.path.join(os.path.curdir, 'checkpoints')
FIGURES_PATH = os.path.join(os.path.curdir, 'figures')
LOGS_PATH = os.path.join(os.path.curdir, 'logs')
