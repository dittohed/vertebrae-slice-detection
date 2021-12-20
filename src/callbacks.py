import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

import cv2
import numpy as np

from . import config

def get_callbacks(subdir):
    """
    Returns list of callbacks for training.
    """

    es = EarlyStopping(monitor='val_distance', mode='min', 
                                  patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_distance', mode='min', 
                                  factor=0.5, patience=5, verbose=1)
    checkpoint = ModelCheckpoint(monitor='val_distance', mode='min', save_weights_only=True,
                                  filepath=os.path.join(config.CHECKPOINT_PATH, subdir, 'model'), 
                                  save_best_only=True, verbose=1)
    csv_logger = CSVLogger(os.path.join(config.LOGS_PATH, subdir+'.csv'))

    return [es, reduce_lr, checkpoint, csv_logger]

class PreviewOutput(Callback):
    """
    Callback for visualizing predictions for validation data
    on each epoch end.
    """

    def __init__(self, x, y, subdir):
        """
        Takes full-size images preprocessed by 
        utils.prepare_for_inference function.
        """

        super().__init__()
        self.x = x
        self.y = y
        self.subdir = subdir

    def on_epoch_end(self, epoch, logs=None):
        print('Saving output preview...')

        for i, (img, y_true) in enumerate(zip(self.x, self.y)):
            y_pred = self.model.predict(np.expand_dims(img, 0))

            # revert [-1, 1] scaling
            img = img.copy()
            img = (img + 1) / 2 * 255

            # convert to rgb (reverse order an account of using imwrite)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # overlay prediction on image (red)
            img[np.argmax(y_pred), :, 2] = 255

            # overlay true label on image (blue)
            img[np.argmax(y_true), :, 0] = 255
            
            # overlay green if y_true == y_pred
            if np.argmax(y_pred) == np.argmax(y_true):
                img[np.argmax(y_true), :, :] = 0
                img[np.argmax(y_true), :, 1] = 255

            path = os.path.join(config.OUTPUT_PATH, 'preview_output', 
                                              self.subdir, f'epoch_{epoch}')
            if not os.path.exists(path):
                os.makedirs(path)

            cv2.imwrite(os.path.join(path, f'{i}.png'), img)
