import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

import src.config as config
import src.generators as generators
import src.utils as utils
import src.loaders as loaders
import src.models as models
import src.metrics as metrics

if __name__ == '__main__':
    model = models.get_model(config.MODEL_NAME)

    if config.V_LEVEL == 'L3':
        subdir = 'kanavati_l3_sagittal_bce_oversamp_2'
        ds_name = 'test_l3_sagittal.npz'
    elif config.V_LEVEL == 'T12':
        subdir = '' # TODO
        ds_name = 'test_t12_sagittal.npz'
        
    model.load_weights(os.path.join(config.CHECKPOINT_PATH, subdir, 'model'))
    x_test, y_test, ids_test, thicks_test = loaders.get_test_data(ds_name)
    x_test, y_test = utils.prepare_for_inference(x_test, y_test, to_32=True)

    # whole images (as U-Net input) evaluation
    y_pred = []
    for img in x_test:
        y_pred.append(model.predict(np.expand_dims(img, axis=0)))

    y_true = np.array([np.argmax(y) for y in y_test])
    print(f'Whole images distance mean [mm]: {utils.distance(y_true, np.array([np.argmax(y) for y in y_pred]))}')
    print(f'Whole images median [mm]: {utils.median(y_true, np.array([np.argmax(y) for y in y_pred]))}')
    # TODO: dać do pliku .csv id i numery przekrojów, ale to w inference
            