"""
Script for performing inference on new data.
"""

import os

import cv2
import numpy as np
import pandas as pd

import src.config as config
import src.loaders as loaders
import src.models as models

if __name__ == '__main__':
    model = models.get_model('Kanavati')

    if config.INF_V_LEVEL == 'L3':
        subdir = 'kanavati_l3_frontal_bce_2'
    elif config.INF_V_LEVEL == 'T12':
        subdir = 'kanavati_t12_frontal_bce_pretrained_1'

    model.load_weights(os.path.join(config.CHECKPOINT_PATH, subdir, 'model'))

    x, ids, thicks = loaders.get_inference_data()

    # --- inference using whole images ---
    y_pred = []
    for img in x:
        y_pred.append(model.predict(np.expand_dims(img, axis=0)))

    y_pred = np.array([np.argmax(y) for y in y_pred])
    
    # --- storing results ---
    if not os.path.isdir(config.OUTPUT_PATH):
        os.mkdir('output')

    if config.VISUALIZE:
        for i in range(x.shape[0]):
            img = x[i]
            pred = y_pred[i]

            img = (img + 1) / 2 * 255
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # overlay prediction on image (blue)
            img[pred, :, 0] = 255

            cv2.imwrite(os.path.join(config.OUTPUT_PATH, f'{ids[i]}.png'), img)

    y_pred = np.round(y_pred/thicks) # mm to slices

    df = pd.DataFrame({
        'exam_id': ids,
        f'{config.INF_V_LEVEL}_slice_num': y_pred
    })
    df.to_csv(os.path.join(config.OUTPUT_PATH, f'{config.INF_V_LEVEL}_slices.csv'))