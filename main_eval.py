"""
Script for evaluating model on test set.
"""

import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

import src.config as config
import src.utils as utils
import src.loaders as loaders
import src.models as models

if __name__ == '__main__':
    model = models.get_model(config.MODEL_NAME)

    if config.V_LEVEL == 'L3':
        subdir = 'kanavati_l3_frontal_bce_2'
        ds_name = 'test_l3_frontal.npz'
    elif config.V_LEVEL == 'T12':
        subdir = 'kanavati_t12_frontal_bce_pretrained_1'
        ds_name = 'test_t12_frontal.npz'
        
    model.load_weights(os.path.join(config.CHECKPOINT_PATH, subdir, 'model'))
    x_test, y_test, ids_test, thicks_test = loaders.get_test_data(ds_name)
    x_test, y_test = utils.prepare_for_inference(x_test, y_test, to_32=True)

    # whole images (as U-Net input) evaluation
    y_pred = []
    for img in x_test:
        pred = model.predict(np.expand_dims(img, axis=0))
        y_pred.append(pred)

    y_true = np.array([np.argmax(y) for y in y_test])
    y_conf = np.array([np.amax(y) for y in y_pred])
    y_pred = np.array([np.argmax(y) for y in y_pred])

    print(f'Whole images distance mean and std [mm]: {utils.distance(y_true, y_pred)}')
    print(f'Whole images median [mm]: {utils.median(y_true, y_pred)}')
    print(f'Whole images distance mean and std [crops]: {utils.distance(y_true, y_pred, in_crops=True, thicks=thicks_test)}')
    print(f'Whole images median [crops]: {utils.median(y_true, y_pred, in_crops=True, thicks=thicks_test)}')

    # preds visualization
    if not os.path.isdir(config.OUTPUT_PATH):
        os.mkdir('output')

    for idx in range(x_test.shape[0]):
        sample_img = x_test[idx]
        sample_pred = y_pred[idx]
        sample_true = y_true[idx]

        sample_img = (sample_img + 1) / 2 * 255
        sample_img = cv2.cvtColor(sample_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # overlay prediction on image (red)
        sample_img[sample_pred, :, 0] = 255
        # overlay true label on image (blue)
        sample_img[sample_true, :, 2] = 255

        title = f'True: {sample_true}, pred: {sample_pred}, confidence: {y_conf[idx]}'

        plt.imshow(sample_img)
        plt.title(title)
        plt.savefig(os.path.join(config.OUTPUT_PATH, f'{config.V_LEVEL}_{idx}.png'))
        plt.close() 
            
