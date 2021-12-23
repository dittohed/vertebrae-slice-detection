import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

import src.config as config
import generators as generators
import src.utils as utils
import src.loaders as loaders
import src.models as models
import src.metrics as metrics

if __name__ == '__main__':
    model = models.get_model('Kanavati')
    model.load_weights(os.path.join(config.CHECKPOINT_PATH, 'kanavati_l3_frontal_bce_oversamp_1', 'model')) # TODO: set subdir

    x_train, x_val, y_train, y_val = loaders.get_data_l3()
    
    # --- preparation for inference ---
    # whole val images 
    # TODO: later choose one and del copy()

    x_val32, y_val32 = utils.prepare_for_inference(x_val.copy(), y_val.copy(), to_32=True)
    x_val, y_val = utils.prepare_for_inference(x_val, y_val)

    # --- evaluation ---
    # crop-by-crop evaluation
    y_pred = models.predict_whole(model, x_val)
    y_true = np.array([np.argmax(y) for y in y_val])
    print(f'Crop-by-crop mean distance [mm]: {utils.distance(y_true, y_pred)}')

    for idx in range(x_val.shape[0]):
        sample_img = x_val[idx]
        sample_pred = y_pred[idx]
        sample_true = y_val[idx]

        sample_img = (sample_img + 1) / 2 * 255
        sample_img = cv2.cvtColor(sample_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if sample_pred != -1:
            # overlay prediction on image (red)
            sample_img[sample_pred, :, 0] = 255
        if np.any(sample_true):
            # overlay true label on image (blue)
            sample_img[np.argmax(sample_true), :, 2] = 255

        if sample_pred != -1 and np.any(sample_true):
            title = f'True: {np.argmax(sample_true)}, pred: {sample_pred}'
        elif sample_pred == -1 and not np.any(sample_true):
            title = 'No true value and no prediction'
        elif sample_pred == -1:
            title = f'True: {np.argmax(sample_true)}, no prediction'
        elif not np.any(sample_true):
            title = f'Pred: {sample_pred}, no true'

        plt.imshow(sample_img, cmap='gray')
        plt.title(title)
        plt.savefig(f'./output/tmp/{idx}crop-by-crop.png')
        plt.close() 

    # whole images (as U-Net input) evaluation
    y_pred = []
    for img in x_val32:
        y_pred.append(model.predict(np.expand_dims(img, axis=0)))

    y_true = np.array([np.argmax(y) for y in y_val32])
    print(f'Whole images mean distance [mm]: {utils.distance(y_true, np.array([np.argmax(y) for y in y_pred]))}')

    for idx in range(x_val32.shape[0]):
        sample_img = x_val32[idx]
        sample_pred = y_pred[idx]
        sample_true = y_val32[idx]

        sample_img = (sample_img + 1) / 2 * 255
        sample_img = cv2.cvtColor(sample_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if np.any(sample_pred):
            # overlay prediction on image (red)
            sample_img[np.argmax(sample_pred), :, 0] = 255
        if np.any(sample_true):
            # overlay true label on image (blue)
            sample_img[np.argmax(sample_true), :, 2] = 255

        if np.any(sample_pred) and np.any(sample_true):
            title = f'True: {np.argmax(sample_true)}, pred: {np.argmax(sample_pred)}, confidence: {np.amax(sample_pred)}'
        elif not np.any(sample_pred) and not np.any(sample_true):
            title = 'No true value and no prediction'
        elif not np.any(sample_pred):
            title = f'True: {np.argmax(sample_true)}, no prediction'
        elif not np.any(sample_true):
            title = f'Pred: {np.argmax(sample_pred)}, no true'

        plt.imshow(sample_img, cmap='gray')
        plt.title(title)
        plt.savefig(f'./output/tmp/{idx}whole.png')
        plt.close() 
            