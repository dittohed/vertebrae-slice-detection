import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

import src.config as config
import src.generator as generator
import src.utils as utils
import src.preprocessing as preprocessing
import src.models as models
import src.metrics as metrics

if __name__ == '__main__':
    model = models.get_model('Kanavati')
    model.load_weights(os.path.join(config.CHECKPOINT_PATH, 'crops', 'model'))

    x, y, spacings, names = utils.load_data(config.USE_FRONT)
    x, y, slices, heights = preprocessing.normalize_data(x[:64], y[:64], spacings)
    x_train, y_train, \
        x_val, y_val, spacings_val, slices_val, heights_val = utils.split_data(x, y, spacings, slices, heights)

    # --- data preparation ---
    # random crop for each val image
    gen_val = generator.DataGenerator(x_val.copy(), y_val.copy(), validation=True,
                                input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE)
    for x_val_crop, y_val_crop in gen_val: # TODO: zamienić batch na całą epokę
        break

    # whole val images 
    x_val, y_val = utils.prepare_for_inference(x_val, y_val) 
    
    # --- evaluation ---
    # random crop evaluation
    y_pred = model.predict(x_val_crop)
    print(f'Crop mean distance [mm]: {np.mean(metrics.distance(y_val_crop, y_pred))}')

    for idx in range(16):
        sample_img = x_val_crop[idx]
        sample_pred = y_pred[idx]
        sample_true = y_val_crop[idx]

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
        plt.savefig(f'./output/tmp/{idx}crop.png')
        plt.close() 

    # whole images (crop-by-crop) evaluation
    y_pred = models.predict_whole(model, x_val)
    y_true = np.array([np.argmax(y) for y in y_val])
    print(f'Crop-by-crop mean distance [mm]: {utils.distance(y_true, y_pred)}')

    for idx in range(16):
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
    for img in x_val:
        y_pred.append(model.predict(np.expand_dims(img, axis=0)))

    y_true = np.array([np.argmax(y) for y in y_val])
    print(f'Whole images mean distance [mm]: {utils.distance(y_true, np.array([np.argmax(y) for y in y_pred]))}')

    for idx in range(16):
        sample_img = x_val[idx]
        sample_pred = y_pred[idx]
        sample_true = y_val[idx]

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
            