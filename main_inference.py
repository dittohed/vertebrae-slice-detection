# coś nie tak z checkpointem?

import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

import src.config as config
import src.generator as generator
import src.utils as utils
import src.preprocessing as preprocessing
import src.models as models

if __name__ == '__main__':
    model = models.get_model()
    model.load_weights(os.path.join(config.CHECKPOINT_PATH, 'train0', 'model'))

    x, y, spacings, names = utils.load_data(config.USE_FRONT)
    x, y = preprocessing.normalize_data(x, y, spacings)

    x_train, y_train, names_train, x_val, y_val, names_val = utils.split_data(x, y, names)

    # random crop for each val image
    gen_val = generator.DataGenerator(x_val[:8], y_val[:8], validation=True,
                                input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE)
    for x_val_crop, y_val_crop in gen_val:
        break

    # full-size val images
    x_val, y_val = utils.prepare_for_inference(x_val[:8].copy(), y_val[:8].copy()) 
    idx = 0
    
    # compare (crop)
    y_pred = model.predict(x_val_crop)
    # y_pred[y_pred <= 0.5] = 0

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
        title = f'True: {np.argmax(sample_true)}, pred: {np.argmax(sample_pred)}'
    elif not np.any(sample_pred) and not np.any(sample_true):
        title = 'No true value and no prediction'
    elif not np.any(sample_pred):
        title = f'True: {np.argmax(sample_true)}, no prediction'
    elif not np.any(sample_true):
        title = f'Pred: {np.argmax(sample_pred)}, no true'

    plt.imshow(sample_img, cmap='gray')
    plt.title(title)
    plt.savefig('./output/tmp/crop.png')
    plt.close() 

    # compare (full-size)
    y_pred = models.predict_whole(model, x_val)

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
    plt.savefig('./output/tmp/full.png')
    plt.close() 


            