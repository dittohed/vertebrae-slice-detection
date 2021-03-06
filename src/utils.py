import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgaug as ia

from . import config
from . import preprocessing

def y_to_keypoint(x, y):
    """
    Converts labels to imgaug's Keypoint objects
    (allows to automatically adjust and track them when augmenting).
    """

    keypoints = []

    for i in range(y.shape[0]):
        center = x[i].shape[1] // 2
        shape = x[i].shape + (1,)
        keypoint = ia.KeypointsOnImage([ia.Keypoint(x=center, y=y[i])], shape=shape)
        keypoints.append(keypoint)

    return keypoints

def y_to_onehot(y, input_shape):
    y_onehot = np.zeros(input_shape[0])

    if y < input_shape[0] and y >= 0:
        y_onehot[int(min(round(y), input_shape[0]-1))] = 1 # y jest floatem, jak to jest?

    return y_onehot

def save_orig_crop_comparison(img, label, crop_img, crop_label, img_idx):
    _, axs = plt.subplots(1, 2, figsize=(20, 20))
    for ax in axs:
        ax.axis('off')

    img = img.copy()
    img[label, :] = 255
    axs[0].imshow(img, vmin=0, vmax=255, cmap='gray')
    axs[0].set_title('Original')

    if crop_label >= 0 and crop_label < crop_img.shape[0]:
        crop_img = crop_img.copy()
        crop_img[crop_label, :] = 255
        axs[1].imshow(crop_img, vmin=0, vmax=255, cmap='gray')
        axs[1].set_title('Crop')
    else:
        axs[1].imshow(crop_img, vmin=0, vmax=255, cmap='gray')
        axs[1].set_title('Crop (no label)')

    plt.savefig(os.path.join(config.OUTPUT_PATH, f'{img_idx}.png'))
    plt.close()

def save_orig_aug_comparison(img, label, aug_img, aug_label, img_idx):
    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    for ax in axs:
        ax.axis('off')

    img = img.copy()
    img[label, :] = 255
    axs[0].imshow(img, vmin=0, vmax=255, cmap='gray')
    axs[0].set_title('Original')

    aug_label = np.nonzero(aug_label)[0]
    if aug_label.shape == (0,):
        axs[1].imshow(aug_img, vmin=0, vmax=255, cmap='gray')
        axs[1].set_title('Augmented crop (no label)')
    else:
        aug_img = aug_img.copy()
        aug_img[aug_label, :] = 255
        axs[1].imshow(aug_img, vmin=0, vmax=255, cmap='gray')
        axs[1].set_title('Augmented crop')

    plt.savefig(os.path.join(config.OUTPUT_PATH, f'{img_idx}.png'))
    plt.close()

def prepare_for_inference(x, y=None, to_32=False):
    """
    Prepares data for model's inference 
    (same as training data generator processing).
    Takes 2D images and integer labels,
    returns 3D (with dummy color channel) images and one-hot labels.
    """

    y_new = []
    for i in range(x.shape[0]):
        # both width and height should be divisible by 32 (maxpooling and concats)
        # and not less than training crops size

        if to_32:
            new_height = int(
                max(np.ceil((x[i].shape[0] / 32)) * 32, config.INPUT_SHAPE[0]))
            new_width = int(
                max(np.ceil((x[i].shape[1] / 32)) * 32, config.INPUT_SHAPE[1]))
            x[i], y[i] = preprocessing.pad_img(x[i], y[i], (new_height, new_width))
        else:
            if x[i].shape[0] < config.INPUT_SHAPE[0] or x[i].shape[1] < config.INPUT_SHAPE[1]:
                x[i], y[i] = preprocessing.pad_img(x[i], y[i], config.INPUT_SHAPE)

        # add dummy color channel
        x[i] = np.expand_dims(x[i], 2).astype(np.float32)
        y_new.append(
            np.expand_dims(y_to_onehot(y[i], x[i].shape), 1)
        )

        # grayscale to RGB
        if config.RGB:
            x[i] = np.repeat(x[i], 3, axis=-1)

    # pixel values to [-1, 1]
    x = (x / 255) * 2 - 1

    return x, np.array(y_new, dtype=object)

def plot_learning(history, plot_name):
    """
    Plots learning history information and saves the figure under a specified name.
    """

    df = pd.DataFrame(history.history)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # fig.tight_layout(pad=3)

    axs[0].plot('loss', label='loss', data=df)
    axs[0].plot('val_loss', label='val. loss', data=df)
    axs[0].set_title(f'Loss vs. epochs (min. val.: {df["val_loss"].min()})')
    axs[0].set_ylabel('Loss')

    axs[1].plot('distance', '.', label='distance', data=df)
    axs[1].plot('val_distance', '.', label='val. distance', data=df)
    axs[1].set_title(f'Distance vs. epochs (min. val.: {df["val_distance"].min()})')
    axs[1].set_ylabel('Distance')

    for ax in axs:
        ax.set_xlabel('Epochs')
        ax.legend()

    plt.savefig(os.path.join(config.FIGURES_PATH, f'{plot_name}.png'))
    plt.close()

def median(y_true, y_pred, in_crops=False, thicks=None):
    """
    Returns a median of distances between real value labels.
    """
        
    errors = np.abs(y_true-y_pred)
    if in_crops:
        errors = errors / thicks     

    return np.median(errors)

def save_imgs_dist(imgs, title):
    """
    Saves a scatterplot of images dimensions (width and heights).
    """

    heights = []
    widths = []

    for i, img in enumerate(imgs):
        if i == 0:
            print(img.shape[0])
            print(img.shape[1])
            plt.imshow(img, cmap='gray')
            plt.show() 

        heights.append(img.shape[0])
        widths.append(img.shape[1])

    plt.scatter(widths, heights)
    plt.xlabel('Widths [px=mm]')
    plt.ylabel('Heights [px=mm]')
    plt.title('Loaded images dimensions')
    plt.savefig(os.path.join(config.FIGURES_PATH, f'{title}.png'))
    plt.close()

def distance(y_true, y_pred, in_crops=False, thicks=None):
    """
    Returns a mean distance between real value labels along with std.
    """

    errors = np.abs(y_true-y_pred)
    if in_crops:
        errors = errors / thicks

    return np.mean(errors), np.std(errors)
