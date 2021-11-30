import os

from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import imgaug as ia

from . import config
from . import preprocessing

def load_data(use_front):
    """
    Loads and returns all the needed data.
    """

    # loads dict of ndarrays containing unnormalized MIPs
    data = np.load(os.path.join(config.DATA_PATH, 'l3_dataset.npz'), allow_pickle=True)
    
    if use_front:
        x = data['images_f']
    else:
        x = data['images_s']
    # x contains images of different shapes

    names = data['names'] # examination names
    y = np.zeros_like(names, dtype=np.float32) # array of shape of ex_names
    y_data = data['ydata'] # for loading labels into y
    spacings = data['spacings'] # triplet for each image

    data.close()

    # TODO: so far, some weird way of loading labels
    for _, v in y_data.item().items():
        y += v
    y /= len(y_data.item()) # diving by 2

    return x, y, spacings, names

def save_preprocessed(x, y, names):
    """
    Saves preprocessed images to .npz file.
    """

    np.savez_compressed(os.path.join(config.DATA_PATH, 'preprocessed'),
                        x=x, y=y, names=names)

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

# TODO; delete
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

# TODO; delete
def save_orig_aug_comparison(img, label, aug_img, aug_label, img_idx):
    # fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    # for ax in axs:
    #     ax.axis('off')

    # img = img.copy()
    # img[label, :] = 255
    # axs[0].imshow(img, vmin=0, vmax=255, cmap='gray')
    # axs[0].set_title('Original')

    # aug_label = np.nonzero(aug_label)[0]
    # # overlay = np.zeros_like(aug_img)
    # # overlay[aug_label, :] = 255
    # if aug_label.shape == (0,):
    #     axs[1].imshow(aug_img, vmin=0, vmax=255, cmap='gray')
    #     axs[1].set_title('Augmented crop (no label)')
    # else:
    #     aug_img = aug_img.copy()
    #     # aug_img[aug_label[0], :] = 255
    #     aug_img[aug_label, :] = 255
    #     axs[1].imshow(aug_img, vmin=0, vmax=255, cmap='gray')
    #     axs[1].set_title('Augmented crop')

    # plt.savefig(os.path.join(config.OUTPUT_PATH, f'{img_idx}.png'))

    aug_label = np.nonzero(aug_label)[0]
    aug_img = aug_img.copy()
    aug_img[aug_label, :] = 255
    cv2.imwrite(os.path.join(config.OUTPUT_PATH, f'{img_idx}.png'), aug_img)

def split_data(x, y, names):
    rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    for train_idx, val_idx in rs.split(list(range(len(x)))):
        pass

    x_train = x[train_idx]
    y_train = y[train_idx]
    names_train = names[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    names_val = names[val_idx]

    return x_train, y_train, names_train, x_val, y_val, names_val

def prepare_for_inference(x, y):
    """
    Prepares data for model's inference (same as training data processing).
    Takes 2D images and integer labels,
    returns 3D images and one-hot labels.
    """

    y_new = []
    for i in range(x.shape[0]):
        # both width and height should be dividible by 32 (maxpooling and concats)
        new_height = np.ceil((x[i].shape[0] / 32)) * 32
        new_width = np.ceil((x[i].shape[1] / 32)) * 32

        # if too small
        # TODO

        x[i] = preprocessing.pad_img(x[i], (new_height, new_width)) 

        # add dummy color channel
        x[i] = np.expand_dims(x[i], 2).astype(np.float32)
        y_new.append(
            np.expand_dims(y_to_onehot(y[i], x[i].shape), 1)
        )

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

def median(model, x, y_true):
    """
    Calculates and returns median of error 
    given trained model and instance of InferenceDataGenerator.
    """

    # doing loops below due to different img sizes

    y_pred = []
    for img in x:
        y_pred.append(
          np.argmax(model.predict(np.expand_dims(img, axis=0)), axis=1))
    y_pred = np.array(y_pred).flatten()

    y_true = []
    for label in y:
      y_true.append(
        np.argmax(label, axis=0)
      )
    y_true = np.array(y_true).flatten()

    errors = np.abs(y_pred-y_true)
    return np.median(errors)
