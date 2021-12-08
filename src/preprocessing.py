import os

import numpy as np
import cv2
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter1d
from imgaug import augmenters as iaa

from . import config

def normalize_data(x, y, spacings):
    """
    Normalizes images spatially and in regard to pixel values.
    """

    x_norm = [] 
    y_norm = []
    max_heights = [] # for transforming milimeters back to slice number during inference

    for i in range(y.shape[0]):
        img = zoom(x[i], [spacings[i][2], spacings[i][0]])
        img = reduce_hu_scale(img)
        x_norm.append(img)

        # y_norm.append(int(y[i]*spacings[i][2]))
        y_norm.append(
            int(min(np.round(y[i]*spacings[i][2]), img.shape[0]))
        )

        max_heights.append(x[i].shape[0])

    # dtype=object due to images of different shapes
    return np.array(x_norm, dtype=object), np.array(y_norm), y, np.array(max_heights)

def reduce_hu_scale(img):
    """
    Thresholds pixel values to [100, 1500] interval (Hounsfield scale), then scales to [0, 1] * 255.
    """

    img = np.clip(img, config.HU_LOWER, config.HU_UPPER)
    img = (img-config.HU_LOWER) / (config.HU_UPPER-config.HU_LOWER)
    img *= 255

    return img.astype(np.float32) # TODO: czy na pewno float32?, tutaj chyba jest konwersja na ndarray

def pad_img(img, label, input_shape):
    # TODO: uwaga, wcześniej nie brałem pod uwagę
    # że padding wysokości ma wpływ na zmianę labela!
    # Wywoływałem też tę metodę za każdym razem w generatorze.
    """
    Pads image with 0 to match given input_shape.
    """

    h_diff = max(0, input_shape[0]-img.shape[0])
    w_diff = max(0, input_shape[1]-img.shape[1])

    h_pads = (h_diff//2, h_diff//2 + 1) if h_diff % 2 else (h_diff/2, h_diff/2)
    w_pads = (w_diff//2, w_diff//2 + 1) if w_diff % 2 else (w_diff/2, w_diff/2)

    img = cv2.copyMakeBorder(img, int(h_pads[0]), int(h_pads[1]), int(w_pads[0]), int(w_pads[1]),
                                borderType=cv2.BORDER_CONSTANT, value=0)

    # top padding influences label
    return img, label + int(h_pads[0])

def get_random_crops(img, label, input_shape, n_crops):
    # TODO: border_shift?
    """
    Returns `n_samples` image crops of `input_shape` shape and updated labels,
    with chance of not preserving label with `config.ANYWHERE_RATE` rate.
    """

    crops = []
    nolabel_indices = [] # for keeping track of crops with no vertrebrae
    labels = []
    for i in range(n_crops):

        # pick upper corner for crop
        if np.random.rand() < config.ANYWHERE_RATE:
            # crop anywhere
            upper_y = np.random.randint(0, img.shape[0]-input_shape[0])
            upper_x = np.random.randint(0, img.shape[1]-input_shape[1])
        else:
            # crop so that it contains vertebrae (with some distance to border)
            upper_y = np.random.randint(
                max(0, label-input_shape[0]+config.Y_DIST), 
                min(img.shape[0]-input_shape[0], label-config.Y_DIST))
            upper_x = np.random.randint(
                max(0, input_shape[1]//2-config.X_DIST),
                min(img.shape[1]-input_shape[1], input_shape[1]//2+config.X_DIST)
            )

        crops.append(img[upper_y : upper_y+input_shape[0],
                            upper_x : upper_x+input_shape[1]])
        
        # calculate new label after cropping
        if label - upper_y >= img.shape[0] or upper_y > label:
            nolabel_indices.append(i)
            labels.append(-1)
        else:
            labels.append(label-upper_y)

    return np.stack(crops), np.stack(labels), nolabel_indices
    
def get_random_crop(img, label, input_shape):
    """
    Returns image crop of `input_shape` shape and updated label,
    with chance of not preserving label with `config.ANYWHERE_RATE` rate.
    """

    # pick upper corner for crop
    if np.random.rand() < config.ANYWHERE_RATE:
        # crop anywhere
        upper_y = np.random.randint(0, img.shape[0]-input_shape[0]+1)
    else:
        # crop so that it contains vertebrae
        upper_y = np.random.randint(
            max(0, label-input_shape[0]+1),
            min(img.shape[0]-input_shape[0], label)+1)
    
    upper_x = max(0, img.shape[1]//2-input_shape[1]//2)

    crop_img = img[upper_y : upper_y+input_shape[0],
                        upper_x : upper_x+input_shape[1]]
    crop_label = label - upper_y

    return crop_img, crop_label

# TODO: ogarnąć
def augment_slice_thickness(img, max_r=5): 
    r = np.random.randint(1, max_r+1)
    return np.expand_dims(cv2.resize(img[::r], img.shape[:2][::-1]), 2)

def func_images(images, random_state, parents, hooks):
    result = []

    for image in images:
        image_aug = augment_slice_thickness(image, max_r=8)
        result.append(image_aug)

    return result

def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images

def get_augmentation_sequence():
    """
    Returns imgaug's Sequntial object.
    """

    slice_thickness_augmenter = iaa.Lambda(
    func_images=func_images, # funkcja wywoływana dla każdego batcha obrazów
    func_keypoints=func_keypoints) # funkcja wywoływana dla każdego batcha etykiet)

    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Fliplr(0.5)), # horizontal flip
        iaa.Sometimes(0.1, iaa.Add((-70, 70))), # adding to pixel values
        iaa.Sometimes(0.5, iaa.Affine(
            scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)} 
        )), # skaluje obrazy do 80-120% wymiaru, niezależnie dla wymiarów
        iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.01))), 
        # skaluje obrazy lokalnie o mały procent (zniekształcenia)
        # 2 poniższe są najbardziej agresywne i tworzą różne artefakty
        iaa.Sometimes(0.1,
                      iaa.BlendAlphaSimplexNoise(iaa.OneOf(
                          # nakłada przetworzony obraz na oryginalny obraz maskując przetworzony blobami
                          [iaa.Add((150, 255)), iaa.Add((-100, 100))]), sigmoid_thresh=5)),
        iaa.Sometimes(0.1, iaa.OneOf([iaa.CoarseDropout((0.01, 0.15), size_percent=(0.02, 0.08)),
                                      iaa.CoarseSaltAndPepper(p=0.2, size_percent=0.01),
                                      iaa.CoarseSalt(p=0.2, size_percent=0.02)
                                      ])),
        iaa.Sometimes(0.25, slice_thickness_augmenter) 
    ])

    return aug_seq

def get_heatmap(y_batch, sigma):
    """
    Returns heatmaps created by applying gaussian blur on one-hot vectors.
    """

    return gaussian_filter1d(y_batch, sigma)

def clip_imgs(imgs):
    """
    Used to clip augmented images pixels' values back to [0, 255]. 
    Values are clipped (not scaled) on account of preserving thresholding logic 
    when loading original MIP images - augmentations might simulate artefacts and implants, 
    but the real ones wouldn't make other values darker while loading and 
    thresholding using HU (that would happen when using scaling instead).
    """

    return np.clip(imgs, 0, 255)
