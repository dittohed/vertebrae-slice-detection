import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

from . import config
from . import preprocessing

def get_ds_dict(ds_names, v_level):
    """
    For joining datasets used in thesis.
    """

    ds_dict = {}

    for ds_name in ds_names:
        data = np.load(os.path.join(config.DATA_PATH, f'{ds_name}.npz'), allow_pickle=True)
        y = data['v_levels']

        if v_level == 'L3':
            y = np.asarray(
                [l3_lvl for [t12_lvl, l3_lvl] in y])
        elif v_level == 'T12':
            y = np.asarray(
                [t12_lvl for [t12_lvl, l3_lvl] in y])
        
        if config.USE_FRONT:
            x = data['imgs_f']
        else:
            x = data['imgs_s']
        x = x[y != -1]

        # normalization (100-1500 HU + [0, 255] float values range)
        x_norm = []
        for img in x:
            img = preprocessing.reduce_hu_scale(img)
            x_norm.append(img)
        x_norm = np.array(x_norm, dtype=object) # dtype=object due to images of different shapes
        
        ids = data['ids']
        ids = ids[y != -1]
        ids_new = []
        for exam_id in ids:
            ids_new.append(f'{ds_name}_{exam_id}') # to not mix ids between datasets
        ids = np.asarray(ids_new)

        thicks = data['thicks']
        thicks = thicks[y != -1]

        y = y[y != -1]

        ds_dict[ds_name] = {
            'x': x_norm,
            'y': y,
            'ids': ids,
            'thicks': thicks
        } 

    return ds_dict

def get_data_l3():
    """
    Kanavati: 1006 (100%-0-0)
    RAW: 140 (0-50%-50%)
    VerSe2019: 123 (0-50%-50%)
    """
    
    # loading
    ds_names = ['Kanavati', 'RAW', 'VerSe2019'] # datasets
    ds_dict = get_ds_dict(ds_names, 'L3') 

    kanavati_x_train = ds_dict['Kanavati']['x']
    kanavati_y_train = ds_dict['Kanavati']['y']

    raw_x_val, raw_x_test, \
    raw_y_val, raw_y_test, \
    raw_ids_val, raw_ids_test, \
    raw_thicks_val, raw_thicks_test = train_test_split(
       ds_dict['RAW']['x'], ds_dict['RAW']['y'], 
       ds_dict['RAW']['ids'], ds_dict['RAW']['thicks'],
       test_size=0.5, shuffle=True, 
       random_state=7)

    verse_x_val, verse_x_test, \
    verse_y_val, verse_y_test, \
    verse_ids_val, verse_ids_test, \
    verse_thicks_val, verse_thicks_test = train_test_split(
       ds_dict['VerSe2019']['x'], ds_dict['VerSe2019']['y'], 
       ds_dict['VerSe2019']['ids'], ds_dict['VerSe2019']['thicks'],
       test_size=0.5, shuffle=True, 
       random_state=7)

    # merging val & test datasets   
    x_val = np.concatenate(
        [raw_x_val, verse_x_val])
    y_val = np.concatenate(
        [raw_y_val, verse_y_val])

    # preserving more information for test set 
    # (ultimate goal is to return a slice number for given examination)
    x_test = np.concatenate(
        [raw_x_test, verse_x_test])
    y_test = np.concatenate(
        [raw_y_test, verse_y_test])
    ids_test = np.concatenate(
        [raw_ids_test, verse_ids_test])
    thicks_test = np.concatenate(
        [raw_thicks_test, verse_thicks_test]) 

    # saving test set for final test
    suffix = 'frontal' if config.USE_FRONT else 'sagittal'
    np.savez_compressed(os.path.join(config.DATA_PATH, f'test_l3_{suffix}.npz'), x=x_test, y=y_test,
                    ids=ids_test, thicks=thicks_test)

    return kanavati_x_train, x_val, kanavati_y_train, y_val

def get_data_t12():
    """
    TH12: 107 (60%-20%-20%)
    RAW: 138 (60%-20%-20%)
    VerSe2019: 105 (60%-20%-20%)
    """

    # loading
    ds_names = ['TH12', 'RAW', 'VerSe2019'] # datasets
    ds_dict = get_ds_dict(ds_names, 'T12')

    # train-val-test splits
    th12_x_train, th12_x_valtest, \
    th12_y_train, th12_y_valtest, \
    th12_ids_train, th12_ids_valtest, \
    th12_thicks_train, th12_thicks_valtest = train_test_split(
       ds_dict['TH12']['x'], ds_dict['TH12']['y'], 
       ds_dict['TH12']['ids'], ds_dict['TH12']['thicks'],
       test_size=0.4, shuffle=True, 
       random_state=7)

    th12_x_val, th12_x_test, \
    th12_y_val, th12_y_test, \
    th12_ids_val, th12_ids_test, \
    th12_thicks_val, th12_thicks_test = train_test_split(
       th12_x_valtest, th12_y_valtest, th12_ids_valtest, th12_thicks_valtest,
       test_size=0.5, shuffle=True, 
       random_state=7)

    raw_x_train, raw_x_valtest, \
    raw_y_train, raw_y_valtest, \
    raw_ids_train, raw_ids_valtest, \
    raw_thicks_train, raw_thicks_valtest = train_test_split(
       ds_dict['RAW']['x'], ds_dict['RAW']['y'], 
       ds_dict['RAW']['ids'], ds_dict['RAW']['thicks'],
       test_size=0.4, shuffle=True, 
       random_state=7)

    raw_x_val, raw_x_test, \
    raw_y_val, raw_y_test, \
    raw_ids_val, raw_ids_test, \
    raw_thicks_val, raw_thicks_test = train_test_split(
       raw_x_valtest, raw_y_valtest, raw_ids_valtest, raw_thicks_valtest,
       test_size=0.5, shuffle=True, 
       random_state=7)

    verse_x_train, verse_x_valtest, \
    verse_y_train, verse_y_valtest, \
    verse_ids_train, verse_ids_valtest, \
    verse_thicks_train, verse_thicks_valtest = train_test_split(
       ds_dict['VerSe2019']['x'], ds_dict['VerSe2019']['y'], 
       ds_dict['VerSe2019']['ids'], ds_dict['VerSe2019']['thicks'],
       test_size=0.4, shuffle=True, 
       random_state=7)

    verse_x_val, verse_x_test, \
    verse_y_val, verse_y_test, \
    verse_ids_val, verse_ids_test, \
    verse_thicks_val, verse_thicks_test = train_test_split(
       verse_x_valtest, verse_y_valtest, verse_ids_valtest, verse_thicks_valtest,
       test_size=0.5, shuffle=True, 
       random_state=7)

    # merging train, val & test datasets
    x_train = np.concatenate(
            [th12_x_train, raw_x_train, verse_x_train])
    y_train = np.concatenate(
        [th12_y_train, raw_y_train, verse_y_train])

    x_val = np.concatenate(
        [th12_x_val, raw_x_val, verse_x_val])
    y_val = np.concatenate(
        [th12_y_val, raw_y_val, verse_y_val])

    # preserving more information for test set 
    # (ultimate goal is to return a slice number for given examination)
    x_test = np.concatenate(
        [th12_x_test, raw_x_test, verse_x_test])
    y_test = np.concatenate(
        [th12_y_test, raw_y_test, verse_y_test])
    ids_test = np.concatenate(
        [th12_ids_test, raw_ids_test, verse_ids_test])
    thicks_test = np.concatenate(
        [th12_thicks_test, raw_thicks_test, verse_thicks_test]) 

    # saving test set for final test
    suffix = 'frontal' if config.USE_FRONT else 'sagittal'
    np.savez_compressed(os.path.join(config.DATA_PATH, f'test_t12_{suffix}.npz'), x=x_test, y=y_test,
                    ids=ids_test, thicks=thicks_test)

    return x_train, x_val, y_train, y_val

def get_test_data(ds_name):
    """
    Reads .npz archive with test data.
    """

    data = np.load(os.path.join(config.DATA_PATH, ds_name), allow_pickle=True)

    return data['x'], data['y'], data['ids'], data['thicks']

def get_inference_data():
    """
    Reads .npz archive and preprocesses arrays for inference.
    """

    data = np.load(os.path.join(config.DATA_PATH, config.INF_DS_NAME), allow_pickle=True)
    x, ids, thicks = data['x'], data['ids'], data['thicks']

    # padding (same as utils.prepare_for_inference(), but with no y)
    for i in range(x.shape[0]):

        # comment line below for testing inference on test_*_frontal.npz files
        x[i] = preprocessing.reduce_hu_scale(x[i]) # HU to [0, 255] with thresholding

        # both width and height should be divisible by 32 (maxpooling and concats)
        # and not less than training crops size

        new_height = int(
            max(np.ceil((x[i].shape[0] / 32)) * 32, config.INPUT_SHAPE[0]))
        new_width = int(
            max(np.ceil((x[i].shape[1] / 32)) * 32, config.INPUT_SHAPE[1]))

        h_diff = max(0, new_height-x[i].shape[0])
        w_diff = max(0, new_width-x[i].shape[1])

        h_pads = (h_diff//2, h_diff//2 + 1) if h_diff % 2 else (h_diff/2, h_diff/2)
        w_pads = (w_diff//2, w_diff//2 + 1) if w_diff % 2 else (w_diff/2, w_diff/2)

        x[i] = cv2.copyMakeBorder(x[i], int(h_pads[0]), int(h_pads[1]), int(w_pads[0]), int(w_pads[1]),
                                    borderType=cv2.BORDER_CONSTANT, value=0)

        # add dummy color channel
        x[i] = np.expand_dims(x[i], 2).astype(np.float32)

    # pixel values to [-1, 1]
    x = (x / 255) * 2 - 1

    return x, ids, thicks