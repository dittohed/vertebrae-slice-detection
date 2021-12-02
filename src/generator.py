import tensorflow as tf
import numpy as np

from . import preprocessing
from . import config
from . import utils

class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of images and labels. 
    Uses different augmentation techniques.
    """

    def __init__(self, x, y,  validation=False, input_shape=[256, 256, 1],
                batch_size=8, shuffle=True):
        """
        * imgs_per_batch - number of original images to use when generating batch with augmentation;
        * shuffle - whether to reshuffle data on epoch start.
        """

        self.x = x
        self.y = y
        self.validation = validation
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.max_sigma = config.MAX_SIGMA # for blurring vertebrae level
        self.min_sigma = config.MIN_SIGMA
        self.epoch = 0 # for controlling sigma value
        self.indices = np.arange(x.shape[0])

        self.aug_seq = preprocessing.get_augmentation_sequence().to_deterministic()
        self.curr_idx = 0 # TODO: for tests only, to be deleted
        self.used_indices = [] # TODO: for test only, to be deleted

        if self.shuffle and not self.validation:
            np.random.shuffle(self.indices)

        # pad imgs smaller than given input_shape
        for i in range(x.shape[0]):
            img = self.x[i]
            label = self.y[i]

            if img.shape[0] < self.input_shape[0] or img.shape[1] < self.input_shape[1]:
                print(f'Doing padding: {img.shape} -> {self.input_shape}')
                self.x[i], self.y[i] = preprocessing.pad_img(img, label, self.input_shape)

    def __len__(self):
        """
        Returns number of batches per epoch.
        """

        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """
        Returns a single batch given index.
        """

        batch_indices = self.indices[index * self.batch_size : 
                                    min(len(self.indices), (index+1) * self.batch_size)]
        self.used_indices.extend(batch_indices.tolist())

        x, y = self.__gen_batch(batch_indices)
        return x, y 

    def __next__(self):
        # TODO: for tests only, to be deleted
        x, y = self.__getitem__(self.curr_idx)
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        
        return x. y

    def __gen_batch(self, batch_indices):
        """
        Returns a generated batch given images indices.
        Originally it used n images to generate m*n images via augmentation.
        Now it uses n images to generate n images via augmentation.
        """       

        x_batch = np.zeros((len(batch_indices), self.input_shape[0], self.input_shape[1]))
        y_batch = np.zeros((len(batch_indices)))

        # gradually decreasing for better learning
        sigma = max(self.max_sigma - self.epoch, self.min_sigma)

        for i, img_idx in enumerate(batch_indices):
            img = self.x[img_idx]
            label = self.y[img_idx]

            crop_img, crop_label = preprocessing.get_random_crop(
                                img, label, self.input_shape)

            # (optional) saves a visual comparison for debugging
            # utils.save_orig_crop_comparison(img, label, crop_img, crop_label, img_idx)
            
            x_batch[i] = crop_img
            y_batch[i] = crop_label

        y_batch = utils.y_to_keypoint(x_batch, y_batch)

        # augmentations
        if not self.validation:
            x_batch = self.aug_seq.augment_images(x_batch.astype(np.float32))
            y_batch = self.aug_seq.augment_keypoints(y_batch)
            x_batch = preprocessing.clip_imgs(x_batch) 

        y_batch = np.array([utils.y_to_onehot(y.keypoints[0].y, self.input_shape) for y in y_batch])

        # (optional) saves a visual comparison for debugging
        # for i, img_idx in enumerate(batch_indices):
        #     img = self.x[img_idx]
        #     label = self.y[img_idx]
        #     aug_img = x_batch[i]
        #     aug_label = y_batch[i]

        #     utils.save_orig_aug_comparison(img, label, aug_img, aug_label, img_idx)

        # label blur
        if not self.validation:
            y_batch = preprocessing.get_heatmap(y_batch, sigma)

        # add dummy color channel
        x_batch = np.expand_dims(x_batch, 3)
        y_batch = np.expand_dims(y_batch, 2)

        # pixel values to [-1, 1]
        x_batch = (x_batch / 255) * 2 - 1

        return x_batch, y_batch

    def on_epoch_end(self):
        self.epoch += 1
        if self.shuffle and not self.validation:
            np.random.shuffle(self.indices)

# TODO: remove
class InferenceDataGenerator(tf.keras.utils.Sequence):
    """
    Returns batches of single images of varying size. 
    Used to being able to predict for images of any size.
    """

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        """
        Returns number of batches per epoch.
        """

        return self.x.shape[0]

    def __getitem__(self, index):
        """
        Returns a single batch given index.
        """

        return np.expand_dims(self.x[index], axis=0), np.expand_dims(self.y[index], axis=0)
