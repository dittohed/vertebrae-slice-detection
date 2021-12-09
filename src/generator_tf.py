"""
Reference:
https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18

1) Tworzę generator zwracający indeksy
2) Tworzę Dataset na podstawie generatora
3) W map() daję ciężką funkcję
4) Używam num_parallel_calls, żeby zrównoleglić ciężkie funkcje
*) Chyba wywołuję funkcję dla pojedynczej próbki i używam batch_size - popatrzeć na inne przykłady
"""

import tensorflow as tf
import numpy as np

from . import preprocessing
from . import utils

class DataGenerator():
    def __init__(self, x, y):
        self.x = x
        self.y = y       

    def get_tfdataset(self):
        def map_func(i):
            i = i.numpy()
            
            img = self.x[i]
            label = self.y[i]

            x, y = preprocessing.get_random_crop(
                                img, label, self.input_shape)

            # TODO: augmentations
            # y = utils.y_to_keypoint([x], [y])
            # ...

            y = utils.y_to_onehot(y, self.input_shape)

            # TODO: other

            # add dummy color channel
            x = np.expand_dims(x, 3)
            y = np.expand_dims(y, 2)

            return x, y

        generator = list(range(self.x.shape[0]))

        dataset = tf.data.Dataset.from_generator(lambda: generator, tf.uint8) # TODO: potem trzeba na 16
        dataset = dataset.shuffle(buffer_size=self.x.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.map(lambda i: tf.py_function(func=map_func, 
                                                           inp=[i], 
                                                           Tout=[tf.uint8, tf.uint8]),
                                                           num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)

        return dataset
        
        
