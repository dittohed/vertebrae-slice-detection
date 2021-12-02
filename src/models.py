import os

from tensorflow.keras.layers import Conv1D, SeparableConv2D, Conv2D, \
UpSampling1D, MaxPooling2D, Dropout, \
BatchNormalization, Activation, \
add, Layer, Input, InputSpec, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

from . import config

# wykonuje num_blocks operacji określonych przez conv_unit, a następnie max pooling 2x2
def conv_block(inp, num_filters=64, kernel_size=3, momentum=0.8, padding='same', pool_size=2, num_blocks=1,
               dilation_rate=(1, 1), separable=False):

    # określa 'jednostkę' (konwolucja + BatchNormalization + ReLU)
    def conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding='same', do_act=True, dilation_rate=(1, 1),
                  separable=False):

        if separable: # specyfikacja rodzaju konwolucji (SeparableConv2D - szybsza i lżejsza)
            ConvFn = SeparableConv2D
        else:
            ConvFn = Conv2D

        conv = ConvFn(num_filters, kernel_size, padding=padding, dilation_rate=dilation_rate)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = Activation('relu')(conv)

        return conv

    conv = inp
    for i in range(num_blocks):
        conv = conv_unit(conv, num_filters, kernel_size, momentum, padding, True, dilation_rate=dilation_rate,
                         separable=separable)

    if pool_size is not None:
        pool = MaxPooling2D(pool_size=pool_size)(conv)

        return conv, pool
    else:
        return conv

def up_conv_block_add_1D(inp, inp2, num_filters=64, kernel_size=3, momentum=0.8, padding='same', up_size=2,
                         num_blocks=3, is_residual=False):

    # określa 'jednostkę' (konwolucja + BatchNormalization + ReLU)
    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding='same', do_act=True):
        conv = Conv1D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = Activation('relu')(conv)
        return conv

    if is_residual:
        inp = Conv1D(num_filters, 1, padding=padding)(inp)

    # upsampling na aktualnym tensorze
    inp = UpSampling1D(size=up_size)(inp)
    upcov = inp

    # konwersja 2D -> 1D na połączonym tensorze
    inp2 = GlobalMaxHorizontalPooling2D()(inp2)

    # złączenie
    upcov = concatenate([upcov, inp2], axis=2)

    upcov = Dropout(0.25)(upcov)
    for i in range(num_blocks):
        upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding, True)

    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding, False)

    if is_residual:
        upcov = add([upcov, inp])
    upcov = Activation('relu')(upcov)

    return upcov

class _GlobalHorizontalPooling2D(Layer):
    '''
    Abstract class for different global pooling 2D layers.
    '''

    def __init__(self, data_format=None, **kwargs):
        super(_GlobalHorizontalPooling2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        #     return (input_shape[0], input_shape[1], input_shape[2])
        # else:
        return (input_shape[0], input_shape[1], input_shape[3])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalHorizontalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# oblicza max z wierszy w celu konwersji 2D -> 1D
class GlobalMaxHorizontalPooling2D(_GlobalHorizontalPooling2D):
    '''
    Global max pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    '''

    def call(self, inputs):
        # if self.data_format == 'channels_last':
        return K.max(inputs, axis=[2])
        # else:
        #     return K.max(inputs, axis=[3])

# https://keras.io/api/metrics/
def distance(y_true, y_pred):
    x_true = K.flatten(K.argmax(y_true, axis=1)) # dla każdego obrazu max indeks (batch_size,)
    valid = K.cast(K.sum(y_true, axis=(1, 2)) > config.THRESHOLD, 'float32') # jeżeli jest mocna predykcja

    x_pred = K.flatten(K.argmax(y_pred, axis=1)) # dla każdego obrazu max indeks (batch_size,)
    d = K.cast(x_true - x_pred, 'float32') # różnica w mm

    return K.abs(valid * d) # * d # kwadratowa róznica w mm  (batch_size,)

def get_model():
    '''
    Definiuje model odpowiadający CNNLine z repozytorium (apps/slice_detection/models.py).
    Są małe niezgodności z pracą (dodatkowa konwolucja x1 przy pod koniec bloku przy wchodzeniu do góry
    oraz liczba filtrów w przedostatniej warstwie modelu).
    Niemniej jednak, liczba parametrów zgadza się z tą podaną w  pracy!
    '''

    input_shape = (None, None, 1)
    inputs = Input(input_shape)

    # w dół
    conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
    # dwie 32-filtrowe konwolucje 3x3, a następnie max pooling 2x2

    conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
    # dwie 64-filtrowe konwolucje 3x3, a następnie max pooling 2x2

    conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
    # dwie 128-filtrowe konwolucje 3x3, a następnie max pooling 2x2

    conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=5, num_blocks=1, pool_size=4)
    # jedna 256-filtrowa konwolucja 5x5, a następnie max pooling 4x4

    conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)
    # dwie 512-filtrowe konwolucje 3x3 bez max poolingu

    conv_mid = GlobalMaxHorizontalPooling2D()(conv_mid)
    # max z wierszy

    # do góry
    conv6 = up_conv_block_add_1D(conv_mid, conv5, num_filters=256, kernel_size=5, num_blocks=1, up_size=4)
    # łączy poprzedni wynik 2D z upscale'owanym obecnym wynikiem 1D
    # wykonuje jedną 256-filtrową konwolucję x5 + dodatkowo konwolucję x1

    conv7 = up_conv_block_add_1D(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
    # łączy poprzedni wynik 2D z upscale'owanym obecnym wynikiem 1D
    # wykonuje dwie 128-filtrowe konwolucje x3 + dodatkowo konwolucję x1

    conv8 = up_conv_block_add_1D(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
    # łączy poprzedni wynik 2D z upscale'owanym obecnym wynikiem 1D
    # wykonuje dwie 128-filtrowe konwolucje x3 + dodatkowo konwolucję x1

    conv9 = up_conv_block_add_1D(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)
    # łączy poprzedni wynik 2D z upscale'owanym obecnym wynikiem 1D
    # wykonuje dwie 128-filtrowe konwolucje x3 + dodatkowo konwolucję x1

    conv10 = Conv1D(1, 1, activation='sigmoid', name='last_conv', padding='same')(conv9)
    # przekształca wielowarstwowy pasek 1D w jednowarstwowy pasek 1D z prawdopodobieństwami

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=config.OPTIMIZER, loss='binary_crossentropy', loss_weights=[1000],  
                 metrics=[distance])
    # using loss_weights for preventing extremely low values

    return model

def predict_whole(model, x):
    # TODO: Czy zachowanie offsetu pomoże? To jest ok eksperyment!
    """
    Predicts vertebrae level for whole-size images by 
    using non-overlapping, centered crops (windows) of training crops sizes.
    Prediction is done for all crops, then predicted vertebrae levels 
    are calculated.
    """

    # prepare crops 
    crops = []
    num_crops = {}
    for i, img in enumerate(x):
        assert img.shape[0] >= config.INPUT_SHAPE[0]
        assert img.shape[1] >= config.INPUT_SHAPE[1]

        num_crops[i] = int(np.ceil(img.shape[0] / config.INPUT_SHAPE[0]))

        # so that crop are centered     
        x_center = img.shape[1] // 2 # TODO: środkować według krzywizny kręgosłupa
        x_left = x_center - config.INPUT_SHAPE[1] // 2
        x_right = x_center + config.INPUT_SHAPE[1] // 2

        for j in range(num_crops[i]):
            y_upper = min(j * config.INPUT_SHAPE[0], img.shape[0] - config.INPUT_SHAPE[0])
            crops.append(img[y_upper : y_upper + config.INPUT_SHAPE[0], x_left : x_right])

    crops = np.asarray(crops) # TODO: sprawdzić typ
    y_crops = model.predict(crops) # batch_size = 32 by default
    # TODO: sprawdzić shape, zakładam, że (n_crops, 256)

    # determine predicted vertebrae location using max probability
    y = []
    l_curr = 0
    for key in num_crops:
        r_curr = l_curr + num_crops[key]
        y_group = np.stack(y_crops[l_curr : r_curr]) # predictions associated with single image

        max_prob = np.max(y_group)
        # if max_prob > config.THRESHOLD:
        #     # crop with maximum probability
        #     max_prob_crop = np.argmax(np.max(y_group, axis=-1)) 
        #     # vertebrae level within crop with maximum probability
        #     max_prob_crop_idx = np.argmax(y_group, axis=-1)[max_prob_crop]
        #     # verebrae level within whole image
        #     y_pred = min(max_prob_crop * config.INPUT_SHAPE[0], 
        #                 x[key].shape[0] - config.INPUT_SHAPE[0]) + max_prob_crop_idx
        # else:
        #     y_pred = -1 # -1 if no vertebrae predicted
        
        # crop with maximum probability
        max_prob_crop = np.argmax(np.max(y_group, axis=-1)) 
        # vertebrae level within crop with maximum probability
        max_prob_crop_idx = np.argmax(y_group, axis=-1)[max_prob_crop]
        # verebrae level within whole image
        y_pred = min(max_prob_crop * config.INPUT_SHAPE[0], 
                    x[key].shape[0] - config.INPUT_SHAPE[0]) + max_prob_crop_idx

        y.append(y_pred) 
        l_curr = r_curr

    y = np.asarray(y)
    return y

if __name__ == '__main__':
    print(get_model().summary())