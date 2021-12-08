import os

from tensorflow.keras.layers import Conv1D, SeparableConv2D, Conv2D, \
UpSampling1D, Conv1DTranspose, MaxPooling2D, Dropout, \
BatchNormalization, Activation, LeakyReLU, \
add, Layer, Input, InputSpec, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB7
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import numpy as np

from . import config
# import config

def down_block(input_tensor, n_filters, k_size=3, n_conv=2, 
              use_maxpool=True, pool_size=2, leaky_relu=False, bn_momentum=0.99):
    """
    Performs two 2D convolutions with additional operations 
    (in a similar manner to U-Net's contracting path blocks).
    """

    down = input_tensor
    for _ in range(n_conv):
        down = Conv2D(n_filters, k_size, padding='same')(down)
        down = BatchNormalization(momentum=bn_momentum)(down) # Kanavati et al. use non-standard momentum
        if leaky_relu:
            down = LeakyReLU(1e-2)(down)
        else:
            down = Activation('relu')(down)

    if use_maxpool:
        pool = MaxPooling2D(pool_size)(down)
    else:
        pool = Conv2D(n_filters, k_size, strides=pool_size, padding='same')(down)
        pool = BatchNormalization(momentum=bn_momentum)(pool)
        if leaky_relu:
            pool = LeakyReLU(1e-2)(pool)
        else:
            pool = Activation('relu')(pool)

    return down, pool

def up_block(input_tensor1, input_tensor2, n_filters, k_size=3, n_conv=2,
            use_transpose=True, upscale=2, for_eff=False, leaky_relu=False, bn_momentum=0.99):
    """
    Performs upscale'ing, concatenation and two 1D convolutions with additional operations 
    (in a similar manner to U-Net's expanding path blocks).
    """

    if use_transpose:
        up = Conv1DTranspose(n_filters, upscale, strides=upscale, padding='same')(input_tensor1)
    else:
        up = UpSampling1D(upscale)(input_tensor1) # Kanavati et al. don't use transpose convolutions

    if for_eff: 
        up = Dropout(0.25)(up)

    up = concatenate([up, GlobalMaxHorizontalPooling2D()(input_tensor2)])

    if not for_eff: # EfficientNet applies built-in dropout before skip connection
        up = Dropout(0.25)(up)
    
    for _ in range(n_conv):
        up = Conv1D(n_filters, k_size, padding='same')(up)
        up = BatchNormalization(momentum=bn_momentum)(up)
        if leaky_relu:
            up = LeakyReLU(1e-2)(up)
        else:
            up = Activation('relu')(up)

    return up

# as in sarcopenia-ai
class _GlobalHorizontalPooling2D(Layer):
    """
    Abstract class for different global pooling 2D layers.
    """

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

class GlobalMaxHorizontalPooling2D(_GlobalHorizontalPooling2D):
    """
    Calulates max value for each row to convert 2D to 1D.
    """

    def call(self, inputs):
        return K.max(inputs, axis=[2])

# https://keras.io/api/metrics/
def distance(y_true, y_pred):
    x_true = K.flatten(K.argmax(y_true, axis=1)) # dla każdego obrazu max indeks (batch_size,)
    valid = K.cast(K.sum(y_true, axis=(1, 2)) > config.THRESHOLD, 'float32') # jeżeli jest mocna predykcja

    x_pred = K.flatten(K.argmax(y_pred, axis=1)) # dla każdego obrazu max indeks (batch_size,)
    d = K.cast(x_true - x_pred, 'float32') # różnica w mm

    return K.abs(valid * d) # * d # kwadratowa róznica w mm  (batch_size,)

def get_model_kanavati():
    """
    Definiuje model odpowiadający CNNLine z repozytorium (apps/slice_detection/models.py).
    Są małe niezgodności z pracą (dodatkowa konwolucja x1 przy pod koniec bloku przy wchodzeniu do góry
    oraz liczba filtrów w przedostatniej warstwie modelu).
    Niemniej jednak, liczba parametrów zgadza się z tą podaną w  pracy!
    """

    input_shape = (None, None, 1)
    inputs = Input(input_shape)

    down1, pool1 = down_block(inputs, 32, bn_momentum=0.8)
    down2, pool2 = down_block(pool1, 64, bn_momentum=0.8)
    down3, pool3 = down_block(pool2, 128, bn_momentum=0.8)
    down4, pool4 = down_block(pool3, 256, k_size=5, n_conv=1, pool_size=4, bn_momentum=0.8)

    mid, _ = down_block(pool4, 512, 3)
    mid = GlobalMaxHorizontalPooling2D()(mid)

    # in expanding path, Kanavati et al. use additional convolution with k_size=1
    up1 = up_block(mid, down4, 256, k_size=5, n_conv=1, use_transpose=False, upscale=4, bn_momentum=0.8)
    up1 = Conv1D(256, 1, padding='same')(up1)
    up1 = BatchNormalization(momentum=0.8)(up1)
    up1 = Activation('relu')(up1)

    up2 = up_block(up1, down3, 128, use_transpose=False, bn_momentum=0.8)
    up2 = Conv1D(128, 1, padding='same')(up2)
    up2 = BatchNormalization(momentum=0.8)(up2)
    up2 = Activation('relu')(up2)

    up3 = up_block(up2, down2, 128, use_transpose=False, bn_momentum=0.8)
    up3 = Conv1D(128, 1, padding='same')(up3)
    up3 = BatchNormalization(momentum=0.8)(up3)
    up3 = Activation('relu')(up3)

    up4 = up_block(up3, down1, 128, use_transpose=False, bn_momentum=0.8)
    up4 = Conv1D(128, 1, padding='same')(up4)
    up4 = BatchNormalization(momentum=0.8)(up4)
    up4 = Activation('relu')(up4)

    outputs = Conv1D(1, 1, activation='sigmoid', padding='same')(up4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model_eff():
    """
    Coś tam mam, ale najlepiej byłoby dobrać wersję, która będzie mieć najbardziej zbliżony shape.
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    """
    inputs = Input(shape=(None, None, 3))

    # TODO: be careful with scaling, normalization and 3 channels
    backbone = EfficientNetB7(weights='imagenet', include_top=False, input_tensor=inputs)
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    block0 = backbone.get_layer('normalization').output 
    block1 = backbone.get_layer('block1d_add').output
    block2 = backbone.get_layer('block2g_add').output
    block3 = backbone.get_layer('block3g_add').output
    block4 = backbone.get_layer('block4j_add').output

    conv_mid = GlobalMaxHorizontalPooling2D()(block4)

    up1 = up_block(conv_mid, block3, 256, for_eff=True)
    up2 = up_block(up1, block2, 128, for_eff=True)
    up3 = up_block(up2, block1, 128, for_eff=True)
    up4 = up_block(up3, block0, 64, for_eff=True)

    outputs = Conv1D(1, 1, activation='sigmoid', padding='same')(up4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model_own():
    """
    Model a'la nnU-Net (bez maxpoolingu, instance normalization? i leaky ReLU 1e-2)
    """
    # TODO: instance normalization?
    input_shape = (None, None, 1)
    inputs = Input(input_shape)

    down1, pool1 = down_block(inputs, 32, use_maxpool=False, leaky_relu=True)
    down2, pool2 = down_block(pool1, 64, use_maxpool=False, leaky_relu=True)
    down3, pool3 = down_block(pool2, 128, use_maxpool=False, leaky_relu=True)
    down4, pool4 = down_block(pool3, 256, use_maxpool=False, leaky_relu=True)

    mid, _ = down_block(pool4, 512, use_maxpool=False, leaky_relu=True)
    mid = GlobalMaxHorizontalPooling2D()(mid)

    up1 = up_block(mid, down4, 256, leaky_relu=True)
    up2 = up_block(up1, down3, 128, leaky_relu=True)
    up3 = up_block(up2, down2, 64, leaky_relu=True)
    up4 = up_block(up3, down1, 32, leaky_relu=True)

    outputs = Conv1D(1, 1, activation='sigmoid', padding='same')(up4)

    model = Model(inputs=[inputs], outputs=[outputs])
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

def get_model(model_name):
    """
    Returns a compiled model, according to given model name.
    Common conventions used in models' architectures (inspired by sarcopenia-ai):
    - 'same' padding
    - conv + BN + activation scheme
    - using dropout with skip-connections only (expanding path)
    """

    if model_name == 'Kanavati':
        model = get_model_kanavati()
    elif model_name == 'Efficient':
        model = get_model_eff()
    elif model_name == 'Own':
        model = get_model_own()

    model.compile(optimizer=config.OPTIMIZER, loss='binary_crossentropy', # loss_weights=[1000],  
                metrics=[distance])
    return model

if __name__ == '__main__':
    get_model('Efficient').summary()