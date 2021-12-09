from tensorflow.keras import callbacks
import src.config as config
import src.generator as generator
import src.utils as utils
import src.preprocessing as preprocessing
import src.models as models
import src.callbacks as callbacks

if __name__ == '__main__':

    print('Validating using crops.')

    x, y, spacings, names = utils.load_data(config.USE_FRONT)
    x, y, slices, heights = preprocessing.normalize_data(x, y, spacings)

    # optional
    # utils.save_imgs_dist(x, 'front' if experiment[0] else 'sagittal')

    x_train, y_train, \
        x_val, y_val, spacings_val, slices_val, heights_val = utils.split_data(x, y, spacings, slices, heights)

    gen_train = generator.DataGenerator(x_train, y_train, 
                            input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE, rgb=config.RGB)
    gen_val = generator.DataGenerator(x_val, y_val, validation=True,
                            input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE, rgb=config.RGB)

    model = models.get_model(config.MODEL_NAME)
    history = model.fit(gen_train, validation_data=gen_val,
                    epochs=config.NUM_EPOCHS, 
                    callbacks=callbacks.get_callbacks('crops_nosigma'))
    utils.plot_learning(history, 'crops_nosigma')