from tensorflow.keras import callbacks
import src.config as config
import src.generator as generator
import src.utils as utils
import src.preprocessing as preprocessing
import src.models as models
import src.callbacks as callbacks

if __name__ == '__main__':

    print('Validating using whole images.')

    x, y, spacings, names = utils.load_data(config.USE_FRONT)
    x, y = preprocessing.normalize_data(x, y, spacings)

    # optional
    # utils.save_imgs_dist(x, 'front' if experiment[0] else 'sagittal')

    x_train, y_train, names_train, x_val, y_val, names_val = utils.split_data(x, y, names)
    x_val, y_val = utils.prepare_for_inference(x_val, y_val) 

    gen_train = generator.DataGenerator(x_train, y_train, 
                            input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE)
    gen_val = generator.InferenceDataGenerator(x_val, y_val)

    model = models.get_model()
    history = model.fit(gen_train, validation_data=gen_val,
                    epochs=config.NUM_EPOCHS, 
                    callbacks=callbacks.get_callbacks('whole'))
    utils.plot_learning(history, 'whole')