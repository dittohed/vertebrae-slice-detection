# TODO: gen summaries

import os

import src.config as config
import src.generators as generators
import src.utils as utils
import src.loaders as loaders
import src.models as models
import src.callbacks as callbacks

if __name__ == '__main__':
    model = models.get_model(config.MODEL_NAME)

    x_train, x_val, y_train, y_val = loaders.get_data_kanavati()

    gen_train = generators.DataGenerator(x_train, y_train, 
                            input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE, rgb=config.RGB)

    x_val, y_val = utils.prepare_for_inference(x_val, y_val, to_32=True)
    gen_val = generators.InferenceDataGenerator(x_val, y_val)
    
    history = model.fit(gen_train, validation_data=gen_val,
                    epochs=config.NUM_EPOCHS, 
                    callbacks=callbacks.get_callbacks(config.EXP_NAME))
    utils.plot_learning(history, config.EXP_NAME)