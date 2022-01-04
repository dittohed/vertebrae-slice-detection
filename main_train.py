"""
Script for running training experiments.
"""

import src.config as config
import src.generators as generators
import src.utils as utils
import src.loaders as loaders
import src.models as models
import src.callbacks as callbacks

if __name__ == '__main__':
    model = models.get_model(config.MODEL_NAME)

    if config.V_LEVEL == 'L3':
        x_train, x_val, y_train, y_val = loaders.get_data_l3()
    elif config.V_LEVEL == 'T12':
        x_train, x_val, y_train, y_val = loaders.get_data_t12()
        # use line below if using pretrained l3 model
        # model.load_weights(os.path.join(config.CHECKPOINT_PATH, 
        #                'kanavati_l3_frontal_bce_2', 'model'))

    gen_train = generators.DataGenerator(x_train, y_train, 
                           input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE, rgb=config.RGB)

    # lines below for fine-tuning on whole size images
    # model.load_weights(os.path.join(config.CHECKPOINT_PATH, '<checkpoint_subdir>', 'model'))
    # x_train, y_train = utils.prepare_for_inference(x_train, y_train, to_32=True)
    # gen_train = generators.InferenceDataGenerator(x_train, y_train, shuffle=True)

    # lines below for using whole size images for validation
    x_val, y_val = utils.prepare_for_inference(x_val, y_val, to_32=True)
    gen_val = generators.InferenceDataGenerator(x_val, y_val)

    # lines below for using image crops for validation (not recommended)
    # gen_val = generators.DataGenerator(x_val, y_val, validation=True, 
    #                         input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE, rgb=config.RGB) 
    
    history = model.fit(gen_train, validation_data=gen_val,
                    epochs=config.NUM_EPOCHS, 
                    callbacks=callbacks.get_callbacks(config.EXP_NAME))
    utils.plot_learning(history, config.EXP_NAME)
