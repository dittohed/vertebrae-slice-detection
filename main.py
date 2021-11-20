# Resources:
# https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18
# https://dzlab.github.io/dltips/en/keras/data-generator/
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
# https://stackoverflow.com/questions/53620163/clarification-about-keras-utils-sequence

from tensorflow.keras import callbacks
import src.config as config
import src.generator as generator
import src.utils as utils
import src.preprocessing as preprocessing
import src.models as models
import src.callbacks as callbacks

if __name__ == '__main__':

    experiments = [
        (False, 'train0'),
        (True, 'train1')
    ]
    for experiment in experiments:
        print(f'Running experiment: {experiment[1]}')

        x, y, spacings, names = utils.load_data(experiment[0])
        x, y = preprocessing.normalize_data(x, y, spacings)
        x_train, y_train, names_train, x_val, y_val, names_val = utils.split_data(x, y, names)

        gen_train = generator.DataGenerator(x_train, y_train, 
                                input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE)
        gen_val = generator.DataGenerator(x_val, y_val, validation=True,
                                input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE)

        # for i, (x, y) in enumerate(gen_train):
        #     if i == len(gen_train):
        #         break

        model = models.get_model()
        history = model.fit(gen_train, validation_data=gen_val,
                        epochs=config.NUM_EPOCHS, 
                        callbacks=callbacks.get_callbacks(experiment[1]))
        utils.plot_learning(history, experiment[1])