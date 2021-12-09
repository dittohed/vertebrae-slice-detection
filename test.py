from src.generator_tf import DataGenerator

dataset = DataGenerator().get_tfdataset()
for x, y in dataset.as_numpy_iterator():
    print(x.shape)
    print(y.shape)

