from tensorflow import keras
from tensorflow.keras import layers
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

modelPath = os.path.abspath(os.path.dirname(__file__)) + "\\model\\demo"

model = keras.models.load_model(modelPath)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.RMSprop(),
              metrics=["accuracy"])

test_scores = model.evaluate(x_test, y_test)
print("Test Loss ", test_scores[0])
print("Test Accuracy ", test_scores[1])
