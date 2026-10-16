
import tensorflow as tf
import numpy as np

print("Project: TensorFlow Lite Converter")
print("Author: SELVASH")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="sgd", loss="mse")

x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

model.fit(x, y, epochs=100, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("linear_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted successfully!")
print("Saved as linear_model.tflite")
