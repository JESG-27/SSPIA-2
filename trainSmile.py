import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator
import pickle

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

training_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_ds = training_datagen.flow_from_directory(
    "datasets/smilingornot",
    classes=["smile", "non_smile"],
    color_mode="rgb",
    batch_size=16,
    target_size=(64, 64),
    class_mode="categorical",
    subset="training"
)

model = keras.models.Sequential([
    layers.Conv2D(64, 2, activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, 2, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', 'binary_accuracy'])
model.summary()
history = model.fit(train_ds, epochs=20, verbose=1)

model.save("smileornot.h5")