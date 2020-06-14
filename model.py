from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, optimizers
import numpy as np
import pickle


def dense(input_shape=(28, 28, 1), num_classes=10):
    keras_model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])

    keras_model.add(Activation('softmax'))

    keras_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    return keras_model


def cnn(input_shape=(32, 32, 3), num_classes=10):
    keras_model = Sequential()
    keras_model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    keras_model.add(Activation('relu'))
    keras_model.add(Conv2D(32, (3, 3)))
    keras_model.add(Activation('relu'))
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    keras_model.add(Dropout(0.25))

    keras_model.add(Conv2D(64, (3, 3), padding='same'))
    keras_model.add(Activation('relu'))
    keras_model.add(Conv2D(64, (3, 3)))
    keras_model.add(Activation('relu'))
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    keras_model.add(Dropout(0.25))

    keras_model.add(Flatten())
    keras_model.add(Dense(512))
    keras_model.add(Activation('relu'))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(num_classes))
    keras_model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    keras_model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    return keras_model


if __name__ == '__main__':
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = False
    num_predictions = 20

    train_data_path = '/Users/ayushtiwari/Desktop/federated-learning/data/fashion_mnist/train/client_0.pickle'
    test_data_path = '/Users/ayushtiwari/Desktop/federated-learning/data/fashion_mnist/test/server.pickle'

    train_dataset = pickle.load(open(train_data_path, 'rb'))
    x_train = train_dataset["x_train"]
    y_train = train_dataset["y_train"]

    test_dataset = pickle.load(open(test_data_path, 'rb'))
    x_test = test_dataset["x_test"]
    y_test = test_dataset["y_test"]

    model = dense(input_shape=x_train.shape[1:], num_classes=10)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit(datagen.flow(x_train, y_train,
                               batch_size=batch_size),
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  workers=4)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
