# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import tensorflow as tf
# import datasets
import tensorflow_dataset as tfds

tfds.disable_progress_bar()

# helper libs

import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = metadata.features['label'].names
print("class names: {}".format(class_names))

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))


# preprocess the data
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

    # The map function applies the normalize function
    # to each element in the train and test datasets

    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()

    for image, label in test_dataset.take(1):
        break
        image = image.numpy().reshape((28, 28))

    # plot image
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Display first 25 images and their class name
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(train_dataset.take(25)):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    plt.show()

    # model and layers
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Train
    BATCH_SIZE = 32
    train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    train_dataset = test_dataset.cache().batch(BATCH_SIZE)

    # Fitting
    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
    print('Accuracy on test dataset:', test_accuracy)

    # make predictions
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)
        predictions.shape
        predictions[0]
        np.argmax(predictions[0])
        test_labels[0]

        # Graph to look at the full set of 10 class predictions
        def plot_image(i, predictions_array, true_labels, images):
            predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img[..., 0], cmap=plt.cm.binary)
            predicted_label = np.argmax(predictions_array)
            if predicted_label == true_label:
                color = 'blue'
            else:
                color = 'red'

            plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                                 100*np.max(predictions_array),
                                                 class_names[true_label]),
                       color=color)

            def plot_value_array(i, predictions_array, true_label):
                predictions_array, true_label = predictions_array[i], true_label[i]
                plt.grid(False)
                plt.xticks([])
                plt.yticks([])
                thisplot = plt.bar(range(10), predictions_array, color="#777777")
                plt.ylim([0, 1])
                predicted_label = np.argmax(predictions_array)

                thisplot[predicted_label].set_color('red')
                thisplot[true_label].set_color('blue')

                # image 0
                i = 0
                plt.figure(figsize=(6, 3))
                plt.subplot(1, 2, 1)
                plot_image(i, predictions, test_labels, test_images)
                plt.subplot(1, 2, 2)
                plot_value_array(i, predictions, test_labels)

                # 12th image
                i = 0
                plt.figure(figsize=(6, 3))
                plt.subplot(1, 2, 1)
                plot_image(i, predictions, test_labels, test_images)
                plt.subplot(1, 2, 2)
                plot_value_array(i, predictions, test_labels)

                # image from test
                img = test_images[0]
                print(img.shape)
                img = np.array([img])
                print(img.shape)

                predictions_single = model.predict(img)
                print(predictions_single)

                plot_value_array(0, predictions_single, test_labels)
                _ = plt.xticks(range(10), class_names, rotation=45)

                np.argmax(predictions_single[0])
