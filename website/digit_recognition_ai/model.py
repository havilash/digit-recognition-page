import tensorflow as tf

# import numpy as np
# import matplotlib.pyplot as plt
# import random

"""data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = x_train / 255
x_test = x_test / 255


def train_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=[28, 28]),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=10)

    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy:", acc)

    model.save("model.h5")


def recognize_test_data():
    model = tf.keras.models.load_model("model.h5")

    prediction = model.predict(x_test)
    print(prediction)

    plt.rcParams.update({"font.size": 6})
    plt.figure(figsize=(9, 9))
    for i in range(0, 8 * 2, 2):
        p = random.randint(0, len(y_test))
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[p], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(prediction[p]))

        plt.subplot(4, 4, i + 1 + 1)
        plt.bar(
            range(0, 10),
            prediction[p],
            color=["black" if np.argmax(prediction[p]) == y_test[p] else "red"],
        )

    plt.show()"""


def recognize(img):
    model = tf.keras.models.load_model("model.h5")
    prediction = model.predict(img)
    return prediction
