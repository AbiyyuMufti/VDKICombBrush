import numpy as np
import pandas as pd
from imutils import paths
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from ai_process import AiProcess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from preprocessing import SimpleDatasetLoader, ImageResizer
from pyimagesearch.preprocessing import ImageToArrayPreprocessor


class OurCNN(AiProcess):
    def __init__(self, height, width, depth, learning_rate):
        super().__init__()
        # !!! Creating Model !!!
        # initialize the model along with the input shape to be "channels last"
        self.model = Sequential()
        self.classes = 2
        input_shape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # define the first (and only) CONV => RELU layer
        self.model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        self.model.add(Activation("relu"))
        # softmax classifier
        self.model.add(Flatten())
        self.model.add(Dense(self.classes))
        self.model.add(Activation("softmax"))
        opt = SGD(learning_rate=learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
        print(self.model.summary())

    def fit(self, data, test_size):
        data = data.astype("float") / 255.0
        (self.train_value, self.test_value, self.train_labels, self.test_labels) = train_test_split(data, labels, test_size=test_size, random_state=42)
        le = LabelBinarizer()
        self.train_labels = le.fit_transform(self.train_labels)
        self.test_labels = le.fit_transform(self.test_labels)
        self.train_labels = np.hstack((self.train_labels, 1 - self.train_labels))
        self.test_labels = np.hstack((self.test_labels, 1 - self.test_labels))

    def train(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.H = self.model.fit(x=self.train_value, y=self.train_labels,
                                validation_data=(self.test_value, self.test_labels),
                                batch_size=batch_size, epochs=epochs, verbose=1)

    def plot_history(self):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()

    def review(self):
        accuracy = accuracy_score(self.test_labels.argmax(axis=1), self.prediction.argmax(axis=1))
        report = classification_report(self.test_labels.argmax(axis=1), self.prediction.argmax(axis=1), target_names=["brush", "comb"])
        conf_matrix = confusion_matrix(self.test_labels.argmax(axis=1), self.prediction.argmax(axis=1))
        return accuracy, report, conf_matrix

    def predict(self, test=None):
        if test is not None:
            self.test_value = test

        predictions = self.model.predict(self.test_value, batch_size=self.batch_size)
        self.prediction.append(predictions.argmax(axis=1))
        return self.prediction


if __name__ == '__main__':
    # my_data = pd.read_csv(r"D:\VDKICombBrush\800ImagesFeatures.csv")
    imagePaths = list(paths.list_images(r"D:/VDKICombBrush/resources/"))
    sp = ImageResizer(32, 32)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    cnn = OurCNN(32,32,3,0.01)
    cnn.fit(data, 0.1)
    cnn.train(30, 100)
    cnn.plot_history()
    cnn.predict()
    cnn.plot_history()
