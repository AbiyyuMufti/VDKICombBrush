from our_own_ai_process.ai_process import AiProcess
from imutils import paths
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from preprocessing import SimpleDatasetLoader, ImageResizer
from preprocessing.image_to_array import ImageToArrayPreprocessor
import numpy as np


class OurCNN(AiProcess):
    """ Simple CNN Model """
    def __init__(self, height, width, depth, learning_rate, batch_size, epochs, classes=2):
        """
        Creating Sequential models with the specified image dimension
        :param height: image height
        :param width: image width
        :param depth: image dept
        :param learning_rate: learning rate of the models using SGD
        :param batch_size: number of batch in one epochs
        :param epochs: number of epochs to train the models
        :param classes: number of categories (if we use this for other classification)
        """
        super().__init__()
        # !!! Creating Model !!!
        self.model = Sequential()
        self.classes = classes

        # initialize the model along with the input shape to be "channels last"
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

        # Stochastic Gradiant Decent as the loss function optimizer
        opt = SGD(learning_rate=learning_rate)

        # Because only 2 category, it is a binary
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

        # Print the model summary after successfully instantiate it
        print(self.model.summary())

        # Start values
        self.H = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_labels_2dim = None

    def fit(self, data_and_label: tuple, test_size):
        """
        Fit function direct from image data
        :param data_and_label: image data in specified dimension and the labels
        :param test_size: the factor split between test and training data set
        """
        input_data, labels = data_and_label
        input_data = input_data.astype("float") / 255.0
        (self.train_value, self.test_value, self.train_labels, self.test_labels_2dim) = train_test_split(input_data, labels, test_size=test_size, random_state=42)
        le = LabelBinarizer()

        # Binaries the labels so the dimension is compatible with the models
        self.train_labels = le.fit_transform(self.train_labels)
        self.test_labels_2dim = le.fit_transform(self.test_labels_2dim)
        self.train_labels = np.hstack((self.train_labels, 1 - self.train_labels))
        self.test_labels_2dim = np.hstack((self.test_labels_2dim, 1 - self.test_labels_2dim))

    def train(self, **kwargs):
        self.H = self.model.fit(x=self.train_value, y=self.train_labels,
                                validation_data=(self.test_value, self.test_labels_2dim),
                                batch_size=self.batch_size, epochs=self.epochs, verbose=1)

    def plot_history(self):
        try:
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
        except AttributeError as e:
            print("please train your data first!", e)

    def predict(self, test=None, *args, **kwargs):
        if test is not None:
            self.test_value = test
        predictions = self.model.predict(self.test_value, batch_size=self.batch_size)
        self.prediction = predictions.argmax(axis=1).tolist()
        self.test_labels = self.test_labels_2dim.argmax(axis=1).tolist()
        return self.prediction


if __name__ == '__main__':
    imagePaths = list(paths.list_images(r"D:/VDKICombBrush/resources/"))
    sp = ImageResizer(32, 32)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, label) = sdl.load(imagePaths, verbose=500)
    cnn = OurCNN(32, 32, 3, 0.01, 100, 500)
    cnn.fit((data, label), 0.25)
    cnn.train()
    cnn.plot_history()
    cnn.predict()
    cnn.plot_history()
    print(*cnn.review())
