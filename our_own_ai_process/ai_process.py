import abc
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from our_own_ai_process import utility

class AiProcess(abc.ABC):
    """ Abstract class for each AI Methods von our own implementation """
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.train_value = []
        self.train_labels = []
        self.test_value = []
        self.test_labels = []
        self.prediction = []

    def fit(self, data: pd.DataFrame, test_size):
        """
        Fit the data from pandas data frame to train and test data
        :param data: pd.DataFrame from the featured extracted csv data
        :param test_size: the percentage of the test sample from entire data
        """
        self.train_data, self.test_data = utility.train_test_split(data, test_size)
        self.train_labels = self.train_data.labels
        self.train_value = self.train_data.drop('labels', axis=1)
        self.test_labels = self.test_data.labels
        self.test_value = self.test_data.drop('labels', axis=1)

    @abc.abstractmethod
    def train(self, **kwargs):
        """ Abstract Methods that represent training process """
        ...

    @abc.abstractmethod
    def predict(self, test=None, **kwargs) -> list:
        """ Abstract Methods that represent the prediction process """
        ...

    def review(self):
        """
        Give the classification report and the confusion matrix
        :return: report and confusion matrix
        """
        try:
            # accuracy = accuracy_score(self.test_labels, self.prediction)
            report = classification_report(self.test_labels, self.prediction, target_names=["brush", "comb"])
            conf_matrix = confusion_matrix(self.test_labels, self.prediction)
            return report, conf_matrix
        except ValueError as e:
            print("Please do training or predicting first!", e)

