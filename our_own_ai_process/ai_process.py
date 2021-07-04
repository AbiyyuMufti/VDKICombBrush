import abc
import pandas as pd
import utility
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class AiProcess(abc.ABC):
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.test_label = []
        self.prediction = []
        pass

    def fit(self, data: pd.DataFrame, test_size):
        self.train_data, self.test_data = utility.train_test_split(data, test_size)

    @abc.abstractmethod
    def predict(self, test=None):
        ...

    def review(self):
        accuracy = accuracy_score(self.testY, self.prediction)
        report = classification_report(self.testY, self.prediction, target_names=["brush", "comb"])
        conf_matrix = confusion_matrix(self.testY, self.prediction)
        return accuracy, report, conf_matrix
