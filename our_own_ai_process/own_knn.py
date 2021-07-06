from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler
from our_own_ai_process.ai_process import AiProcess
from collections import Counter
import pandas as pd


class OurKNearestNeighbour(AiProcess):
    def __init__(self, k, algorithm="euclidean"):
        """
        Initiate K-Nearest Neighbour approach of machine learning
        :param k: k number of neighbours
        :param algorithm: euclidean distance or manhattan
        """
        super().__init__()
        self.K = k
        distance_type = {"manhattan": 1, "euclidean": 2}
        self.p = distance_type[algorithm.lower()]

    def train(self, **kwargs):
        """K-Nearest Neighbour methode didn't really train data"""
        pass

    def predict(self, test=None, **kwargs):
        """
        :param test: test data to be predicted, if empty using stored test data
        :param kwargs: no additional kwargs needed
        :return: list of prediction
        """
        if test is not None:
            self.test_value = test
        # scale the pandas data frame using Scaler to an array
        scale = StandardScaler()
        self.train_value = scale.fit_transform(self.train_value)
        self.test_value = scale.transform(self.test_value)
        # for each test value the distance to each element will be calculated
        for i, test_point in enumerate(self.test_value):
            distance = []
            for each_train_val in self.train_value:
                distance.append(minkowski(test_point, each_train_val, self.p))
            # to make sorting of the distance easier using pandas data frame again
            df_dists = pd.DataFrame(data=distance, columns=["dist"],
                                    index=self.train_labels.index)
            # stored the k nearest neighbour in a variable
            df_nn = df_dists.sort_values(by=['dist'], axis=0)[:self.K]
            # counting the vote, to which category it has more neighbour
            counter = Counter(self.train_labels[df_nn.index])
            # the category with the most vote is the prediction
            self.prediction.append(counter.most_common()[0][0])
        return self.prediction


if __name__ == '__main__':
    # Example of using this class !
    my_data = pd.read_csv(r"D:\VDKICombBrush\ImagesFeatures.csv")
    fiveNearestNeighbour = OurKNearestNeighbour(5, "euclidean")
    fiveNearestNeighbour.fit(my_data, 0.10)
    fiveNearestNeighbour.predict()
    print(*fiveNearestNeighbour.review())
