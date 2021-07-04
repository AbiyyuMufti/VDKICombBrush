from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler

from ai_process import AiProcess
from collections import Counter
import pandas as pd


class OurKNearestNeighbour(AiProcess):
    def __init__(self, k, algorithm="euclidean"):
        super().__init__()
        self.K = k
        distance_type = {"manhattan": 1, "euclidean": 2}
        self.p = distance_type[algorithm.lower()]

    def predict(self, test=None):
        self.train_labels = self.train_data.labels
        self.train_value = self.train_data.drop('labels', axis=1)
        if test is None:
            self.test_labels = self.test_data.labels
            self.test_value = self.test_data.drop('labels', axis=1)
        else:
            self.test_value = test
        scaler = StandardScaler()
        self.train_value = scaler.fit_transform(self.train_value)
        self.test_value = scaler.transform(self.test_value)
        for i, test_point in enumerate(self.test_value):
            distance = []
            for each_train_val in self.train_value:
                distance.append(minkowski(test_point, each_train_val, self.p))
            df_dists = pd.DataFrame(data=distance, columns=["dist"],
                                    index=self.train_labels.index)
            df_nn = df_dists.sort_values(by=['dist'], axis=0)[:self.K]
            counter = Counter(self.train_labels[df_nn.index])
            self.prediction.append(counter.most_common()[0][0])
        return self.prediction


if __name__ == '__main__':
    my_data = pd.read_csv(r"D:\VDKICombBrush\800ImagesFeatures.csv")
    OKKN = OurKNearestNeighbour(5, "euclidean")
    OKKN.fit(my_data, 0.25)
    OKKN.predict()
    print(*OKKN.review())