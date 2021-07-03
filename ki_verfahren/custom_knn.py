from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski


class CustomKNN:
    def __init__(self, k, dist_calculation="euclidean"):
        DISTANCE_METRIC = {
            "manhattan": 1,
            "euclidean": 2
        }
        self.np_data = None
        self.np_label = None
        self.order = DISTANCE_METRIC[dist_calculation]
        self.K = k

    def fit(self, data: pd.DataFrame, label: pd.DataFrame):
        self.data = data
        self.label = label
        self.np_data = data.to_numpy()
        self.np_label = label.to_numpy()

    def predict(self, test_data: pd.DataFrame):
        test_data = test_data.to_numpy()
        prediction = np.zeros(len(test_data))

        for i, cur_data in enumerate(test_data):
            distance = np.zeros(len(self.np_data))
            for j, each_data in enumerate(self.np_data):
                distance[i] = minkowski(cur_data, each_data, self.order)

            df_dists = pd.DataFrame(data=distance, columns=['dist'], index=self.label.index)
            df_nn = df_dists.sort_values(by=['dist'], axis=0)[:self.K]
            counter = Counter(self.label[df_nn.index])
            prediction[i] = counter.most_common()[0][0]
        return np.array(prediction)
