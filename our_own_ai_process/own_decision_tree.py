import pandas as pd

from ai_process import AiProcess


class OurDecisionTree(AiProcess):
    def __init__(self):
        super().__init__()

    def predict(self, test=None):
        pass

    def plot_tree(self):
        pass




if __name__ == '__main__':
    if __name__ == '__main__':
        df = pd.read_csv('800ImagesFeatures.csv')
        ORF = OurDecisionTree()
        ORF.fit(df, 0.1)
        ORF.predict()
        ORF.review()