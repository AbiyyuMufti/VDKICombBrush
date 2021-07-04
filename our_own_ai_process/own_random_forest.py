import pandas as pd

from ai_process import AiProcess


class OurRandomForrest(AiProcess):
    def __init__(self):
        super().__init__()

    def predict(self, test=None):
        pass


if __name__ == '__main__':
    df = pd.read_csv('800ImagesFeatures.csv')
    ORF = OurRandomForrest()
    ORF.fit(df, 0.1)
    ORF.predict()
    ORF.review()
