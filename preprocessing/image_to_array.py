from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    """ Pre Processor to conver image to an array using tensorflow preprocessing """
    def __init__(self, data_format=None):
        self.dataFormat = data_format

    def preprocess(self, image, **kwargs):
        return img_to_array(image, data_format=self.dataFormat)
