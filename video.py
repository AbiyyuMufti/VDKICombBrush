import cv2
import numpy as np
import pandas as pd
from imutils import paths

from our_own_ai_process import OurDecisionTree, OurRandomForrest, OurKNearestNeighbour, OurCNN
from preprocessing import ImageResizer, FeatureExtraction, SimpleDatasetLoader
from preprocessing.image_to_array import ImageToArrayPreprocessor
# listing all images inside the resources!
imagePaths = list(paths.list_images("resources/"))
# initialize the image resizer, load the dataset from disk, and reshape the data matrix
sp = ImageResizer(256, 256)
# another image resizer for data in cnn
sp2 = ImageResizer(32, 32)
# initialize the feature extractor that will save it to csv file
f_ext = FeatureExtraction()
# initialize the converter to array using keras, will be use in cnn
iap = ImageToArrayPreprocessor()
# initialize data set loader that will load the images and do the preprocessing before it
sdl = SimpleDatasetLoader(preprocessors=[sp, f_ext, sp2, iap])

print("[INFO] loading images...")
# load the images
(data, label) = sdl.load(imagePaths, verbose=100, show=20)

# extraction direct to panda data frame
df = f_ext.extract_to_panda(label)
df.sample(10)
# extract to the csv file
# uncomment to extract to a csv file
# f_ext.extract_to_table("features.csv", labels)

ODT = OurDecisionTree()
ODT.fit(df, 0.1)
ODT.train()
ODT.plot_tree()

ODT.predict()
print(*ODT.review())

ORF = OurRandomForrest()
ORF.fit(df, 0.1)
ORF.train()
ORF.predict()
print(*ORF.review())
#
fiveNearestNeighbour = OurKNearestNeighbour(5, "euclidean")
fiveNearestNeighbour.fit(df, 0.10)
fiveNearestNeighbour.predict()
print(*fiveNearestNeighbour.review())

cnn = OurCNN(32, 32, 3, 0.01, 50, 100)
cnn.fit((data, label), 0.25)
cnn.train()
cnn.plot_history()
cnn.predict()
print(*cnn.review())


def predictCNN(image):
    return cnn.predict(image)

def predictKNN(data_frame):
    return fiveNearestNeighbour.predict(data_frame)

def predictDT(data_frame):
    return ODT.predict(data_frame)

def predictRF(data_frame):
    return ORF.predict(data_frame)


if __name__ == '__main__':

    preprocess = [sp, f_ext, sp2, iap]
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cv2.namedWindow("Input")
    choice = ["Comb", "Brush"]
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        for p in preprocess:
            res_image = p.preprocess(frame, show=True, live=True)
        res_image = np.array([res_image])
        res_image = res_image.astype("float")/255.0
        # df = f_ext.extract_to_panda([1])
        # print(df.head())

        c = cv2.waitKey(1)
        if c == 27:
            break
        elif c == ord(' '):
            x = cnn.predict(res_image)
            print(choice(x))

    cap.release()
    cv2.destroyAllWindows()