
from imutils import paths

from preprocessing import FeatureExtraction, ImageResizer
from preprocessing import SimpleDatasetLoader

args = {
	"dataset": "resources",
	"neighbors": 1,
	"jobs": -1
}


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = ImageResizer(256, 256)
f_ext = FeatureExtraction("800ImagesFeatures.csv")
sdl = SimpleDatasetLoader(preprocessors=[sp, f_ext])

if __name__ == '__main__':
	(data, labels) = sdl.load(imagePaths, verbose=10)
	f_ext.extract_to_table(labels, "w")
