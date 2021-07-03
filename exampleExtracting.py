
from imutils import paths

from preprocessing import FeatureExtraction, ImageResizer
from preprocessing import SimpleDatasetLoader

args = {
	"dataset": "ressources2",
	"neighbors": 1,
	"jobs": -1
}


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = ImageResizer(256, 256)
f_ext = FeatureExtraction("Features2.csv")
sdl = SimpleDatasetLoader(preprocessors=[sp, f_ext])


(data, labels) = sdl.load(imagePaths, verbose=2)


keys = f_ext.list_of_features[0].keys()
print(keys)


feat = f_ext.list_of_features
print(feat)


if __name__ == '__main__':
	f_ext.extract_to_table(labels, "w")
