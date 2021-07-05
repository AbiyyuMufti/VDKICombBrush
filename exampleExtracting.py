from imutils import paths
from preprocessing import ImageResizer, FeatureExtraction
from preprocessing import SimpleDatasetLoader
import time


imagePaths = list(paths.list_images("resources/"))
# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = ImageResizer(256, 256)
f_ext = FeatureExtraction("ImagesFeatures2.csv")
sdl = SimpleDatasetLoader(preprocessors=[sp, f_ext])


if __name__ == '__main__':
	print("[INFO] loading images...")
	start1 = time.perf_counter()
	(data, labels) = sdl.load(imagePaths, verbose=100)
	end1 = time.perf_counter()
	start2 = time.perf_counter()
	df = f_ext.extract_to_table(labels)
	end2 = time.perf_counter()
	print("Loading Images:", end1 - start1)
	print("Create table Images:", end2 - start2)
