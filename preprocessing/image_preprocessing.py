# import the necessary packages
import csv
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


class ImageResizer:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class FeatureExtraction:
	def __init__(self, csv_files):
		self.csv_data = csv_files
		self.list_of_features = []

	def preprocess(self, image):
		img_copy = image.copy()
		img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		dark_bright_threshold = 200
		mean_of_gray_image = np.mean(img_gray)
		if mean_of_gray_image < dark_bright_threshold:
			# Image is dark
			contour_recognition_threshold = 120
		else:
			# Image is bright
			contour_recognition_threshold = 200

		_, img_thresh = cv2.threshold(img_gray, contour_recognition_threshold, 255, cv2.THRESH_BINARY)
		kernel = np.ones((3, 3), np.uint8)
		img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
		contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.cv2.CHAIN_APPROX_NONE)

		im_boundary = (img_thresh.shape[0] - 1) * (img_thresh.shape[1] - 1)
		areas = [cv2.contourArea(ar) for ar in contours]
		cnt = [x for x in areas if x != im_boundary]
		cnt = contours[areas.index(max(cnt))]
		contour_area = cv2.contourArea(cnt)
		to_show_contour = img_copy.copy()
		cv2.drawContours(to_show_contour, cnt, -1, (0, 255, 0), 2, cv2.LINE_AA)


		rect = cv2.minAreaRect(cnt)
		rect_area = rect[1][0] * rect[1][1]
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		to_show_box = img_copy.copy()
		cv2.drawContours(to_show_box, [box], 0, (0, 0, 255), 2)

		hull = cv2.convexHull(cnt)
		hull_area = cv2.contourArea(hull)
		to_show_hull = img_copy.copy()
		cv2.drawContours(to_show_hull, [hull], 0, (255, 0, 0), 2)

		contour_perimeters = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.001 * contour_perimeters, True)
		approximation_area = cv2.contourArea(approx)
		to_show_approx = img_copy.copy()
		cv2.drawContours(to_show_approx, [approx], -1, (0, 0, 255), 3)

		corners = cv2.goodFeaturesToTrack(np.float32(img_gray), 100, 0.01, 10)
		corners = np.int0(corners)
		to_show_corners = img_copy.copy()
		for corner in corners:
			x, y = corner.ravel()
			cv2.circle(to_show_corners, (x, y), 3, (80, 127, 255), 2)

		h_corners = cv2.cornerHarris(np.float32(img_gray), 2, 3, 0.04)
		h_corners = np.int0(h_corners)
		to_show_corners_harris = img_copy.copy()
		h_threshold = 0.05
		for i in range(h_corners.shape[0]):
			for j in range(h_corners.shape[1]):
				if h_corners[i, j] > h_corners.max() * h_threshold:
					cv2.circle(to_show_corners_harris, (j, i), 1, (0, 0, 255), 1)

		amount_h_corners = len(h_corners[h_corners > h_corners.max() * h_threshold])

		ret = {
			"contour_points": len(cnt),
			"amount_contours": len(contours),
			"rect_area": rect_area,
			"hull_area": hull_area,
			"approximation_area": approximation_area,
			"contour_perimeters": contour_perimeters,
			"corners": len(corners),
			"harris_corners": amount_h_corners,
			"ratio_wide_length": rect[1][0] / rect[1][1],
			"contour_length_area_ratio": contour_perimeters / contour_area,
			"contour_length_rect_area_ratio": contour_perimeters / rect_area,
			"contour_length_hull_area_ratio": contour_perimeters / hull_area,
			"contour_rect_length_ratio": contour_perimeters / (2 * (rect[1][0] + rect[1][1])),
			"contour_hull_length_ratio": contour_perimeters / cv2.arcLength(hull, True),
			"extent": contour_area / rect_area,
			"solidity": contour_area / hull_area,
			# ""
			"hull_rectangle_ratio": hull_area / rect_area,
		}

		self.list_of_features.append(ret)

		return image

	def extract_to_table(self, labels, fmt="a"):
		le = LabelEncoder()
		labels = le.fit_transform(labels)
		for val in self.list_of_features:
			val["labels"] = labels[self.list_of_features.index(val)]
		keys = self.list_of_features[0].keys()
		with open(self.csv_data, fmt) as out:
			dict_writer = csv.DictWriter(out, keys)
			dict_writer.writeheader()
			for dat in self.list_of_features:
				dict_writer.writerow(dat)
			# dict_writer.writerows(self.csv_data)
