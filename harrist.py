import cv2 as cv
import numpy as np
import os
import csv


def rename(path):
    i = 0
    for im in os.listdir(path):
        new_name = "PC{}".format(i)
        print(im)
        i = i + 1
        os.rename(im, new_name)


def corner_harist(img):
    gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    new_im = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
    new_im[dst > 0.01 * dst.max()] = 1

    # cv.imshow("orig", new_im)
    # result is dilated for marking the corners, not important
    # dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    # cv.imshow("inimage", img)
    return new_im


path = r"C:\Users\akangmufti\PycharmProjects\TestKamBruest\images\train"

if __name__ == '__main__':
    i = 0
    anzahl = []
    name = []
    for im in os.listdir(path):
        im_path = os.path.join(path, im)
        if os.path.isfile(im_path):
            src_img = cv.imread(im_path)
            src_img = cv.resize(src_img, (400, 400))
            new_im = corner_harist(src_img)
            name.append(im)
            anzahl.append(new_im.sum())
            # cv.imshow("img_red{}".format(i), src_img)
            # cv.imshow("img{}".format(i), src_img)
            i = 1 + i
    # cv.waitKey()
    print(name)
    print(anzahl)
    res = zip(name, anzahl)
    res = [*res]
    print(res)
    with open('data.csv', 'w', newline='\n') as file:
        datawriter = csv.writer(file, delimiter=';')
        datawriter.writerows(res)
