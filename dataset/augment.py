import cv2
import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import random
import matplotlib.pyplot as plt
from PIL import ImageFilter
import os
import ntpath
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imutils

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def crop_number_half(img, label):
    imgray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    src_crop_img = label.copy()
    src_crop_label = label.copy()
    imgh, imgw = label.shape[:2]
    print (imgw, imgh)
    # new_label = np.zeros((imgw, imgh, 3))
    print (len(contours))
    # contours = contours[1:]
    if len(contours)>0:
        ind = random.randint(0, len(contours)-1)
        cnt = contours[ind]
        # for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        print ( x,y,w,h )

        if w > h:
            crop_img = img[0:int(y+h/2), :]
            crop_label = label[0:int(y+h/2), :]

        else:
            crop_img = img[: ,0:int(x+w/2)]
            crop_label = label[: ,0:int(x+w/2)]

        crop_h, crop_w = crop_img.shape[:2]
        if (crop_h > imgh/2) and  (crop_w > imgw/2):
            img = crop_img.copy()
            label = crop_label.copy()

        tmp_new = cv2.resize(img, (500,500))
        tmp_old = cv2.resize(label, (500,500))
        cv2.imshow("label",tmp_new)
        cv2.imshow("old",tmp_old)
        cv2.waitKey()
        cv2.destroyAllWindows()


    return img, label



if __name__ == "__main__":
    dir_paths = [ "/home/ryzen/ai/data/coco/aadhaar_mask_long/images",
                     "/home/ryzen/ai/data/coco/udaan_actual/images",
                    "/home/ryzen/ai/data/coco/udaan_like_augmented/train/images",]
    label_path = [  "/home/ryzen/ai/data/coco/aadhaar_mask_long/annotations",
                   "/home/ryzen/ai/data/coco/udaan_actual/annotations",
                    "/home/ryzen/ai/data/coco/udaan_like_augmented/train/annotations"]
    image_list = []
    label_list = []
    for index, dir_paths_ in enumerate(dir_paths):
        for path, subdirs, files in os.walk(dir_paths_):
            for name in files:
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    filename_leaf = path_leaf(filename)
                    # print (os.path.join(dir_paths_, filename_leaf))
                    if os.path.isfile(os.path.join(dir_paths_, filename_leaf)):
                        # print (os.path.join(label_path[index], filename_leaf))
                        if os.path.isfile(os.path.join(label_path[index], filename_leaf)):
                            image_list.append(os.path.join(dir_paths_, filename_leaf))
                            label_list.append(os.path.join(label_path[index], filename_leaf))

    for img, label in zip(image_list, label_list):
        img = cv2.imread(img)
        label = cv2.imread(label)
        crop_number_half(img, label)