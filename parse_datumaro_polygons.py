import json
import numpy
from PIL import Image, ImageDraw
import cv2
import numpy as np

from shapely.geometry import Polygon
import rasterio.features
import matplotlib.pyplot as plt

data = "/home/abs/abs/bisenet/data/annotations/default.json"

with open(data) as f:
  data = json.load(f)

# print(data)


for i in data["items"]:
    path = i["id"]
    cvimg = cv2.imread("data/images/"+path+".jpg")
    height, width = cvimg.shape[:2]
    annotations = i["annotations"]

    for annotation in annotations:
        this_polygon = annotation["points"]
        this_polygon = list(map(int, this_polygon))
        it = iter(this_polygon)
        this_polygon = [*zip(it, it)] 
        print(this_polygon)

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(this_polygon, outline=1, fill=255)


        mask = numpy.array(img)
        # print(type(mask))
        # print(type(mask))
        # cv2.imshow("1", mask)
        cv2.imwrite("data/annotations/"+path+".jpg", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


