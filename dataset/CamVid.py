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
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random
import matplotlib.pyplot as plt
from PIL import ImageFilter
import os
import ntpath
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imutils

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class CamVid(torch.utils.data.Dataset):
    def __init__(self, dir_paths, label_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = []
        self.label_list = []
        if not isinstance(dir_paths, list):
            dir_paths = [dir_paths]
        
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
                                self.image_list.append(os.path.join(dir_paths_, filename_leaf))
                                self.label_list.append(os.path.join(label_path[index], filename_leaf))


        # self.image_list.sort()
        # self.label_list.sort()

        z = list(zip(self.image_list, self.label_list))
        # print(z)
        random.seed(0)
        random.shuffle(z)
        print ("shuffled")
        z = z*100
        self.image_list, self.label_list = zip(*z)

        if mode=="train":
            self.image_list = self.image_list[: len(self.image_list)- 50]
            self.label_list = self.label_list[: len(self.label_list)- 50]
        else:
            self.image_list = self.image_list[len(self.image_list)-50: ]
            self.label_list = self.label_list[len(self.label_list)-50: ]

        print (len(self.image_list))
        print (len(self.label_list))
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        # self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_list]
        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)
        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def crop_number_half(self, img, label):
        imgray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        src_crop_img = label.copy()
        src_crop_label = label.copy()
        imgh, imgw = label.shape[:2]
        # print (imgw, imgh)
        # new_label = np.zeros((imgw, imgh, 3))
        # print (len(contours))
        # contours = contours[1:]
        if len(contours)>0:
            ind = random.randint(0, len(contours)-1)
            cnt = contours[ind]
            # for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)

            # print ( x,y,w,h )

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

            # tmp_new = cv2.resize(img, (500,500))
            # tmp_old = cv2.resize(label, (500,500))
            # cv2.imshow("label",tmp_new)
            # cv2.imshow("old",tmp_old)
            # cv2.waitKey()
            # cv2.destroyAllWindows()


        return img, label

    def proposal_net(self, label):
        imgray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        src_crop_img = label.copy()
        src_crop_label = label.copy()
        imgh, imgw = label.shape[:2]
        new_label = np.zeros((imgw, imgh, 3))
        # print (len(contours))

        if len(contours)>0:
            # cnt = random.choices(contours)

            for cnt in contours:
                # print (cnt)

                # for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)

                if w > h:
                    # print (1)
                    const = random.uniform(1, 1.1)
                    y_offset = int (h * const)
                    x_offset = int (w * 1)
                    new_label = cv2.rectangle(new_label, (max(0,x-x_offset), max(0,y-y_offset)),  
                                                        (min(x+w+x_offset, imgw), min(y+h+y_offset, imgh)), (255,255,255), -1)

                else:
                    # print (2)
                    const = random.uniform(1, 1.1)
                    x_offset = int (w * const)
                    y_offset = int (h * 1)
                    new_label = cv2.rectangle(new_label, (max(0,x-x_offset), max(0,y-y_offset)),  
                                                        (min(x+w+x_offset, imgw), min(y+h+y_offset, imgh)), (255,255,255), -1)

            # tmp_new = cv2.resize(new_label.copy(), (500,500))
            # tmp_old = cv2.resize(label.copy(), (500,500))
            # cv2.imshow("label",tmp_new)
            # cv2.imshow("old",tmp_old)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            label = new_label

        return label

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        # angle = random.choice([0,90,180,270])
        angle = 0

        img = Image.open(self.image_list[index])
        img = img.convert('RGB')



        # scale = random.choice(self.scale)
        scale = 1
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            pass
            # img = transforms.Resize(scale, Image.BILINEAR)(img)
            # img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # # =====================================

        img = np.array(img)
        if self.mode == 'train':
            img = imutils.rotate_bound(img, angle)
        # if bool(random.getrandbits(1)):
        # if random.choice([1,2,3]) == 1:
        #     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #     retval2,img = cv2.threshold(img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #     img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        if bool(random.getrandbits(1)):
            rand_rad = random.choice([3,5])
            # img = img.filter(ImageFilter.GaussianBlur(radius=rand_rad))
            kernel = np.ones((rand_rad,rand_rad),np.float32)/25
            img = cv2.filter2D(img,-1,kernel)

        # load label
        # print (self.label_list[index])
        label = Image.open(self.label_list[index])
        label = label.convert('RGB')


        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            pass
            # label = transforms.Resize(scale, Image.NEAREST)(label)
            # label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)
        if self.mode == 'train':
            label = imutils.rotate_bound(label, angle)
        # print ("imgsize", img.shape)
        # print ("labelsize", label.shape)
        # if self.mode == 'train':
        #     # if angle==0:
        #     img, label = self.crop_number_half(img, label)
        #         # cv2.imshow("1", img)
        #         # cv2.imshow("2", label)
        #         # cv2.waitKey(0)
        #         # cv2.destroyAllWindows()     

        # augment image and label
        # if self.mode == 'train':
        #     seq_det = self.fliplr.to_deterministic()
        #     img = seq_det.augment_image(img)
        #     label = seq_det.augment_image(label)


        if self.mode == 'train':
            resize = iaa.Scale({'height': 720, 'width': 720})
            resize_det = resize.to_deterministic()
            img = resize_det.augment_image(img)
            label = resize_det.augment_image(label)

        # if self.mode == 'train':
        #     label = self.proposal_net(label)




        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()


        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)
            # print ("imgsize", img.size())
            # print ("labelsize", label.size())
            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()
            # print (label.shape)
            # label = np.expand_dims(label, axis=0)
            # label = torch.unsqueeze(label, 0)


            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    pass

