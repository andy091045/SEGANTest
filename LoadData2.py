# from PIL import Image
# import pandas as pd
# import os
# import numpy as np
#
# # df = pd.read_csv(r'C:\Users\user\Desktop\crop2_data\images')
# df = pd.read_csv(r'D:\110598066\segan\crop2_data\data_csv\tr.csv')
# rawImageAdress = r'D:\110598066\segan\crop2_data\images'
# maskImageAdress = r'D:\110598066\segan\crop2_data\masks'
#
# imageR = Image.open(os.path.join(rawImageAdress, df['ImageId'][0])).convert('RGB')
# imageM = Image.open(os.path.join(maskImageAdress, df['MaskId'][15]))
# imageM = Image.fromarray(np.array(imageM)[:,:,1]).convert('P')
#
# # print(os.getcwd())
# imageM.show()
#----------------------------------------------------------------------------------------------
import os
import numpy as np
from glob import glob
from PIL import Image
import pandas as pd
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from transform import ReLabel, ToLabel, Scale, HorizontalFlip, VerticalFlip, ColorJitter
import random

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root=None):
        self.size = (180,135)
        self.root = root
        # if not os.path.exists(self.root):
        #     raise Exception("[!] {} not exists.".format(root))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(255, 1),
        ])
        #sort file names
        self.df = pd.read_csv(r'D:\110598066\segan\crop2_data\data_csv\tr.csv')
        self.input_paths = r'D:\110598066\segan\crop2_data\images'
        self.label_paths = r'D:\110598066\segan\crop2_data\masks'
        # self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("ISIC-2017_Training_Data"))))
        # self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.png'.format("ISIC-2017_Training_Part1_GroundTruth"))))
        # self.name = os.path.basename(root)
        # if len(self.input_paths) == 0 or len(self.label_paths) == 0:
        #     raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.input_paths, self.df['ImageId'][index])).convert('RGB')
        # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        label = Image.open(os.path.join(self.label_paths, self.df['MaskId'][index]))
        mask = np.array(label)[:,:,1]
        # mask = np.stack([mask]*3, axis=-1)
        label = Image.fromarray(mask)

        image = self.img_resize(image)
        # image_hsv = self.img_resize(image_hsv)
        label = self.label_resize(label)
        # brightness_factor = 1 + random.uniform(-0.4,0.4)
        # contrast_factor = 1 + random.uniform(-0.4,0.4)
        # saturation_factor = 1 + random.uniform(-0.4,0.4)
        # hue_factor = random.uniform(-0.1,0.1)
        # gamma = 1 + random.uniform(-0.1,0.1)

        #randomly flip images
        if random.random() > 0.5:
            image = HorizontalFlip()(image)
            # image_hsv = HorizontalFlip()(image_hsv)
            label = HorizontalFlip()(label)
        if random.random() > 0.5:
            image = VerticalFlip()(image)
            # image_hsv = VerticalFlip()(image_hsv)
            label = VerticalFlip()(label)

        #randomly crop image to size 128*128
        w, h = image.size
        th, tw = (128,128)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if w == tw and h == th:
            image = image
            # image_hsv = image_hsv
            label = label
        else:
            if random.random() > 0.5:
                image = image.resize((128,128),Image.BILINEAR)
                # image_hsv = image_hsv.resize((128,128),Image.BILINEAR)
                label = label.resize((128,128),Image.NEAREST)
            else:
                image = image.crop((x1, y1, x1 + tw, y1 + th))
                # image_hsv = image_hsv.crop((x1, y1, x1 + tw, y1 + th))
                label = label.crop((x1, y1, x1 + tw, y1 + th))
        # angle = random.randint(-20, 20)
        # image = image.rotate(angle, resample=Image.BILINEAR)
        # image_hsv = image_hsv.rotate(angle, resample=Image.BILINEAR)
        # label = label.rotate(angle, resample=Image.NEAREST)
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)


        label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.df['ImageId'])


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, root):
        size = (128,128)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

        ])
        self.hsv_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Scale(size, Image.NEAREST),
            ToLabel(),
            ReLabel(255, 1),
        ])
    #     #sort file names
    #     self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("ISIC-2017_Test_v2_Data"))))
    #     self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.png'.format("ISIC-2017_Test_v2_Part1_GroundTruth"))))
    #     self.name = os.path.basename(root)
    #     if len(self.input_paths) == 0 or len(self.label_paths) == 0:
    #         raise Exception("No images/labels are found in {}".format(self.root))
    #
    # def __getitem__(self, index):
    #     image = Image.open(self.input_paths[index]).convert('RGB')
    #     # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
    #     label = Image.open(self.label_paths[index]).convert('P')

        self.df = pd.read_csv(r'D:\110598066\segan\crop2_data\data_csv\tt.csv')
        self.input_paths = r'D:\110598066\segan\crop2_data\images'
        self.label_paths = r'D:\110598066\segan\crop2_data\masks'
        # self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("ISIC-2017_Training_Data"))))
        # self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.png'.format("ISIC-2017_Training_Part1_GroundTruth"))))
        # self.name = os.path.basename(root)
        # if len(self.input_paths) == 0 or len(self.label_paths) == 0:
        #     raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.input_paths, self.df['ImageId'][index])).convert('RGB')
        # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        label = Image.open(os.path.join(self.label_paths, self.df['MaskId'][index]))
        label = Image.fromarray(np.array(label)[:, :, 1])

        if self.img_transform is not None:
            image = self.img_transform(image)
            # image_hsv = self.hsv_transform(image_hsv)
        else:
            image = image
            # image_hsv = image_hsv

        if self.label_transform is not None:
            label = self.label_transform(label)
        else:
            label = label
        # image = torch.cat([image,image_hsv],0)

        return image, label

    def __len__(self):
        return len(self.df['ImageId'])



def loader(dataset, batch_size, num_workers=0, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader
