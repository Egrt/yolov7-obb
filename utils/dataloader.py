from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input
from utils.utils_rbox import poly2rbox, rbox2poly

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + 1 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, rbox  = self.get_random_data_with_Mosaic(lines, self.input_shape)
                
            if self.mixup and self.rand() < self.mixup_prob:
                lines           = sample(self.annotation_lines, 1)
                image_2, rbox_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, rbox      = self.get_random_data_with_MixUp(image, rbox, image_2, rbox_2)
        else:
            image, rbox      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        rbox        = np.array(rbox, dtype=np.float32)
        
        #---------------------------------------------------#
        #   对真实框进行预处理
        #---------------------------------------------------#
        nL          = len(rbox)
        labels_out  = np.zeros((nL, 7))
        if nL:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            rbox[:, [0, 2]] = rbox[:, [0, 2]] / self.input_shape[1]
            rbox[:, [1, 3]] = rbox[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            #---------------------------------------------------#
            labels_out[:, 1]  = rbox[:, -1]
            labels_out[:, 2:] = rbox[:, :5]
            
        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True, show=False):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(float,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2,4,6]] = box[:, [0,2,4,6]]*nw/iw + dx
                box[:, [1,3,5,7]] = box[:, [1,3,5,7]]*nh/ih + dy
                #------------------------------#
                #   将polygon转换为rbox
                #------------------------------#
                rbox          = np.zeros((box.shape[0], 6))
                rbox[..., :5] = poly2rbox(box[..., :8])
                rbox[..., 5]  = box[..., 8]
                keep = (rbox[:, 0] >= 0) & (rbox[:, 0] < w) \
                        & (rbox[:, 1] >= 0) & (rbox[:, 0] < h) \
                        & (rbox[:, 2] > 5) | (rbox[:, 3] > 5)
                rbox = rbox[keep]
            return image_data, rbox

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2,4,6]] = box[:, [0,2,4,6]]*nw/iw + dx
            box[:, [1,3,5,7]] = box[:, [1,3,5,7]]*nh/ih + dy
            if flip: box[:, [0,2,4,6]] = w - box[:, [0,2,4,6]]
            #------------------------------#
            #   将polygon转换为rbox
            #------------------------------#
            rbox          = np.zeros((box.shape[0], 6))
            rbox[..., :5] = poly2rbox(box[..., :8])
            rbox[..., 5]  = box[..., 8]
            keep = (rbox[:, 0] >= 0) & (rbox[:, 0] < w) \
                    & (rbox[:, 1] >= 0) & (rbox[:, 0] < h) \
                    & (rbox[:, 2] > 5) | (rbox[:, 3] > 5)
            rbox = rbox[keep]
        #------------------------------#
        #   检查旋转框
        #------------------------------#
        if show:
            draw  = ImageDraw.Draw(image)
            polys = rbox2poly(rbox[..., :5])
            for poly in polys:
                draw.polygon(xy=list(poly))
            image.show()
        return image_data, rbox
    
    def merge_rboxes(self, rboxes, cutx, cuty):
        merge_rbox = []
        for i in range(len(rboxes)):
            for rbox in rboxes[i]:
                tmp_rbox = []
                xc, yc, w, h = rbox[0], rbox[1], rbox[2], rbox[3]
                tmp_rbox.append(xc)
                tmp_rbox.append(yc)
                tmp_rbox.append(h)
                tmp_rbox.append(w)
                tmp_rbox.append(rbox[-1])
                merge_rbox.append(rbox)
        merge_rbox = np.array(merge_rbox)
        return merge_rbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4, show=False):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        rbox_datas  = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(float,box.split(',')))) for box in line_content[1:]])
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2,4,6]] = iw - box[:, [0,2,4,6]]
            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            rbox_data = []
            #---------------------------------#
            #   对rbox进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2,4,6]] = box[:, [0,2,4,6]]*nw/iw + dx
                box[:, [1,3,5,7]] = box[:, [1,3,5,7]]*nh/ih + dy
                #------------------------------#
                #   将polygon转换为rbox
                #------------------------------#
                rbox          = np.zeros((box.shape[0], 6))
                rbox[..., :5] = poly2rbox(box[..., :8])
                rbox[..., 5]  = box[..., 8]
                keep = (rbox[:, 0] >= 0) & (rbox[:, 0] < w) \
                        & (rbox[:, 1] >= 0) & (rbox[:, 0] < h) \
                        & (rbox[:, 2] > 5) | (rbox[:, 3] > 5)
                rbox = rbox[keep]
                rbox_data = np.zeros((len(rbox),6))
                rbox_data[:len(rbox)] = rbox
            
            image_datas.append(image_data)
            rbox_datas.append(rbox_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_rboxes = self.merge_rboxes(rbox_datas, cutx, cuty)
        #---------------------------------#
        #   检查旋转框
        #---------------------------------#
        if show:
            new_img = Image.fromarray(new_image) 
            draw    = ImageDraw.Draw(new_img)
            polys   = rbox2poly(new_rboxes[..., :5])
            for poly in polys:
                draw.polygon(xy=list(poly))
            new_img.show()
        return new_image, new_rboxes

    def get_random_data_with_MixUp(self, image_1, rbox_1, image_2, rbox_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(rbox_1) == 0:
            new_rboxes = rbox_2
        elif len(rbox_2) == 0:
            new_rboxes = rbox_1
        else:
            new_rboxes = np.concatenate([rbox_1, rbox_2], axis=0)
        return new_image, new_rboxes
    
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes
