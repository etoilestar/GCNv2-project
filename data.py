import torch
import pandas as pd
import random
#import albumentations
import os
import cv2
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from config import params

def get_mean_var(img):
    mean = np.mean(img)
    var = np.sqrt(np.sum((img-mean)**2))
#    print(mean, var)
    return mean, var

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 21)

    y, x = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

class Warp:
    """warp变换：
    输入:
    q，t:相机的位姿矩阵，q：（a, b, c, w）,t:(x, y, z)
         point:输入坐标点矩阵，size:(2,)
    中间变量:
    self.Pi, self.Pi_1: 映射矩阵及其逆
    self.M1，self.M2，self.M1_1:四元数转化成的旋转矩阵及其逆
    输出:
    point_out:输出坐标点矩阵，size:(2,)
    """
    def __init__(self, img_size):
        self.focal = float(500)
        self.Pi = np.array([[self.focal,    0,          img_size[1]/2],
                            [0,             self.focal, img_size[0]/2],
                            [0,             0,          1]],dtype=np.float32)
        self.Pi_1 = np.linalg.inv(self.Pi)

    def _Transform(self, q, t):
        qua = q.copy()
        q = np.outer(qua, qua)
        rot_matrix = (np.array([[0.5 - q[1][1] - q[2][2], q[0][1] + q[2][3], q[0][2] - q[1][3]],
                               [q[0][1] - q[2][3], 0.5 - q[0][0] - q[2][2], q[1][2] + q[0][3]],
                               [q[0][2] + q[1][3], q[1][2] - q[0][3], 0.5 - q[0][0] - q[1][1]]], dtype=np.float32).T)*2
        T_matrix = np.expand_dims(t, axis=-1)
        Transform_extend = np.concatenate((np.concatenate((rot_matrix,T_matrix), axis = 1),np.array([[0, 0, 0, 1]],dtype = np.float32)),axis = 0)
        return Transform_extend

    def load_M1(self, q1, t1):
        self.M1 = self._Transform(q1, t1)
        self.M1_1 = np.linalg.inv(self.M1)

    def load_M2(self, q2, t2):
        self.M2 = self._Transform(q2, t2)

    def __call__(self, point):
        self.point = np.expand_dims(point, axis=-1)
        V3d = np.dot(self.Pi_1, np.concatenate((self.point, np.expand_dims(np.array([1],dtype = np.float32), -1)), axis=0))
        V4d_1 =np.dot(self.M1_1, np.concatenate((V3d, np.expand_dims(np.asarray([1],dtype = np.float32), -1)), axis=0))
        V4d_2 = np.dot(self.M2, V4d_1)
        V3d = V4d_2[[0,1,2],:]
        V3d = np.dot(self.Pi, V3d)
        V3d = V3d[[0,1], :]/V3d[-1]
        point_out = np.round(V3d)
        point_out = point_out.reshape(point.shape)
#        print(np.all(point_out == point))
        return point_out

"""选择数据增强方式"""
#AUGMENTATIONS_TRAIN = albumentations.Compose([
        #albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE),
#        albumentations.OneOf([
#            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
#            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
#        ]),
#        albumentations.OneOf([
#            albumentations.Blur(blur_limit=4, p=1),
#            albumentations.MotionBlur(blur_limit=4, p=1),
#            albumentations.MedianBlur(blur_limit=4, p=1)
#        ], p=0.5),
        # albumentations.HorizontalFlip(p=0.5),
        # albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
        #                                 interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
#        albumentations.ToFloat(max_value=255.0,p=1.0)
#],p=1)

#AUGMENTATIONS_TEST =  albumentations.Compose([
#        albumentations.ToFloat(max_value=255.0,p=1.0)
#],p=1)


class MYdata(Dataset):
    def __init__(self, image_root, label_file, gt_file, size=(640, 480), mode='train'):
        self.size = size
        self.mode = mode
        self.image_path = image_root
        image_class = sorted(os.listdir(image_root))
        """在图片列表中选取其中一帧，并把其下一帧作为参考帧，组合成新的列表"""
        self.original_img_list = []
        original_img_list_1 = []
        for path in image_class:
            path_list = sorted(os.listdir(os.path.join(image_root,path)))
            path_list = list(map(lambda x, y: y+'-'+x, path_list, [path]*len(path_list)))
            self.original_img_list += path_list
            original_img_list_1 += path_list[:-1]
        random.shuffle(original_img_list_1)
#        print(self.original_img_list)
        if self.mode == 'train':
            self.mylist = original_img_list_1[:int(0.99*len(original_img_list_1))]
        else:
            self.mylist = original_img_list_1[int(0.99*len(original_img_list_1)):]

        print(self.mode, 'data number:', len(self.mylist))

        """gt和label的文件列表"""
        self.list = image_class
        self.label_file = label_file
        self.gt_file = gt_file

        """gt_array"""
        self.gt_arrays = {}
        for gt in os.listdir(gt_file):
            gt_image_name_list = []
            name, _ = os.path.splitext(gt)
            with open(os.path.join(gt_file, gt), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if l[0] == '#':
                        continue
                    gt_image_name = float(l.split()[0])
           #         print(gt_image_name)
                    gt_image_name_list.append(gt_image_name)
            gt_image_name_array = np.asarray(gt_image_name_list)
            self.gt_arrays[name] = gt_image_name_array

    def get_img(self, images):
        """获取img，并归一化"""
        img1 = cv2.resize(cv2.imread(os.path.join(self.image_path,images[0].split('-')[0],images[0].split('-')[1]), 0), self.size)
        img1 = np.array(img1, dtype=np.float32)
        img2 = cv2.resize(cv2.imread(os.path.join(self.image_path,images[1].split('-')[0],images[1].split('-')[1]), 0), self.size)
        img2 = np.array(img2, dtype=np.float32)
        mean1, var1 = get_mean_var(img1)
        mean2, var2 = get_mean_var(img2)
        assert not np.any(np.isnan(img2)), 'img error'
        return img1/255.0, img2/255.0

    def find_near(self, img):
        """在groundTruth中找到与图片帧时间戳最相近的qt值"""
        circum, f = img.split('-')
        float_img = float(f)
#        print(self.gt_arrays.keys())
        image_name_compare = np.abs(self.gt_arrays[circum] - float_img)
        loc = np.argmin(np.abs(image_name_compare))
        near_image = self.gt_arrays[circum][loc]
        return near_image, circum

    def decode(self,k):
        label_dir = []
        key_point_list = []
        flag = False
        if k[-1] == ' ':
            keypoints = k.split(' ')[:-1]
        else:
            keypoints = k.split(' ')
        assert len(keypoints)%2==0, 'keypoint size not match'
        for points in keypoints:
            key_point_list.append(float(points))
            if flag:
                key_point_array = np.asarray(key_point_list)
                label_dir.append(key_point_array.copy())
                key_point_list = []
            flag = not flag
        return label_dir

    def get_QT(self, near_image, circum):
        camera_list_float = []
        with open(os.path.join(self.gt_file, circum+'.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                if l[0] == '#':
                    continue
                gt_image_name = float(l.split()[0])
                if gt_image_name == near_image:
                    camera_list = l.split()[1:]
                    for c in camera_list:
                        camera_list_float.append(float(c))
                    t = np.array(camera_list_float[:3], dtype=np.float32)
                    q = np.array(camera_list_float[3:], dtype=np.float32)
                    return [t, q]

    def generate_mask(self,image, keypoint_str, warp, img1, addition_list=None):
        """生成前后两帧的mask"""
#        mask = np.zeros((self.size[1], self.size[0]), dtype=np.float32)
        img, _ = os.path.splitext(image)
        near_image, circum = self.find_near(img)
        if addition_list==None:
            label_list1= keypoint_str
            label_list = self.decode(label_list1)
            warp.load_M1(self.get_QT(near_image, circum)[1], self.get_QT(near_image, circum)[0])
            mask1 = mask2 = None
        else:
            addition_list = self.decode(addition_list)
            warp.load_M2(self.get_QT(near_image, circum)[1], self.get_QT(near_image, circum)[0])
            label_list = []
            for point in addition_list:
                label_list.append(warp(point))
#                label_list.append([warp([point[1], point[0]])[1],warp([point[1], point[0]])[0]])
            label_list1 = label_list
            mask1 = np.zeros((self.size[1], self.size[0]), dtype=np.float32)
            mask2 = np.zeros((self.size[1], self.size[0]), dtype=np.float32)
            for s, (i,j) in enumerate(zip(label_list,addition_list)):
                if int(float(i[1]))<mask2.shape[0] and int(float(i[0]))<mask2.shape[1] and int(float(i[1]))>=0 and int(float(i[0]))>=0:
                    mask2[int(float(i[1]))][int(float(i[0]))] = 1.0
                    mask1[int(float(j[1]))][int(float(j[0]))] = 1.0
                    draw_gaussian(mask1, [int(float(j[1])), int(float(j[0]))], 8)
                    draw_gaussian(mask2, [int(float(i[1])), int(float(i[0]))], 8)
                    mask2[int(float(i[1]))][int(float(i[0]))] += 0.001*s
                    mask1[int(float(j[1]))][int(float(j[0]))] += 0.001*s
                else:
                    mask1[int(float(j[1]))][int(float(j[0]))] = 1.0
                    draw_gaussian(mask1, [int(float(j[1])), int(float(j[0]))], 8)
                    mask1[int(float(j[1]))][int(float(j[0]))] = -1.0     
        return mask1, mask2, label_list1


    def get_mask(self, images, img1):
        """获取图片对应关键点的真值mask"""
        file_name = images[0].split('-')[0]
        read = pd.read_csv(os.path.join(self.label_file, file_name+'.csv'), header=None)
        warp = Warp(self.size)
        _, _, label_list1 = self.generate_mask(images[0],read[read[0].values == images[0].split('-')[1]][1].values[0], warp, img1)
        mask1, mask2, _ = self.generate_mask(images[1], read[read[0].values == images[1].split('-')[1]][1].values[0], warp, img1, addition_list=label_list1)
        return mask1, mask2

    def __getitem__(self, index):
        image1, image2 = self.get_img((self.mylist[index], self.original_img_list[self.original_img_list.index(self.mylist[index])+1]))
        mask1, mask2 = self.get_mask((self.mylist[index], self.original_img_list[self.original_img_list.index(self.mylist[index])+1]), image1)
#        if self.mode == 'train':
#            augmented1 = AUGMENTATIONS_TRAIN(image=image1, mask=mask1)
#            img1 = augmented1['image']
#            mask1 = augmented1['mask']

#            augmented2 = AUGMENTATIONS_TRAIN(image=image2, mask=mask2)
#            img2 = augmented2['image']
#            mask2 = augmented2['mask']
#        else:
#            augmented1 = AUGMENTATIONS_TEST(image=image1, mask=mask1)
#            img1 = augmented1['image']
#            mask1 = augmented1['mask']

#            augmented2 = AUGMENTATIONS_TEST(image=image2, mask=mask2)
#            img2 = augmented2['image']
#            mask2 = augmented2['mask']
        mask1 = np.expand_dims(mask1, axis=0)
        mask2 = np.expand_dims(mask2, axis=0)
        return image1, mask1, image2, mask2

    def __len__(self):
        return len(self.mylist)

if __name__ == '__main__':
    train_data = DataLoader(MYdata(params['imagepath'], 'data_keypoint1.csv', mode='train'),1, shuffle=False, num_workers=1)
    for step, (inputs1, labels1, inputs2, labels2) in enumerate(train_data):
        print(step)
        break
