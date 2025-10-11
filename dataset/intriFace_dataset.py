'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 数据集元数据采集
'''

import os.path
from copy import deepcopy
import cv2
import math
import torch
import random
import re

import yaml
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

class IntriFaceDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        self.labels = self.data_dict['label']


    def __getitem__(self, index):
        image_path = self.data_dict['image'][index]
        flag = 0
        if 'Face2Face' in image_path or 'NeuralTextures' in image_path:
            flag = 1
        else:
            flag = 0
        envID = -1
        if '\\' in image_path:
            per = image_path.split('\\')[-2]
        else:
            per = image_path.split('/')[-2]
        

        try:
            if 'UADFV' in image_path or 'FaceForensics++' in image_path:
                if '_fake' in per:
                    source_index = int(per.split('_')[0])
                    target_index = source_index
                else:
                    if 'DeepFakeDetection' in image_path:
                        source_index = int(per.split('__')[0].split('_')[0])
                        target_index = int(per.split('__')[0].split('_')[1])
                    elif '__' in per:
                        source_index = int(per.split('__')[0])
                        target_index = source_index
                    else:
                        source_index = int(per)
                        target_index = source_index
            elif 'DFDC' in image_path:
                source_index = 0
                target_index = 0
            elif 'DFDCP' in image_path:
                if len(per.split('_')) == 4:
                    source_index = int(per.split('_')[0])
                    target_index = int(per.split('_')[1])
                    envID = int(per.split('_')[3])
                else:
                    source_index = per.split('_')[0]
                    target_index = source_index
            elif 'Celeb' in image_path:
                pattern = r'id\d+_id\d+'
                match = re.search(pattern, per)
                if not match:
                    target_index = int(per.split('_')[0][2:])
                    source_index = target_index
                else:
                    target_index = int(per.split('_')[0][2:])
                    source_index = int(per.split('_')[1][2:])
                envID = int(per.split('_')[-1])
            else:
                if '_' in per:
                    target_index = int(per.split('_')[0]) 
                    source_index = int(per.split('_')[1])
                else: 
                    source_index = int(per)
                    target_index = source_index
        except ValueError as e:
            source_index = 0
            target_index = 0
        
        label = self.data_dict['label'][index]

        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            return self.__getitem__(0)
        
        image = np.array(image)
        image_trans, _, _ = self.data_aug(image)
        image_trans = self.normalize(self.to_tensor(image_trans))

        return source_index, target_index, image_trans, label, flag, envID

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """

        source_indexes, target_indexes, image_trans, label, flag, envID = zip(*batch)

        # Stack the tensors
        images = torch.stack(image_trans, dim=0)
        labels = torch.LongTensor(label)
        source_ids = torch.LongTensor(source_indexes)
        target_ids = torch.LongTensor(target_indexes)
        flags = torch.LongTensor(flag)
        envIDs = torch.LongTensor(envID)
        
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['source_index'] = source_ids
        data_dict['target_index'] = target_ids
        data_dict['flag'] = flags
        data_dict['env_id'] = envIDs
        data_dict['mask'] = None
        data_dict['landmark'] = None
        return data_dict


def draw_landmark(img,landmark):
    draw = ImageDraw.Draw(img)

    # landmark = np.stack([mean_face_x, mean_face_y], axis=1)
    # landmark *=256
    for i, point in enumerate(landmark):
        draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 0, 0))
        draw.text((point[0], point[1]), str(i), fill=(255, 255, 255))
    return img

