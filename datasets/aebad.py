import os
from enum import Enum

import PIL
from glob import glob
import torch
from torchvision import transforms
from .mvtec import MVTecDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AeBADDataset(MVTecDataset):
    """
    PyTorch Dataset for AeBAD.
    """

    def load_dataset_folder(self):

        phase = 'train' if self.is_train else 'test'
        x = []  # img_paths
        y = []  # label(0 for good, 1 for anomaly)
        mask = []  # gt_paths

        for classname in [self.class_name]:
            classpath = os.path.join(self.dataset_path, classname, phase)
            maskpath = os.path.join(self.dataset_path, classname, "ground_truth")
            
            
            anomaly_types = [i for i in os.listdir(classpath) 
                            if os.path.isdir(os.path.join(classpath, i))]
            
            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                
                # get all images
                img_files = glob(os.path.join(anomaly_path, '**', '*.png'), recursive=True)
                img_files = sorted(img_files)  
                
               
                x.extend(img_files)
                
                
                if anomaly == "good":
                    
                    y.extend([0] * len(img_files))
                    mask.extend([None] * len(img_files))
                else:
                    
                    y.extend([1] * len(img_files))
                    
                    # 处理掩码路径（仅测试集生成掩码路径）
                    if phase in 'test':
                        mask_files = []
                        for img_path in img_files:
                            
                            # img_name = os.path.splitext(os.path.basename(img_path))[0]
                            # mask_path = os.path.join(maskpath, anomaly, img_name)
                            mask_path = os.path.join(maskpath, anomaly, img_path.split('/')[-2],img_path.split('/')[-1])
                            mask_files.append(mask_path)
                        mask.extend(mask_files)
                    else:
                        # 训练集无掩码
                        mask.extend([None] * len(img_files))
        
        return x, y, mask