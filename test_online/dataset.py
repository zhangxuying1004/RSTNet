from torch.utils.data import Dataset
import json
import h5py
import numpy as np


class COCO_TestOnline(Dataset):
    def __init__(self, feat_path, ann_file, max_detections=49):
        """
        feat_path: COCO官方划分的训练集和验证集的特征路径
        ann_file: 训练集或验证集的标注信息，用于获取image_id，进而检索出对应特征
        """
        super(COCO_TestOnline, self).__init__()
        
        # 读取图像信息
        with open(ann_file, 'r') as f:
            self.images_info = json.load(f)['images']
            
        # 读取特征文件
        self.f = h5py.File(feat_path, 'r')    
        
        # 记录特征数目
        self.max_detections = max_detections

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_id = self.images_info[idx]['id']
        precomp_data = self.f['%d_grids' % image_id][()]

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return int(image_id), precomp_data
