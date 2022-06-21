import os
import h5py
import argparse

import torch
import torch.nn as nn


class DataProcessor(nn.Module):
    def __init__(self):
        super(DataProcessor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x)    # [1, d, h, w] => [d, h, w] 
        x = x.permute(1, 2, 0)  # [d, h, w] => [h, w, d]
        return x.view(-1, x.size(-1))   # [h*w, d]


def process_dataset(file_path, feat_paths): 
    print('save the ori grid features to the features with specified size')
    # 加载特征处理器
    processor = DataProcessor()  
    with h5py.File(file_path, 'w') as f:
        for i in tqdm(range(len(feat_paths))):
            # 加载特征
            feat_path = feat_paths[i]
            img_feat = torch.load(feat_path) 
            # 处理特征
            img_feat = processor(img_feat)
            # 保存特征
            img_name = feat_path.split('/')[-1]
            img_id = int(img_name.split('.')[0])
            f.create_dataset('%d_grids' % img_id, data=img_feat.numpy()) 
        f.close()


def get_feat_paths(dir_to_save_feats, data_split='trainval', test2014_info_path=None):
    print('get the paths of raw grid features')
    ans = []
    # 线下训练和测试
    if data_split == 'trainval':
        filenames_train = os.listdir(os.path.join(dir_to_save_feats, 'train2014'))
        ans_train = [os.path.join(dir_to_save_feats, 'train2014', filename) for filename in filenames_train]
        filenames_val = os.listdir(os.path.join(dir_to_save_feats, 'val2014'))
        ans_val = [os.path.join(dir_to_save_feats, 'val2014', filename) for filename in filenames_val]
        ans = ans_train + ans_val
    # 线上测试
    elif data_split == 'test':
        assert test2014_info_path is not None
        with open(test2014_info_path, 'r') as f:
            test2014_info = json.load(f)

        for image in test2014_info['images']:
            img_id = image['id']
            feat_path = os.path.join(dir_to_save_feats, 'test2015', img_id+'.pth')
            assert os.path.exists(feat_path)
            ans.append(feat_path)
        assert len(ans) == 40775
    assert not ans      # make sure ans list is not empty
    return ans
    

def main(args):
    # 加载原始特征的绝对路径
    feat_paths = get_feat_paths(args.dir_to_raw_feats, args.data_split, args.test2014_info_path)
    # 构建处理后特征的文件名和保存路径
    file_path = os.path.join(args.dir_to_processed_feats, 'X101_grid_feats_coco_'+args.data_split+'.hdf5')
    # 处理特征并保存
    process_dataset(file_path, feat_paths)
    print('finished!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='data process') 
    parser.add_argument('--dir_to_raw_feats', type=str, default='./Datasets/X101-features/')
    parser.add_argument('--dir_to_processed_feats', type=str, default='./Datasets/X101-features/')
    # trainval = train2014 + val2014，用于训练和线下测试，test = test2014，用于线上测试
    parser.add_argument('--data_split', type=str, default='trainval')   # trainval, test
    # test2015包含test2014，获取test2014时，先加载test2014索引再加载特征，image_info_test2014.json是保存test2014信息的文件
    parser.add_argument('--test2014_info_path', type=str, default='./Datasets/image_info_test2014.json') 
    args = parser.parse_args()
        
    main(args)

