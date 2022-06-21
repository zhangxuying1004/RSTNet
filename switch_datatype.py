# switch data from float32 to float16
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse


def main(args):

    data_splits = os.listdir(args.dir_to_save_feats)

    for data_split in data_splits:
        print('processing {} ...'.format(data_split))
        if not os.path.exists(os.path.join(args.dir_to_save_float16_feats, data_split)):
            os.mkdir(os.path.join(args.dir_to_save_float16_feats, data_split))

        feat_dir = os.path.join(args.dir_to_save_feats, data_split)
        file_names = os.listdir(feat_dir)
        print(len(file_names))

        for i in tqdm(range(len(file_names))):
            file_name = file_names[i]
            file_path = os.path.join(args.dir_to_save_feats, data_split, file_name)
            data32 = torch.load(file_path).numpy().squeeze()
            data16 = data32.astype('float16')
        
            image_id = int(file_name.split('.')[0])
            saved_file_path = os.path.join(args.dir_to_save_float16_feats, data_split, str(image_id)+'.npy')
            np.save(saved_file_path, data16)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='swith the data type of features') 
    parser.add_argument('--dir_to_raw_feats', type=str, default='./Datasets/X101-features/', help='big data')
    parser.add_argument('--dir_to_float16_feats', type=str, default='./Datasets/X101-features-float16', help='little data')
    args = parser.parse_args()
        
    main(args)

