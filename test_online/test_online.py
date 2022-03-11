
import torch
import argparse
import pickle
import numpy as np
import itertools
import json
from dataset import COCO_Test

import sys
sys.path.append('../')
from data import TextField
from models.rstnet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, TransformerEnsemble

import random
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

from torch.utils.data import DataLoader


if __name__ == '__main__':

    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)

    # 测试集
    parser.add_argument('--features_path', type=str, default='/home/zhangxuying/DataSet/COCO/test_feats_X101.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./cocotest2014.json')
    # # 验证集
    # parser.add_argument('--features_path', type=str, default='/home/DATA/X101_grid_feats_coco_trainval.hdf5')
    # parser.add_argument('--annotation_folder', type=str, default='/home/DATA/m2_annotations/captions_val2014.json')

    # parser.add_argument('--image_ids_path', type=str, default='coco_test_image_id.json')
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    # text_field.vocab = pickle.load(open('../vocab_transformer/vocab_transformer_grid_adaptive9.pkl', 'rb'))
    text_field.vocab = pickle.load(open('../vocab.pkl', 'rb'))

    dataset = COCO_Test(feat_path=args.features_path, ann_file=args.annotation_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    # # 加载模型
    # model_path = [
    #     # '../test_online_models/x101/captioning_model_x101_1.pth',
    #     # '../test_online_models/x101/captioning_model_x101_2.pth',
    #     # '../test_online_models/x101/captioning_model_x101_3.pth',
    #     # '../test_online_models/x101/captioning_model_x101_4.pth'

    #     '../test_online_models/x101/captioning_model_x101_2.pth',
    #     '../test_online_models/x101/captioning_model_x101_3.pth',
    #     '../test_online_models/x101/captioning_model_x101_4.pth',
    #     '../saved_transformer_models/transformer_grid_adaptive9_other_best_test.pth'
    # ]

    model_path = [
        '../test_online_models/x101/captioning_model_x101_2.pth',
        '../test_online_models/x101/captioning_model_x101_3.pth',

        '../saved_transformer_models/transformer_grid_adaptive9_other_32.pth',
        '../saved_transformer_models/transformer_grid_adaptive9_other_35.pth'

    ]
    ensemble_model = TransformerEnsemble(model=model, weight_files=model_path)

    # 生成结果
    outputs = []
    for it, (image_ids, images) in enumerate(iter(dataloader)):
        print('processing {} / {}'.format(it, len(dataloader.dataset) // args.batch_size))
        images = images.to(device)
        with torch.no_grad():
            out, _ = ensemble_model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        caps_gen = [' '.join([k for k, g in itertools.groupby(gen_i)]).strip() for gen_i in caps_gen]
        for i in range(image_ids.size(0)):
            item = {}
            item['image_id'] = int(image_ids[i])
            item['caption'] = caps_gen[i]
            outputs.append(item)

    # 保存结果
    output_path = 'results/captions_test2014_RSTNet_results.json'
    # output_path = 'results/captions_val2014_RSTNet_results.json'
    with open(output_path, 'w') as f:
        json.dump(outputs, f)
    
    print('finished!')
