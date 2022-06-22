import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.rstnet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention

import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import time

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='RSTNet')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)

    parser.add_argument('--features_path', type=str, default='./Datasets/X101-features/X101-grid-coco_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./Datasets/m2_annotations')
    
    # the path of tested model and vocabulary
    parser.add_argument('--language_model_path', type=str, default='./saved_language_models/language_context.pth')
    parser.add_argument('--model_path', type=str, default='./saved_transformer_models/rstnet_best.pth')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl')
    args = parser.parse_args()

    print('The Evaluation of RSTNet')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open(args.vocab_path, 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'], language_model_path=args.language_model_path)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load(args.model_path)
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)
    print('it costs {} s to test.'.format(time.time() - start_time))
    
   
