import random
from data import ImageDetectionsField, TextField
from data import COCO, DataLoader
from models.rstnet.language_model import LanguageModel

import torch
from torch.nn import NLLLoss
from tqdm import tqdm
import argparse
import os
import pickle
import numpy as np

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation', unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                # out = model(detections, captions)
                out, _ = model(captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    model.eval()
    scores = {}
    total_num = 0.
    correct_num = 0.
    with tqdm(desc='Epoch %d - evaluation', unit='ite', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                # out = model(detections, captions)
                out, _ = model(captions)
                captions = captions[:, 1:].contiguous()
                out = torch.argmax(out[:, :-1], dim=-1).contiguous()
                b_s, seq_len = out.size()
                total_num += float(b_s * seq_len)
                correct_num += float((out == captions).sum())
                pbar.update()

    scores['correct_num'] = correct_num
    scores['total_num'] = total_num
    scores['accuracy'] = correct_num / total_num
    return scores


def test():
    x = torch.ones(8, 16).long()
    # x = torch.tensor([3013, 4075, 46, 11, 628]).view(1, 5)
    model = LanguageModel(bert_hidden_size=768, vocab_size=10201)
    # model(x)
    out1, out2 = model(x)
    print(out1.size())      # [8, 16, 10201]
    print(out2.size())      # [8, 16, 768]


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Bert Language Model')
    parser.add_argument('--exp_name', type=str, default='bert_language1')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--features_path', type=str, default='X101-grid-coco_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='m2_annotations')

    args = parser.parse_args()
    print(args)

    print('Bert Language Model Testing')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if not os.path.isfile('vocab_language/vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_language/vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_language/vocab_%s.pkl' % args.exp_name, 'rb'))

    model = LanguageModel(padding_idx=text_field.vocab.stoi['<pad>'], bert_hidden_size=768, vocab_size=len(text_field.vocab)).to(device)

    # load model
    fname = 'saved_language_models/%s_best.pth' % args.exp_name
    if os.path.exists(fname):
        data = torch.load(fname)
        model.load_state_dict(data['state_dict'], strict=False)
        print('Resuming from epoch %d, validation loss %f, and best score %f' % (data['epoch'], data['val_loss'], data['best_score']))

    # Initial conditions
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])

    # Validation loss
    val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
    # Test loss
    test_loss = evaluate_loss(model, dataloader_test, loss_fn, text_field)

    # Validation scores
    val_scores = evaluate_metrics(model, dataloader_val, text_field)
    print("epoch {}: Validation scores", val_scores)
    # Test scores
    test_scores = evaluate_metrics(model, dataloader_test, text_field)
    print("epoch {}: Test scores", test_scores)
