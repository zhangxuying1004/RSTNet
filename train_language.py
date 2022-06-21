import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader

from models.rstnet.language_model import LanguageModel

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import pickle
import numpy as np
from shutil import copyfile

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
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
    with tqdm(desc='Epoch %d - evaluation' % e, unit='ite', total=len(dataloader)) as pbar:
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


def train_xe(model, dataloader_train, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    # print('lr = {}'.format(scheduler.get_lr()[0]))
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
        for it, (detections, captions) in enumerate(dataloader_train):
            detections, captions = detections.to(device), captions.to(device)
            # out = model(detections, captions)
            out, _ = model(captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader_train)
    return loss


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Bert Language Model')
    parser.add_argument('--exp_name', type=str, default='bert_language')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=11328)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')

    parser.add_argument('--features_path', type=str, default='./Datasets/X101-features/X101-grid-coco_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./Datasets/m2_annotations')
    parser.add_argument('--dir_to_save_model', type=str, default='./saved_language_models')

    parser.add_argument('--logs_folder', type=str, default='./language_tensorboard_logs')
    args = parser.parse_args()
    print(args)

    print('Bert Language Model Training')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    model = LanguageModel(padding_idx=text_field.vocab.stoi['<pad>'], bert_hidden_size=768, vocab_size=len(text_field.vocab)).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        if s % 11331 == 0:
            s = 1
        else:
            s = s % 11331

        lr = (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
        if lr > 1e-6:
            lr = 1e-6

        print('s = {}, lr = {}'.format(s, lr))
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    # scheduler = StepLR(optim, step_size=2, gamma=0.5)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_score = .0
    best_test_score = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name)
        else:
            fname = os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_score = data['best_score']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best score %f' % (
                data['epoch'], data['val_loss'], data['best_score']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            break

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        val_scores = evaluate_metrics(model, dataloader_val, text_field)
        print("epoch {}: Validation scores", val_scores)
        val_score = val_scores['accuracy']
        writer.add_scalar('data/val_score', val_score, e)

        # Test scores
        test_scores = evaluate_metrics(model, dataloader_test, text_field)
        print("epoch {}: Test scores", test_scores)
        test_score = test_scores['accuracy']
        writer.add_scalar('data/test_score', test_score, e)

        # Prepare for next epoch
        best = False
        if val_score >= best_score:
            best_score = val_score
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_score >= best_test_score:
            best_test_score = test_score
            best_test = True

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                break
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best score %f' % (
                data['epoch'], data['val_loss'], data['best_score']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_score': val_score,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_score': best_score,
            'use_rl': use_rl,
        }, os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name))

        if best:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
        if best_test:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best_test.pth' % args.exp_name))

        if exit_train:
            writer.close()
            break
