#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import pandas as pd
import combination_classifier as model
import combination_train as train
import numpy as np


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=500, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='final_snapshot_sarc_sentiment', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=5.0, help='l2 constraint of parameters [default: 5.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden-dim', type=int, default=50, help='hidden dimension')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

parser.add_argument('-num-tasks', type=int, default=2, help='number of tasks (default: 1 (sentiment))')
parser.add_argument('-embedding_files', type=list, default=['./dev_sentiment_embeddings.npy', './dev_sarc_sentiment_ds_embeddings.npy'],
                    help='path of the embedding files')
#  './dev_sarc_sentiment_ds_embeddings.npy'
# parser.add_argument('-embedding_file_2', type=str, default='./embeddings.npy', help='path of the second embedding file')
# parser.add_argument('-embedding_file_3', type=str, default='./embeddings.npy', help='path of the third embedding file')
parser.add_argument('-labels_file', type=str, default='./dev_sarc_sentiment_ds_labels.txt', help='path of the labels file')
parser.add_argument('-options', type=int, default=1, help='CSV (1) or TSV (2)')
parser.add_argument('-header', type=bool, default=True, help='Header in file or not')


args = parser.parse_args()

def load_embeddings(args):
    """
    Load embeddings and labels from file
    """
    
    num_tasks = args.num_tasks
    labels_file = args.labels_file
    fp = open(labels_file, "r")
    raw_labels = fp.readlines()
    labels = []
    for label in raw_labels:
        label = int(label[0])
        labels.append(label)
    # labels = label_df['label'].tolist()
    embeddings = np.load(args.embedding_files[0])
    for i in range(1, num_tasks):
        embedding = np.load(args.embedding_files[i])
        args.embed_dim = embedding.shape[1]
        embeddings = np.concatenate((embeddings, embedding), axis=1)
    return embeddings, labels


def load_and_batch_embeddings(args):
    """
    Load embeddings and labels from file
    and make batches out of them
    """

    embeddings, labels = load_embeddings(args)
    batch_size = args.batch_size
    print(len(labels))
    num_batches = len(labels) // batch_size
    data = [[], []]
    for i in range(num_batches - 1):
        data[0].append(embeddings[i*batch_size: (i+1)*batch_size])
        data[1].append(labels[i*batch_size: (i+1)*batch_size])
    return data


def split_data(data, dev_frac=0.1):
    """
    Split the data into training and dev data
    """
    
    dev_batch_num = int(len(data[1]) * dev_frac) + 1
    dev_data = [data[0][:dev_batch_num], data[1][:dev_batch_num]]
    train_data = [data[0][dev_batch_num:], data[1][dev_batch_num:]]
    # train_data = np.array(train_data)
    # dev_data = np.array(dev_data)
    return train_data, dev_data

    
data = load_and_batch_embeddings(args)
train_data, dev_data = split_data(data, dev_frac=0.1)
print(len(train_data), len(train_data[0]), len(dev_data), len(dev_data[0]), len(dev_data[0][0]), len(dev_data[0][0][0]))


args.class_num = 2

args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
combo_model = model.CombinationClassifier(args)

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    combo_model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    combo_model = combo_model.cuda()

    
# train or predict
if args.predict is not None:
    label = train.predict(args.predict, combo_model, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    # try:
    train.eval(data, combo_model, args) 
    # except Exception as e:
    #     print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_data, dev_data, combo_model, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

