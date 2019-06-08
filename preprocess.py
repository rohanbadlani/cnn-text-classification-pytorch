#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle
import pandas as pd
import json
import itertools
import pdb

d = []

def getData(inputFile, dataFile,n):
    funnyCount = 0
    notFunnyCount = 0

    for line in inputFile:
        #print "hey"
        data = json.loads(line)
        if data["votes"]["funny"] > 2: # and funnyCount < n:
            funnyCount += 1
            t = ('1',data["text"].encode('utf-8'))
            d.append(t)

        else: #if notFunnyCount < n:
            notFunnyCount += 1
            t = ('0',data["text"].encode('utf-8'))
            d.append(t)
            #else:
            #   if funnyCount >= n and notFunnyCount >= n:
            #       break

    print len(d)
    pickle.dump(d, dataFile)

    return d

def f(row):
    if row['funny'] > 2:
        return 1
    else:
        return 0

if __name__ == "__main__":

    outputFileName = "humour_train_new1.csv" #raw_input("output file name:")
    #n = input("number of datapoints needed for each class:")

    inputFile = "~/hate-speech-and-offensive-language/classifier/review.json"
    inputFile = "/home/mananrai/cnn-text-classification-pytorch/yelp-sentiment/train2.tsv" #raw_input("Input file: ") #open('dataset/yelp_academic_dataset_review.json','r')
    #dataFile = open(outputFileName, 'w')

    #data_file = inputFile
    '''
    with open(inputFile, 'r') as data_file:
            data = json.load(data_file)

            #print(data.keys())
            #del data['author']
            print dict(itertools.islice(items, 10))
            with open('datav2.json', 'w') as data_file:
                    data = json.dump(data, data_file)
    '''

    # read the entire file into a python array
    with open(inputFile, 'rb') as f:
        data = f.readlines()[:200000]

        # remove the trailing "\n" from each line
        data = map(lambda x: x.rstrip(), data)

        data_json_str = "[" + ','.join(data) + "]"

        # now, load it into pandas
        df = pd.read_json(data_json_str)

    #pdb.set_trace()
    #df = pd.read_json(inputFile)
    #df = df.iloc[:100000]
    df = df[['text', 'funny']]
    #df['label'] = df.apply(f, axis=1)
    #df = df[["text", "label"]]
    df["funny"][df["funny"] < 2] = 0
    df["funny"][df["funny"] >= 2] = 1
    df["text"] = df["text"].apply(lambda x: x.replace('\n', ' '))

    pos_count = df[df["funny"] == 1].cpu().data.numpy().count()
    neg_count = 2 * (200000 - pos_count)
    #grouped = df.groupby("funny")
    positives = df[df["funny"] == 1]
    negatives = df[df["funny"] == 0]
    negatives_filtered = df.sample(neg_count)
    #new_df = grouped[(grouped["funny"] == 1) or (grouped["funny"] == 0 and )]
    df = pd.concat(positives, negatives)
    df = df[["text", "funny"]]
    df.to_csv(outputFileName, encoding='utf-8') #loc[df['column_name'] == some_value]

    #inputFile.close()
    #dataFile.close()