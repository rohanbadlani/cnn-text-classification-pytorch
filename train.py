import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pdb
import psutil
import gc
import pandas as pd

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            #pdb.set_trace()
            #memReport()
            feature, target = batch.text, batch.label
            feature.t_(), target.data.sub_(1)  # batch first, index align
            if(feature.size()[1] < 5):
                continue
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, embedding = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss, 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
            del feature, target, logit, embedding, loss

            
def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    embeddings = []
    for batch in data_iter:
        #cpuStats()
        #memReport()
        #pdb.set_trace()
        feature, target = batch.text, batch.label
        feature.t_(), target.data.sub_(1)  # batch first, index align

        if(feature.size()[1] < 5):
            continue
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        
        logit, embedding = model(feature)        
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += float(loss)
        targets = target.data
        predictions = torch.max(logit, 1)[1].view(target.size()).data
        corrects += (predictions == target.data).sum()
        del feature, target, logit, loss

        if args.test:
            cpu = True
            if cpu:
                targets, predictions = targets.cpu(), predictions.cpu()

            out_file = "out.csv"
            df = pd.DataFrame(data={"targets:" targets, \
                                    "predictions": predictions})
            df.to_csv(out_file)
            del predictions, targets, df
            embeddings.extend(embedding.data)
            del embedding
        else:
           del embedding

        
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    if args.test:
        new_embeddings = []
        for idx, embed in enumerate(embeddings):
            # print(embed.type())
            new_embeddings.append(embed.detach().cpu().numpy())
        print(len(embeddings), len(embeddings[0]))
        np.save('./embeddings.npy', np.array(new_embeddings))
        del embeddings, new_embeddings
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output, embedding = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
