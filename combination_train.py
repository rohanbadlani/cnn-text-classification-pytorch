import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pandas as pd


def train(train_data, dev_data, model, args):
    if args.cuda:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    
    for epoch in range(1, args.epochs+1):
        for batch_idx in range(len(train_data[0])):
            #pdb.set_trace()
            #memReport()
            feature, target = train_data[0][batch_idx], train_data[1][batch_idx]
            feature, target = np.array(feature), np.array(target)
            batch_size = feature.shape[0]
            feature, target = torch.from_numpy(feature), torch.from_numpy(target)
            
            # feature.t_(), target.data.sub_(1)  # batch first, index align
            
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss, 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_data, model, args)
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
            del feature, target, logit, loss

            
def eval(data, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    total = 0
    
    if args.test:
        out_file = str(input("Output file name: "))
        if out_file in ["", "\n", None]:
            out_file = "out.csv"
    
    all_targets = np.array([])
    all_preds = np.array([])
    for batch_idx in range(len(data[0])):
        #cpuStats()
        #memReport()
        #pdb.set_trace()
        feature, target = data[0][batch_idx], data[1][batch_idx]
        feature, target = np.array(feature), np.array(target)
        # print(feature.shape, target.shape)
        batch_size = feature.shape[0]
        feature, target = torch.from_numpy(feature), torch.from_numpy(target)
        # feature.t_(), target.data.sub_(1)  # batch first, index align

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        
        logit = model(feature)        
        loss = F.cross_entropy(logit, target, reduction='mean')

        avg_loss += float(loss)
        
        targets = target.data
        predictions = torch.max(logit, 1)[1].view(target.size()).data
        corrects += (predictions == target.data).sum()
        total += target.data.shape[0]

        if args.test:
            cpu = True
            if cpu or (not args.cuda):
                targets, predictions = targets.cpu().numpy(), predictions.cpu().numpy()
            all_targets = np.concatenate((all_targets, targets))
            all_preds = np.concatenate((all_preds, predictions))
    
    df = pd.DataFrame(data={"targets": all_targets, "predictions": all_preds})
    df.to_csv(out_file)

    avg_loss /= total
    accuracy = 100.0 * corrects/total
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       total))
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
