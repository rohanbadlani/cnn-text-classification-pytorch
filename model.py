import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        #pdb.set_trace()
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num

        Ci2 = Co
        Co2 = args.kernel_num #only one kernel applied.

        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks] # Putting D = 1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 1)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.convs2 = nn.ModuleList([nn.Conv2d(Ci2, Co2, (3, 1))])

        self.dropout = nn.Dropout(args.dropout)
        self.downsample = nn.Linear(Co2*D, 128)
        self.fc1 = nn.Linear(128, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        #pdb.set_trace()

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        #Conv
        x = [F.relu(conv(x)) for conv in self.convs1]  # [(N, Co, W, D), ...]*len(Ks)

        #MaxPool - apply in 1D over all the convolutions
        x = [F.max_pool2d(i, [2, 1]) for i in x]  # [(N, Co, W/2), ...]*len(Ks)

        #Concat
        x = torch.cat(x, 2)

        #Conv
        x = [F.relu(conv(x)) for conv in self.convs2] # [(N, Co2, W, D), ...]*len(Ks2=1)

        #Maxpool
        x = [F.max_pool2d(i, [i.size(2),1]).squeeze(2) for i in x]  # [(N, Co2, W/2, D), ...]*len(Ks2=1)

        #Concat
        x = x[0]

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        
        x = x.view(x.size(0), -1)

        x_drop = self.dropout(x) # (N, len(Ks)*Co)

        embedding = self.downsample(x)
        
        logit = self.fc1(embedding)  # (N, C)
        return logit, embedding
