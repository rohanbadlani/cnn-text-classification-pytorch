import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CombinationClassifier(nn.Module):

    def __init__(self, args):
        super(CombinationClassifier, self).__init__()
        # args = args
        embedding_size = args.embed_dim
        num_tasks = args.num_tasks
        hidden_dim = 50
        class_num = args.class_num
        
        self.fc1 = nn.Linear(num_tasks * embedding_size,
                             hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        
