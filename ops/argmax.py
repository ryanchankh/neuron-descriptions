
import torch
import torch.nn as nn
import torch.nn.functional as F



def hardargmax(logits):
    '''Return vector version of argmax. '''
    _, d = logits.shape
    y = torch.argmax(logits, 1)
    return onehot(y, d)


def onehot(y, num_classes):
    '''Turn discrete labels into onehot vectors. '''
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(y.device)
    return zeros.scatter(scatter_dim, y_tensor, 1)