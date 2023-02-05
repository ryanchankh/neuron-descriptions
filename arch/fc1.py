import torch.nn as nn
from ops.argmax import hardargmax



class FullyConnectedQuerier(nn.Module):
    def __init__(self, input_dim=16441, n_queries=16441, tau=None):
        super().__init__()
        self.input_dim = input_dim
        self.n_queries = n_queries
        
        # Architecture
        self.layer1 = nn.Linear(input_dim, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)

        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(1024)
        self.norm3 = nn.LayerNorm(1024)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        
        # Set starting temperature
        self.tau = tau
        
        # heads
        self.querier = nn.Linear(1024, self.n_queries)


    def update_tau(self, tau):
        self.tau = tau
        print(f'Changed temperature to: {self.tau}')


    def forward(self, mode, x, mask=None):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))
        query_logits = self.querier(x)
 
        # querying
        if mask is not None:
            query_logits = query_logits.masked_fill_(mask == 1, float('-inf'))
        query = self.softmax(query_logits / self.tau)
        query = (hardargmax(query_logits) - query).detach() + query
        return query