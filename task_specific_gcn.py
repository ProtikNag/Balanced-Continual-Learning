import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class GraphClassificationGCN(nn.Module):
    def __init__(self, input_features, hidden_features, output_classes=10, dropout=0.20):
        super(GraphClassificationGCN, self).__init__()

        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.conv3 = GCNConv(hidden_features, hidden_features)

        # Placeholder for dynamically determining the input size
        self.pooling_output_size = None

        # Fully connected layer (will be initialized dynamically)
        self.fc = None
        self.output_classes = output_classes
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # GCN Layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))

        # Global Pooling (Graph-level embedding)
        x_mean = global_mean_pool(x, batch)  # Mean Pooling
        x_max = global_max_pool(x, batch)   # Max Pooling
        x = torch.cat([x_mean, x_max], dim=1)  # Combine pooled features

        # Dynamically initialize the fully connected layer if not already done
        if self.fc is None:
            self.pooling_output_size = x.size(1)
            self.fc = nn.Linear(self.pooling_output_size, self.output_classes)
            self.fc.to(x.device)

        # Fully Connected Layer for Graph Classification
        x = self.fc(x)
        return x
