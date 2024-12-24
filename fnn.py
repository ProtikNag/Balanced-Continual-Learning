import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_features, hidden_features, output_classes=1, dropout=0.20):
        super(FeedForwardNN, self).__init__()

        self.hidden_layer1 = nn.Linear(input_features, hidden_features)
        self.hidden_layer2 = nn.Linear(hidden_features, hidden_features)
        self.output_layer = nn.Linear(hidden_features, output_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, batch=None):
        # Fully connected feedforward layers
        x = F.relu(self.hidden_layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.hidden_layer2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_layer(x)  # Regression output for sine wave amplitude
        return x