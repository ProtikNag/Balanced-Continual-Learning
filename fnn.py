import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, input_features, hidden_features, output_classes=1):
        super(FNN, self).__init__()

        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
