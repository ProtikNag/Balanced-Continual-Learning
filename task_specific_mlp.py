# bcl/models/task_specific_mlp.py
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificMLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=128):
        super(TaskSpecificMLP, self).__init__()
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc2(x))

        return self.output_layer(x)


    # stability gap
