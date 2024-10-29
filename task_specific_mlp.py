# bcl/models/task_specific_mlp.py
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificMLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=256, num_tasks=5):
        super(TaskSpecificMLP, self).__init__()
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        self.task_heads = nn.ModuleList([nn.Linear(hidden_size, 2) for _ in range(num_tasks)])

    def forward(self, x, task_id):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc2(x))
        return self.task_heads[task_id](x)
