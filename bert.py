from transformers import BertModel
import torch.nn as nn

class TextClassificationBERT(nn.Module):
    def __init__(self, num_tasks, num_classes_per_task):
        super(TextClassificationBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.task_heads = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, num_classes)
            for num_classes in num_classes_per_task
        ])

    def forward(self, input_ids, attention_mask, task_id, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use the CLS token representation
        logits = self.task_heads[task_id](pooled_output)  # Task-specific head
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}
