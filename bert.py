from transformers import BertModel
import torch.nn as nn


class TextClassificationBERTWithTaskEmbedding(nn.Module):
    def __init__(self, num_tasks, num_classes):
        super(TextClassificationBERTWithTaskEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.task_embeddings = nn.Embedding(num_tasks, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, task_id, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use the CLS token representation
        task_embedding = self.task_embeddings(task_id).squeeze(1)
        task_specific_output = pooled_output + task_embedding  # Adjust representation with task embedding
        logits = self.classifier(task_specific_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}
