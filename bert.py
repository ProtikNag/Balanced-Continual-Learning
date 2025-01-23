from transformers import BertModel
import torch.nn as nn
import torch


class TextClassificationBERTWithTaskEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_classes=2):
        super(TextClassificationBERTWithTaskEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.task_embeddings = nn.Embedding(embedding_dim, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, task_id, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use the CLS token representation
        task_id_tensor = torch.tensor([task_id], device=input_ids.device)  # Convert task_id to a tensor
        task_embedding = self.task_embeddings(task_id_tensor).squeeze(0)  # Remove the batch dimension
        task_specific_output = pooled_output + task_embedding  # Adjust representation with task embedding
        logits = self.classifier(task_specific_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}
