import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from collections import Counter

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_and_split_text(batch_size, tasks=14, max_samples_per_emotion=50):
    # Load the GoEmotions dataset
    dataset = load_dataset("go_emotions")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Combine all splits for simplicity
    data = dataset['train']

    # Group data by labels (emotions) and limit samples per emotion
    emotion_data = {i: [] for i in range(28)}
    for i, label in enumerate(data['labels']):
        for emotion in label:
            if len(emotion_data[emotion]) < max_samples_per_emotion:
                emotion_data[emotion].append((data['text'][i], emotion))

    # Split emotions into tasks
    emotions_per_task = len(emotion_data) // tasks
    tasks_train, tasks_test = [], []
    task_classes = []

    for task_id in range(tasks):
        task_emotions = list(emotion_data.keys())[task_id * emotions_per_task: (task_id + 1) * emotions_per_task]
        task_classes.append(task_emotions)

        task_texts = []
        task_labels = []
        for emotion in task_emotions:
            task_texts.extend([item[0] for item in emotion_data[emotion]])
            task_labels.extend([emotion] * len(emotion_data[emotion]))

        # Shuffle and split into train/test sets
        indices = torch.randperm(len(task_texts)).tolist()
        task_texts = [task_texts[i] for i in indices]
        task_labels = [task_labels[i] for i in indices]

        train_size = int(0.8 * len(task_texts))
        train_texts = task_texts[:train_size]
        train_labels = task_labels[:train_size]
        test_texts = task_texts[train_size:]
        test_labels = task_labels[train_size:]

        # Print class distribution for the task
        print(f"Task {task_id + 1}: Classes = {task_emotions}")
        print(f"  Train samples per class: {dict(Counter(train_labels))}")
        print(f"  Test samples per class: {dict(Counter(test_labels))}")
        print(f"  Total train samples: {len(train_texts)}, Total test samples: {len(test_texts)}\n")

        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

        tasks_train.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        tasks_test.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

    return tasks_train, tasks_test, task_classes

def show_data_split():
    batch_size = 32
    tasks_train, tasks_test, task_classes = load_and_split_text(batch_size, tasks=14, max_samples_per_emotion=50)

    # Print summary of the task splits
    print("Summary of Tasks and Classes:")
    for i, classes in enumerate(task_classes):
        print(f"Task {i + 1}: {classes}")
    print("\nData loaders created successfully!")

    # Display data shapes for one batch from each task
    print("\nInspecting Data Shapes:")
    for i, train_loader in enumerate(tasks_train):
        print(f"\nTask {i + 1}:")
        for batch in train_loader:  # Get the first batch
            input_ids_shape = batch['input_ids'].shape
            attention_mask_shape = batch['attention_mask'].shape
            labels_shape = batch['labels'].shape

            print(f"  Input IDs Shape: {input_ids_shape}")
            print(f"  Attention Mask Shape: {attention_mask_shape}")
            print(f"  Labels Shape: {labels_shape}")
            break  # Only inspect the first batch of each task


# show_data_split()