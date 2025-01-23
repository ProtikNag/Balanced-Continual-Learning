import csv
import torch
import random
from split_data import load_and_split_text
from bert import TextClassificationBERTWithTaskEmbedding
from bcl_model import BCLModel


def calculate_average_acc(acc_by_task):
    total_acc = sum(acc_by_task.values())
    average_acc = total_acc / len(acc_by_task)
    return average_acc


def calculate_total_forgetting(task_accuracies):
    tasks = list(task_accuracies.values())
    t = len(tasks)

    if t < 2:
        return 0.0  # If there is only one task, no forgetting can occur

    total_forgetting = 0.0
    for i in range(t - 1):
        forgetting = max(tasks[j] - tasks[t - 1] for j in range(i + 1))
        total_forgetting += forgetting

    average_forgetting = total_forgetting / (t - 1)
    return average_forgetting


def remap_labels(dataset):
    """Remap dataset labels to binary 0 and 1."""
    unique_labels = list(set(dataset.labels))  # Access the labels directly
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    dataset.labels = [label_map[label] for label in dataset.labels]  # Remap labels in place
    return dataset


def main():
    # Load data
    tasks_train, tasks_test, task_classes = load_and_split_text(batch_size=32, tasks=14, max_samples_per_emotion=20)

    # Remap labels for binary classification
    for task_id in range(len(tasks_train)):
        train_dataset = tasks_train[task_id].dataset
        test_dataset = tasks_test[task_id].dataset
        remap_labels(train_dataset)  # Remap labels for training data
        remap_labels(test_dataset)  # Remap labels for testing data

    task_sequences = [random.sample(range(14), 14) for _ in range(5)]
    csv_filename = "bert_task_accuracy_results.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence ID", "Task Accuracies", "Average Accuracy", "Average Forgetting"])

        embedding_dim = 100  # Predefine the maximum number of task embeddings
        model = TextClassificationBERTWithTaskEmbedding(embedding_dim=embedding_dim)

        for i, task_order in enumerate(task_sequences):
            print(f"\nSequence {i + 1}: {task_order}")
            loss_by_task = {}
            acc_by_task = {}

            bcl_model = BCLModel(model)

            for task_id, task_index in enumerate(task_order):
                print(f"\nTraining on Task {task_index + 1}")

                train_task_loader = tasks_train[task_index]
                test_task_loader = tasks_test[task_index]

                # Train the model on the current task
                initial_loss, gen_loss, forget_loss = bcl_model.train_task(train_task_loader, task_id=task_index)

                # Evaluate accuracy on all tasks seen so far
                for t_id in range(task_id + 1):
                    test_loader = tasks_test[task_order[t_id]]
                    task_acc = bcl_model.evaluate(test_loader, task_id=task_order[t_id])
                    acc_by_task[task_order[t_id]] = task_acc

                loss_by_task[task_id] = {
                    "initial_loss": initial_loss,
                    "gen_loss": gen_loss,
                    "forget_loss": forget_loss,
                }

            average_accuracy = calculate_average_acc(acc_by_task)
            average_forgetting = calculate_total_forgetting(acc_by_task)

            writer.writerow([f"Sequence {i + 1}", acc_by_task, average_accuracy, average_forgetting])


if __name__ == "__main__":
    main()
