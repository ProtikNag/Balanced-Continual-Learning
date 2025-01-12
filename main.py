import csv
from task_specific_gcn import GraphClassificationGCN
from split_data import load_and_split
from bcl_model import BCLModel
from visualize import plot_combined_loss, plot_combined_acc, plot_taskwise_accuracy_progression
import random
import torch
import itertools

# Automate task sequence generation
def generate_task_sequences(num_tasks):
    sequences = [
        [7, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 7, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 7, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 7, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 7, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 7, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 7, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 8, 7, 9],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 7],
    ]
    return sequences

def calculate_average_acc(acc_by_task):
    total_acc = sum(acc_by_task.values())
    average_acc = total_acc / len(acc_by_task)
    return average_acc

def calculate_total_forgetting(task_accuracies):
    tasks = list(task_accuracies.values())
    t = len(tasks)
    if t < 2:
        return 0.0
    total_forgetting = 0.0
    for i in range(t - 1):
        forgetting = max(tasks[j] - tasks[t - 1] for j in range(i + 1))
        total_forgetting += forgetting
    average_forgetting = total_forgetting / (t - 1)
    return average_forgetting

def main():
    num_tasks = 10
    tasks_train, tasks_test, task_classes = load_and_split(batch_size=16, tasks=num_tasks)
    task_sequences = generate_task_sequences(num_tasks)

    csv_filename = "task_accuracy_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Sequence ID", "Task Accuracies", "Average Accuracy", "Average Forgetting"]
        writer.writerow(header)

        for i, task_order in enumerate(task_sequences):
            print(f"\nSequence {i + 1}: {task_order}")
            loss_by_task = {}
            acc_by_task = {}

            input_features = tasks_train[0].dataset[0].x.shape[1]
            output_classes = len(task_classes)
            model = GraphClassificationGCN(input_features=input_features, hidden_features=64, output_classes=output_classes)
            bcl_model = BCLModel(model)

            for task_id, task_index in enumerate(task_order):
                print(f"\nTraining on Task {task_index + 1} (Classes: {task_classes[task_index]})")
                train_task_loader = tasks_train[task_index]
                initial_loss, gen_loss, forget_loss = bcl_model.train_task(train_task_loader)
                accuracy_results = bcl_model.evaluate(tasks_test)
                loss_by_task[task_id] = {"initial_loss": initial_loss, "gen_loss": gen_loss, "forget_loss": forget_loss}
                acc_by_task[task_id] = accuracy_results

            average_accuracy = calculate_average_acc(acc_by_task)
            average_forgetting = calculate_total_forgetting(acc_by_task)
            print(f"Average Accuracy: {average_accuracy}")
            print(f"Average Forgetting: {average_forgetting}")

            # Convert task accuracies to a string for CSV writing
            task_accuracies_str = ", ".join(f"Task {task_id + 1}: {acc:.4f}" for task_id, acc in acc_by_task.items())

            # Write results to CSV
            writer.writerow([f"Sequence {i + 1}", task_accuracies_str, average_accuracy, average_forgetting])

            # Optional: Plot results
            # plot_combined_loss(loss_by_task, sequence_id=i + 1)
            # plot_combined_acc(acc_by_task, sequence_id=i + 1)
            # plot_taskwise_accuracy_progression(acc_by_task, sequence_id=i + 1)

if __name__ == "__main__":
    main()
