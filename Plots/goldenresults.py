import re
import matplotlib.pyplot as plt

log_path = "/Users/jonathanpena/Desktop/AJ/Plots/golden_stdout.txt"

# Regular expressions to extract epochs and accuracies
epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
acc_re   = re.compile(r"Train Accuracy:\s*([0-9.]+)%\s*\|\s*Test Accuracy:\s*([0-9.]+)%")

epochs, train_acc, test_acc = [], [], []

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    current_epoch = None
    for line in f:
        m_epoch = epoch_re.search(line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            continue

        m_acc = acc_re.search(line)
        if m_acc and current_epoch is not None:
            epochs.append(current_epoch)
            train_acc.append(float(m_acc.group(1)))
            test_acc.append(float(m_acc.group(2)))
            current_epoch = None  

# Safety check
if not epochs:
    raise RuntimeError("No epochs parsed. Check file path/format.")

# Sort by epoch to ensure correct order
z = sorted(zip(epochs, train_acc, test_acc), key=lambda x: x[0])
epochs, train_acc, test_acc = map(list, zip(*z))

# Compute error rate from test accuracy
error_rate = [100 - acc for acc in test_acc]

# Create dual-axis plot
fig, ax1 = plt.subplots(figsize=(8,5))

# Accuracy (left y-axis)
ax1.plot(epochs, train_acc, marker='o', label='Train Accuracy', color='tab:blue')
ax1.plot(epochs, test_acc, marker='s', label='Test Accuracy', color='tab:green')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 100)
ax1.grid(True)

# Error rate (right y-axis)
ax2 = ax1.twinx()
ax2.plot(epochs, error_rate, marker='^', linestyle='--', label='Error Rate', color='tab:red')
ax2.set_ylabel('Error Rate (%)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 100)

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('LeNet-5 MNIST Golden Run: Accuracy vs Error Rate per Epoch')
plt.tight_layout()
plt.savefig('goldenrun_accuracy_error.png', bbox_inches='tight')
plt.show()
