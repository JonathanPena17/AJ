import re
import matplotlib.pyplot as plt

log_path = "golden_stdout.txt"

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
            current_epoch = None  # reset until the next epoch block

# Sanity check
if not epochs:
    raise RuntimeError("No epochs parsed. Check file path/format.")

# Sort by epoch in case they’re out of order
z = sorted(zip(epochs, train_acc, test_acc), key=lambda x: x[0])
epochs, train_acc, test_acc = map(list, zip(*z))

plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, marker='o', label='Train Accuracy')
plt.plot(epochs, test_acc, marker='s', label='Test Accuracy')
plt.title('LeNet-5 MNIST Training Progress (Golden Run)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('goldenrun.png', bbox_inches='tight')
