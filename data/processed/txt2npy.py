import numpy as np


dataset = "20NG"


labels = []
with open(f"./data/processed/{dataset}/test_labels.txt", "r") as f:
    for line in f.readlines():
        labels.append(int(line[:len(line)-1]))

labels = np.array(labels)
np.save(f"./data/processed/{dataset}/test_labels.npy", labels)

labels = []
with open(f"./data/processed/{dataset}/train_labels.txt", "r") as f:
    for line in f.readlines():
        labels.append(int(line[:len(line)-1]))

labels = np.array(labels)
np.save(f"./data/processed/{dataset}/train_labels.npy", labels)
