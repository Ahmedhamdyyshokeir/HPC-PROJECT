from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

csv_path = os.path.expanduser("~/bioinfo_data/data.csv")

# Only master loads the data
if rank == 0:
    print("Master loading data...")
    df = pd.read_csv(csv_path)

    # Assume last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    chunk_size = len(X_train) // size
    remainders = len(X_train) % size
else:
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    chunk_size = None
    remainders = None

X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)
chunk_size = comm.bcast(chunk_size, root=0)
remainders = comm.bcast(remainders, root=0)

my_chunk_size = chunk_size + (1 if rank < remainders else 0)

if rank == 0:
    X_chunks, y_chunks = [], []
    start = 0
    for i in range(size):
        size_i = chunk_size + (1 if i < remainders else 0)
        X_chunks.append(X_train[start:start + size_i])
        y_chunks.append(y_train[start:start + size_i])
        start += size_i
else:
    X_chunks = None
    y_chunks = None

my_X = comm.scatter(X_chunks, root=0)
my_y = comm.scatter(y_chunks, root=0)

print(f"Rank {rank}: training on {len(my_X)} samples")

start_time = time.time()
clf = RandomForestClassifier(n_estimators=50, random_state=rank)
clf.fit(my_X, my_y)
my_pred = clf.predict(X_test)
my_acc = accuracy_score(y_test, my_pred)
train_time = time.time() - start_time

# Gather results
all_preds = comm.gather(my_pred, root=0)
all_accs = comm.gather(my_acc, root=0)
all_times = comm.gather(train_time, root=0)

if rank == 0:
    final_preds = np.zeros((len(y_test), 2))
    for pred in all_preds:
        for i, p in enumerate(pred):
            final_preds[i, p] += 1
    y_final = np.argmax(final_preds, axis=1)
    acc = accuracy_score(y_test, y_final)

    print("\n==== COVID Dataset MPI Results ====")
    print(f"Total samples: {len(X_train)} train, {len(X_test)} test")
    for i, (a, t) in enumerate(zip(all_accs, all_times)):
        print(f"Rank {i}: Accuracy = {a:.4f}, Time = {t:.2f}s")
    print(f"\nFinal Ensemble Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_final))
