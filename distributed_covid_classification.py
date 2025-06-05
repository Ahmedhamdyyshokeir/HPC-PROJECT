from pathlib import Path

covid_script = """
from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

filename = "Covid_Data.xlsx"
sheet_name = "Covid Data"

if rank == 0:
    print("Reading COVID dataset...")
    df = pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl')

    # Drop rows with missing ICU values
    df = df.dropna(subset=["ICU"])

    # Separate features and label
    X = df.drop(columns=["ICU"])
    y = df["ICU"]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Prepare data splitting
    chunk_size = len(X) // size
    remainders = len(X) % size

    X_chunks = []
    y_chunks = []
    start = 0
    for i in range(size):
        end = start + chunk_size + (1 if i < remainders else 0)
        X_chunks.append(X[start:end])
        y_chunks.append(y[start:end])
        start = end
else:
    X_chunks = None
    y_chunks = None
    chunk_size = None
    remainders = None
    scaler = None

# Scatter data
my_X = comm.scatter(X_chunks, root=0)
my_y = comm.scatter(y_chunks, root=0)

print(f"Process {rank} training on {len(my_X)} samples.")

# Train classifier
start_time = time.time()
clf = RandomForestClassifier(n_estimators=50, random_state=rank)
clf.fit(my_X, my_y)
train_time = time.time() - start_time

# Gather predictions
my_preds = clf.predict(my_X)
my_acc = accuracy_score(my_y, my_preds)

all_preds = comm.gather(my_preds, root=0)
all_accs = comm.gather(my_acc, root=0)
all_times = comm.gather(train_time, root=0)

if rank == 0:
    print("\\n===== COVID ICU Classification Results =====")
    for i, (acc, t) in enumerate(zip(all_accs, all_times)):
        print(f"Process {i}: Accuracy = {acc:.4f}, Training Time = {t:.4f}s")

    print(f"Average Accuracy: {np.mean(all_accs):.4f}")
    print(f"Average Training Time: {np.mean(all_times):.4f}s")
"""

path = Path("/mnt/data/distributed_covid_classification.py")
path.write_text(covid_script)
path.name
