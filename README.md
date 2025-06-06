# Mini-HPC and Hybrid HPC-Big Data Clusters for COVID-19 Classification
# Distributed COVID-19 Classification Project

This project presents a practical exploration of distributed machine learning using two computational paradigms:
1. **Message Passing Interface (MPI)** – suited for high-performance computing (HPC).
2. **Apache Spark on Docker Swarm** – for scalable big data processing.

The datasets used involve **COVID-19 patient health data**, focusing on predicting ICU admission status using various clinical features.

---

## 🔍 Project Goals
- Apply distributed learning on real-world biomedical data.
- Compare efficiency and performance of MPI and Spark clusters.
- Develop scalable machine learning pipelines for health data analysis.

---

## 🖥️ Cluster Architectures

### Task 1: MPI Cluster Setup
A minimal HPC cluster is established using MPI with Python. This setup distributes model training across multiple CPUs.

```
[ Master Node ]---[ Worker Node 1 ]
       |
       +-------[ Worker Node 2 ]
```

### Task 2: Docker Swarm + Spark Setup
A hybrid approach using Spark and Jupyter in a Docker Swarm environment to enable scalable data processing.

```
+-----------------------------+
|     Docker Swarm Cluster    |
|  +-----------------------+  |
|  | Spark Master + Jupyter|  |
|  +-----------------------+  |
|     |             |         |
| [Worker 1]    [Worker 2]    |
+-----------------------------+
```

---

## 📂 Data Overview

- **Source**: COVID-19 Clinical Features Dataset
- **Label**: ICU Admission (0 = No, 1 = Yes)
- **Features**: Age, Fever, Cough, Blood metrics, etc.
- **Size**: ~500 entries

---

## 🧪 Experiments

### MPI (distributed_covid_classification.py)
- Parallel training of `RandomForestClassifier` on CPU clusters.
- Data is split and processed independently by each rank.
- Final predictions are ensembled using majority voting.

### Spark (distributed_covid_spark_analysis.py)
- Full pipeline with ChiSq feature selection, standardization, and training.
- Executed across a distributed Spark cluster.
- Results accessed via Spark UI and JupyterLab.

---

## 📈 Summary of Results

| Metric               | MPI                      | Spark                        |
|----------------------|--------------------------|------------------------------|
| Training Time        | ~0.05s                   | ~15–20s                      |
| Test Accuracy        | ~63–67%                  | ~65–70%                      |
| Scalability          | Excellent (CPU parallel) | Good (Big Data support)      |
| Feature Selection    | Manual (top k)           | ChiSqSelector (auto)         |
| Fault Tolerance      | Low                      | High                         |

---

## ⚙️ Setup Instructions

### Task 1: MPI Setup
```bash
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py numpy pandas scikit-learn
```
Hostfile:
```
master slots=1
worker1 slots=2
worker2 slots=2
```
Run:
```bash
mpirun -np 3 --hostfile hostfile python distributed_covid_classification.py
```

### Task 2: Spark + Docker Swarm
```bash
docker swarm init
docker stack deploy -c spark-stack.yml spark_cluster
```
Access:
- JupyterLab: http://<master-ip>:8888
- Spark UI: http://<master-ip>:8080

---

## 📘 Learning Outcomes

- Understanding MPI parallelism and synchronization.
- Applying Spark ML pipelines on health data.
- Building containerized distributed systems.
- Evaluating real-world biomedical data with ML.

---

## 📁 Project Files

```
project_hpc_covid_classification/
├── hostfile
├── distributed_covid_classification.py
├── distributed_covid_spark_analysis.py
├── spark-stack.yml
├── bioinfo_data/
│   └── Covid Data.csv
├── screenshots/
└── Final_Report.pdf
```

