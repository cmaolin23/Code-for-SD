# README

## 📌 Project Overview
This project implements a **reinforcement learning framework based on A2C (Advantage Actor-Critic)** for graph diversity optimization.  
The main goal is:  
- Given a graph, a threshold $\tau$, and a budget $b$, add edges to maximize the diversity of a query node (i.e., the number of connected components in its neighbor subgraph).  
- Two evaluation modes are provided:  
  1. **Inference with a trained A2C model**  
  2. **Greedy baseline algorithm**  

This framework is suitable for research in graph-based combinatorial optimization and reinforcement learning.

---

## 📂 Project Structure
├── main.py # Main entry point
├── args.py # Command-line argument parser
├── utils.py # Utility functions (graph reading, subgraph, diversity calculation, etc.)
├── environment.py # GraphEnv environment (state, action, reward definitions)
├── model_a2c.py # Actor-Critic model definition
├── algorithm.py # A2CTrainer, training and inference logic
├── data/ # Data directory (graph and query files)
│ ├── graph.txt # Edge list of the graph
│ ├── train_queries.txt # Query nodes for training
│ └── test_queries.txt # Query nodes for testing
└── out/ # Output directory (results and models)

---

## ⚙️ Installation
Python 3.8+ is recommended.  

Install dependencies:
```bash
pip install torch
For GPU support:

pip install torch --index-url https://download.pytorch.org/whl/cu118

How to Run？
1. Train and Evaluate
python main.py \
    --graph ./data/graph.txt \
    --train ./data/train_queries.txt \
    --test ./data/test_queries.txt \
    --tau 5 \
    --b 10 \
    --train_epochs 100 \
    --emb_dim 64 \
    --hidden 128 \
    --lr 0.001 \
    --out_dir ./out
. Run Greedy Baseline Only (No Training)
python main.py \
    --graph ./data/graph.txt \
    --train ./data/train_queries.txt \
    --test ./data/test_queries.txt \
    --tau 5 \
    --b 10 \
    --train_epochs 0 \
    --out_dir ./out

Input File Formats
Graph File (graph.txt)

Undirected edge list, one edge per line:
1 2
2 3
3 4
... ...

Query Files (train_queries.txt / test_queries.txt)

One query node ID per line:
12
45
78
...

Output Files

After running, the following will be generated in the out/ directory:
a2c_model.pt: saved model parameters
train_results.txt / test_results.txt: detailed results for each query
Format:
query_id  q0  q_new  inc  edge_num  edge_list
dblp-loss.txt: training loss log
xxx_result.txt: experiment summary (training/testing improvements, runtime, etc.)

| Argument         | Description                                                         | Default |
| ---------------- | ------------------------------------------------------------------- | ------- |
| `--graph`        | Path to input graph file                                            | -       |
| `--train`        | Path to training query file                                         | -       |
| `--test`         | Path to testing query file                                          | -       |
| `--tau`          | Threshold for component size                                        | 5       |
| `--b`            | Budget (number of edges to add)                                     | 10      |
| `--train_epochs` | Number of training epochs (`0` = no training, greedy baseline only) | 0       |
| `--emb_dim`      | Node embedding dimension                                            | 64      |
| `--hidden`       | Hidden layer dimension                                              | 128     |
| `--lr`           | Learning rate                                                       | 0.001   |
| `--out_dir`      | Output directory                                                    | ./out   |

