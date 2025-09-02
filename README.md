# Project Components

This project contains three main components for graph diversity optimization:

1. **Exact Algorithm (`exact`)** – implemented in **C++**  
2. **Greedy Algorithms (`greedy`)** – three heuristic approaches, implemented in **C++**  
3. **Reinforcement Learning (`RL`)** – an A2C-based method, implemented in **Python + PyTorch**

---

## 🔹 Exact Algorithm (`exact`)

### File
`diversity_exact.cpp`

### Language
C++ (tested with GCC 8+)

### Overview
The exact method computes the **maximum diversity improvement** by exploring all possibilities with a **Branch-and-Bound (BnB) algorithm**.  
It guarantees **optimal solutions** under the given parameters:

- Graph file (`graph.txt`)
- Threshold $\tau$ (minimum connected component size to count as "qualified")
- Budget $b$ (maximum number of edges allowed to add)
- Query nodes (from `test_queries.txt`)

### Key Components
- **Union-Find (DSU)**: maintains connected components  
- **Graph Reader**: loads graph, compresses node IDs, builds adjacency list  
- **Neighbor Subgraph Builder**: extracts query node’s neighbor subgraph  
- **Connected Components Search**: BFS/DFS to enumerate components  
- **Branch-and-Bound**: recursively partitions components into groups to maximize diversity score  

### Input
1. **Graph file** (edge list, e.g.):

  1 2

  2 3

  3 4


1. **Query file** (`test_queries.txt`):
   
12

45

78


### Usage
```bash
g++ -O2 diversity_exact_gcc8.cpp -o exact
./exact <graph.txt> <tau> <b> --test test_queries.txt
```

Optional arguments:

  · --trials T : number of trials (default = 1)
  
  · --seed S : random seed (default = 42)
  
  · --query ID : specify single query node
  
  · --directed : treat graph as directed
  

### Output
Results are written to <graph.txt>exact_result.txt.
Each line corresponds to a query node, showing:
  query_id   q0   q_new   inc   edge_num   edge_list
At the end of the file, global statistics are appended, including:
  sum of diversity gain
  total runtime (ms)

## 🔹 Greedy Algorithms (`greedy`)

### File
`greedy.cpp`

### Language
C++ (tested with GCC 8+, C++14)

### Overview
This module provides **three heuristic greedy algorithms** to approximate the diversity improvement:

  1. **Next Fit (NF)** – packs components sequentially until the threshold $\tau$ is reached.  
  2. **Simple (SI)** – packs components using the shortest prefix and fills from the tail.  
  3. **Improved Simple (ISI)** – a more refined grouping method using size-based partitioning.  

Although they **do not guarantee optimal solutions**, they are **much faster** than the exact BnB approach, making them suitable for large graphs.

### Key Components
- **Graph Reader**: reads graph, compresses node IDs, builds adjacency list  
- **Neighbor Subgraph Extraction**: builds subgraph of neighbors for each query node  
- **Connected Components**: computes components within the neighbor subgraph  
- **Greedy Packing**:  
  - NF, SI, ISI grouping strategies  
  - Plan edges within budget $b$  
- **Result Writer**: logs query results and summary statistics  

### Input
1. **Graph file** (edge list)  
2. **Query file** (e.g., `test.txt`)
The same as Exact. 

### Usage
Compile:
```bash
g++ -std=c++14 -O2 -Wall -Wextra -o greedy greedy.cpp
```
Run:
```bash
./greedy <graph.txt> <tau> <b> <method> --test <test.txt> [--seed S] [--trials T]
```

### Arguments

- `<graph.txt>` : input graph file  
- `<tau>` : threshold (minimum component size to count as qualified)  
- `<b>` : budget (max number of edges to add)  
- `<alg>` : choose greedy algorithm  
  - `1` = Next Fit (NF)  
  - `2` = Simple (SI)  
  - `3` = Improved Simple (ISI)  
- `--test file` : test query file  
- `--seed S` : random seed (default = 42)  
- `--trials T` : number of trials (default = 1)  
- `--query ID` : specify single query node (instead of reading test file)  

### Output

Results are written to:
  <graph.txt>Greedy_result.txt

Each query result includes:
Query q0= ... new_q= ... increase= ... time_us= ...

At the end, global statistics are appended:

- Total diversity increase  
- Total runtime (ms)  

