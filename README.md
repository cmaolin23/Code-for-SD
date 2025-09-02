# Project Components

This project contains three main components for graph diversity optimization:

1. **Exact Algorithm (`exact`)** – implemented in **C++**  
2. **Greedy Algorithms (`greedy`)** – three heuristic approaches, implemented in **C++**  
3. **Reinforcement Learning (`RL`)** – an A2C-based method, implemented in **Python + PyTorch**

---

## 🔹 Exact Algorithm (`exact`)

### File
`diversity_exact_gcc8.cpp`

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

2. **Query file** (`test_queries.txt`):
12
45
78


### Usage
```bash
g++ -O2 diversity_exact_gcc8.cpp -o exact
./exact <graph.txt> <tau> <b> --test test_queries.txt

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


