# utils.py
import os
from collections import deque
from typing import List, Tuple, Dict

def read_edge_list(path: str) -> Tuple[List[int], Dict[int,int], List[List[int]]]:
    """
    Read an undirected edge list file (u v per line).
    Returns (id2raw, raw2id, adj)
    """
    edges = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            u = int(parts[0]); v = int(parts[1])
            edges.append((u,v))

    all_nodes = sorted({u for e in edges for u in e})
    raw2id = {raw:i for i,raw in enumerate(all_nodes)}
    id2raw = all_nodes
    n = len(all_nodes)
    adj = [[] for _ in range(n)]
    for u,v in edges:
        iu = raw2id[u]; iv = raw2id[v]
        adj[iu].append(iv)
        adj[iv].append(iu)
    return id2raw, raw2id, adj

def build_neighbor_subgraph(adj: List[List[int]], u: int) -> Tuple[List[int], List[List[int]], List[int]]:
    """
    Given global adjacency list and node index u (compressed id),
    return:
      neighbors: list of global neighbor node indices (N(u))
      subAdj: adjacency among neighbors (0..m-1)
      sub_to_global: map sub-index -> global index
    """
    neigh = list(set(adj[u]))
    neigh.sort()
    m = len(neigh)
    map_to_sub = {g:i for i,g in enumerate(neigh)}
    subAdj = [[] for _ in range(m)]
    for i,g in enumerate(neigh):
        for w in adj[g]:
            j = map_to_sub.get(w, -1)
            if j != -1:
                subAdj[i].append(j)
    return neigh, subAdj, neigh

def connected_components(subAdj: List[List[int]]) -> List[List[int]]:
    """
    BFS/DFS to get connected components of subAdj (indices 0..m-1)
    """
    n = len(subAdj)
    vis = [False]*n
    comps = []
    for i in range(n):
        if vis[i]:
            continue
        q = deque([i])
        vis[i] = True
        comp = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in subAdj[x]:
                if not vis[y]:
                    vis[y] = True
                    q.append(y)
        comps.append(comp)
    return comps

def comps_to_dicts(comps: List[List[int]], sub_to_global: List[int], subAdj: List[List[int]]) -> List[Dict]:
    """
    Convert component index-lists to dicts with size/nodes/edge_cnt/avg_deg/max_deg
    nodes are global indices.
    """
    result = []
    for comp in comps:
        nodes_global = [sub_to_global[i] for i in comp]
        # internal edge count
        edges = 0
        degs = []
        comp_set = set(comp)
        for i in comp:
            di = 0
            for j in subAdj[i]:
                if j in comp_set:
                    di += 1
            degs.append(di)
            edges += di
        edges = edges // 2
        avg_deg = float(sum(degs))/len(degs) if degs else 0.0
        max_deg = float(max(degs)) if degs else 0.0
        result.append({
            'size': len(comp),
            'nodes': nodes_global,
            'edge_cnt': int(edges),
            'avg_deg': float(avg_deg),
            'max_deg': float(max_deg)
        })
    return result

def top_k_unqualified(comps: List[Dict], tau: int, k: int) -> List[Dict]:
    """
    Return top-k largest components with size < tau (unqualified)
    """
    unq = [c for c in comps if c['size'] < tau]
    unq_sorted = sorted(unq, key=lambda x: x['size'], reverse=True)
    return unq_sorted[:k]

def diversity_count(comps: List[Dict], tau: int) -> int:
    return sum(1 for c in comps if c['size'] >= tau)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
