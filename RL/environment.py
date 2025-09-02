# environment.py
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils import build_neighbor_subgraph, connected_components, comps_to_dicts, top_k_unqualified, diversity_count

class GraphEnv:
    """
    Environment for component merging.
    State: features of current unqualified components (list of dicts).
    Action: pair (p, q) indices in the current component list.
    Stopping condition:
      - budget exhausted OR
      - no possible future merge can produce a new qualified component,
        i.e. sum of top (remaining_budget + 1) sizes < tau
    """

    def __init__(self, adj: List[List[int]], query_u:int, tau:int, budget:int):
        self.adj = adj
        self.query_u = query_u
        self.tau = int(tau)
        self.budget0 = int(budget)
        self.ego_nodes = [] 
        self.reset_full()

    def reset_full(self):
        """
        Build initial pool: construct neighbor subgraph, compute components,
        keep qualified count q0 and keep unqualified top-2b as initial pool.
        """
        neigh, subAdj, sub_to_global = build_neighbor_subgraph(self.adj, self.query_u)
        comps_idx = connected_components(subAdj)
        comps_all = comps_to_dicts(comps_idx, sub_to_global, subAdj)
        # q0: initial number of qualified components
        self.q0 = diversity_count(comps_all, self.tau)
        # pool: top-2b largest unqualified components
        top_pool = top_k_unqualified(comps_all, self.tau, 2*self.budget0)
        # copy pool as current components
        # we will mutate self.comps during episode
        self.initial_pool = [dict(c) for c in top_pool]
        self.ego_nodes = neigh
        return self.reset()

    def reset(self):
        self.comps = [dict(c) for c in self.initial_pool]
        self.budget = self.budget0
        self.steps = 0
        # qualified count from original graph is q0; we will compute increases relative to q0
        self.done = False
        return self._get_state()

    def _get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Return:
        - features: (m, 6) numpy array
            [size, indicator, gap, ego_size_norm, internal_density, avg_deg_norm]
        - stage1_mask: (m,) boolean numpy (True = selectable)
        - pair_mask: (m,m) boolean where (i,j)=True if valid pair (both unqualified)
        - remaining budget (int)
        """
        m = len(self.comps)
        if m == 0:
            feats = np.zeros((0, 6), dtype=np.float32)
            return feats, np.zeros((0,), dtype=np.bool_), np.zeros((0, 0), dtype=np.bool_), self.budget

        feats = []
        for c in self.comps:
            size = float(c['size'])
            indicator = 1.0 if c['size'] >= self.tau else 0.0
            gap = float(max(0, self.tau - c['size']))
            ego_size_norm = float(len(self.ego_nodes)) / max(1.0, size)  # |N(u)| / |C_j|

            # internal density ρ(C_j)
            rho = 0.0
            if c['size'] > 1:
                rho = 2.0 * float(c.get('edge_cnt', 0)) / (size * (size - 1.0) + 1e-12)

            # avg degree normalized by max degree in component
            degs = np.array(c.get('degrees', []), dtype=np.float32)
            if len(degs) > 0:
                avg_deg_norm = degs.mean() / max(1.0, degs.max())
            else:
                avg_deg_norm = 0.0

            feats.append([size, indicator, gap, ego_size_norm, rho, avg_deg_norm])

        feats = np.asarray(feats, dtype=np.float32)

        # Stage-1 mask: selectable if unqualified
        stage1_mask = np.array([c['size'] < self.tau for c in self.comps], dtype=np.bool_)

        # Pair mask: only allow merging two unqualified components (i != j)
        pair_mask = np.ones((m, m), dtype=np.bool_)
        for i in range(m):
            for j in range(m):
                if i == j or not (self.comps[i]['size'] < self.tau and self.comps[j]['size'] < self.tau):
                    pair_mask[i, j] = False

        return feats, stage1_mask, pair_mask, self.budget


    def legal_actions_exist(self) -> bool:
        return (len(self.comps) >= 2) and (self.budget > 0)

    def _max_possible_single_merge(self, remaining_budget: int) -> int:
        """
        Compute the maximum possible size of any single component that could be
        formed by merging up to (remaining_budget + 1) largest current components.
        This provides a conservative test whether any new qualified component
        can possibly be created given the remaining budget.
        """
        sizes = sorted([c['size'] for c in self.comps], reverse=True)
        k = min(len(sizes), remaining_budget + 1)
        if k == 0:
            return 0
        return sum(sizes[:k])

    def step(self, action: Tuple[int,int]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, int], float, bool, dict]:
        """
        action: (p, q) indices in current self.comps
        Returns: next_state, reward, done, info
        Reward:
           +1 if the merge creates one or more newly qualified components
           else negative penalty -delta * gap (paper uses delta in (0,1])
        Implementation uses delta=0.1 by default for small penalty.
        """
        if self.done:
            raise RuntimeError("step called after done")

        p, q = action
        m = len(self.comps)
        if p<0 or q<0 or p>=m or q>=m or p==q:
            # invalid action; heavy penalty and end
            self.done = True
            return self._get_state(), -1.0, True, {"invalid": True}

        size_p = self.comps[p]['size']
        size_q = self.comps[q]['size']
        new_size = size_p + size_q

        # Determine deltaSD: how many new qualified components result from this merge
        delta_sd = 1 if (size_p < self.tau and size_q < self.tau and new_size >= self.tau) else 0

        if delta_sd >= 1:
            reward = 1.0
        else:
            g_t = max(0, self.tau - size_p - size_q)
            delta = 0.1
            reward = - delta ** float(g_t)

        # Merge: remove both indices (higher first) and append union comp
        rep_nodes = []
        for idx in sorted([p,q], reverse=True):
            rep_nodes.extend(self.comps[idx].get('nodes', []))
            self.comps.pop(idx)
        new_comp = {
            'size': int(new_size),
            'nodes': rep_nodes,
            'edge_cnt': 0,
            'avg_deg': 0.0,
            'max_deg': 0.0
        }
        self.comps.append(new_comp)

        self.budget -= 1
        self.steps += 1

        # done conditions:
        # 1) budget exhausted
        if self.budget <= 0:
            self.done = True

        # 2) if fewer than 2 unqualified components left -> done
        unq_count = sum(1 for c in self.comps if c['size'] < self.tau)
        if unq_count < 2:
            self.done = True

        # 3) heuristic: if even merging top (budget+1) components cannot reach tau -> done
        max_possible = self._max_possible_single_merge(self.budget)
        if max_possible < self.tau:
            # no future merge can produce a qualified component
            self.done = True

        next_state = self._get_state()
        return next_state, float(reward), bool(self.done), {}

