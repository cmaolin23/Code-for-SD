# main.py
import os
import torch
import json
from args import parse_args
from utils import read_edge_list, ensure_dir, comps_to_dicts, diversity_count
from environment import GraphEnv
from model_a2c import ActorCritic
from algorithm import A2CTrainer
from typing import List

def read_queries(path: str) -> List[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]

def eval_queries_with_model(adj, raw2id, id2raw, queries: List[int], tau: int, b:int,
                            model: ActorCritic, trainer: A2CTrainer, out_path: str):
    results = []
    total_inc = 0
    for q_raw in queries:
        if q_raw not in raw2id:
            results.append((q_raw, "ERROR_NOT_IN_GRAPH"))
            continue
        q_idx = raw2id[q_raw]
        env = GraphEnv(adj, q_idx, tau, b)
        env.reset_full()
        q0 = env.q0
        # use greedy deterministic inference with model
        q0, q_new, inc, edges = trainer.infer_greedy(env, trainer.device)
        total_inc += inc
        results.append((q_raw, q0, q_new, inc, edges))
    # write
    with open(out_path, "w") as f:
        for item in results:
            if isinstance(item[1], str):
                f.write(f"{item[0]}\t{item[1]}\n")
            else:
                q_raw, q0, q_new, inc, edges = item
                edges_str = ";".join(f"{u}-{v}" for (u,v) in edges)
                f.write(f"{q_raw}\t{q0}\t{q_new}\t{inc}\t{len(edges)}\t{edges_str}\n")
        f.write(f"#TOTAL_INC\t{total_inc}\n")
    return total_inc

def eval_queries_with_greedy_baseline(adj, raw2id, id2raw, queries, tau, b, out_path):
    """
    Use the improved-simple (ISI) greedy baseline grouping and chain representatives.
    A simple greedy is provided in-line for evaluation. This is fallback if model not provided.
    """
    from math import inf
    # reuse GraphEnv for building pools then apply simple greedy chaining: chain smallest bins first (approx)
    results = []
    total_inc = 0
    for q_raw in queries:
        if q_raw not in raw2id:
            results.append((q_raw, "ERROR_NOT_IN_GRAPH"))
            continue
        q_idx = raw2id[q_raw]
        env = GraphEnv(adj, q_idx, tau, b)
        env.reset_full()
        q0 = env.q0
        # a very simple baseline: repeatedly merge the two smallest unqualified components
        rem = env.budget
        comps = env.comps  # list of dicts
        inc = 0
        edges = []
        while rem > 0 and len(comps) >= 2:
            comps = sorted(comps, key=lambda c: c['size'])
            a = comps.pop(0); bcomp = comps.pop(0)
            new_size = a['size'] + bcomp['size']
            if new_size >= tau:
                inc += 1
            # record representative edge if nodes exist
            ra = a['nodes'][0] if a['nodes'] else None
            rb = bcomp['nodes'][0] if bcomp['nodes'] else None
            if ra is not None and rb is not None:
                edges.append((ra, rb))
            comps.append({'size': new_size, 'nodes': (a.get('nodes',[])+bcomp.get('nodes',[]))})
            rem -= 1
        q_new = q0 + inc
        results.append((q_raw, q0, q_new, inc, edges))
        total_inc += inc
    with open(out_path, "w") as f:
        for item in results:
            if isinstance(item[1], str):
                f.write(f"{item[0]}\t{item[1]}\n")
            else:
                q_raw, q0, q_new, inc, edges = item
                edges_str = ";".join(f"{u}-{v}" for (u,v) in edges)
                f.write(f"{q_raw}\t{q0}\t{q_new}\t{inc}\t{len(edges)}\t{edges_str}\n")
        f.write(f"#TOTAL_INC\t{total_inc}\n")
    return total_inc

def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    id2raw, raw2id, adj = read_edge_list(args.graph)
    train_queries = read_queries(args.train)
    test_queries = read_queries(args.test)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    trainer = None
    if args.train_epochs > 0:
        # initialize model and trainer
        model = ActorCritic(feat_dim=6, emb_dim=args.emb_dim, hidden=args.hidden)
        trainer = A2CTrainer(model, lr=args.lr, gamma=args.gamma, ent_coef=args.ent_coef,
                             value_coef=args.value_coef, n_step=args.n_step, device=device)
        # simple training loop that iterates over train queries
        for ep in range(args.train_epochs):
            envs = []
            for q_raw in train_queries:
                if q_raw not in raw2id:
                    continue
                q_idx = raw2id[q_raw]
                env = GraphEnv(adj, q_idx, args.tau, args.b)
                env.reset_full()
                envs.append(env)
            if not envs:
                break
            ploss, vloss = trainer.train_epoch(envs, max_steps_per_episode=args.b)
            if ep % 10 == 0:
                print(f"Epoch {ep}: ploss {ploss:.4f}, vloss {vloss:.4f}")
        # save model
        if args.save_path:
            torch.save(model.state_dict(), os.path.join(args.out_dir, "a2c_model.pt"))

    # Evaluation
    if model is not None and trainer is not None:
        print("Evaluating using trained model (deterministic inference)")
        trainer.net = model
        trainer.device = device
        train_total = eval_queries_with_model(adj, raw2id, id2raw, train_queries, args.tau, args.b, model, trainer, os.path.join(args.out_dir, "train_results.txt"))
        test_total  = eval_queries_with_model(adj, raw2id, id2raw, test_queries, args.tau, args.b, model, trainer, os.path.join(args.out_dir, "test_results.txt"))
    else:
        print("No trained model: evaluating baseline greedy")
        train_total = eval_queries_with_greedy_baseline(adj, raw2id, id2raw, train_queries, args.tau, args.b, os.path.join(args.out_dir, "train_results.txt"))
        test_total  = eval_queries_with_greedy_baseline(adj, raw2id, id2raw, test_queries, args.tau, args.b, os.path.join(args.out_dir, "test_results.txt"))

    # summary
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"train_total_inc\t{train_total}\n")
        f.write(f"test_total_inc\t{test_total}\n")

    print("Done. summary written to", os.path.join(args.out_dir, "summary.txt"))

if __name__ == "__main__":
    main()
