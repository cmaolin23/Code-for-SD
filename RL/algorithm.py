# algorithm.py
import math
import torch
import torch.nn.functional as F
from typing import Tuple, List
from model_a2c import ActorCritic
from environment import GraphEnv

EPS = 1e-9

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim=-1):
    """
    logits: (..., n)
    mask: same shape boolean where True = valid
    returns normalized probabilities with zeros for masked entries
    """
    neg_inf = -1e9
    masked = logits.masked_fill(~mask.bool(), neg_inf)
    probs = F.softmax(masked, dim=dim)
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    denom = probs.sum(dim=dim, keepdim=True)
    denom = denom + EPS
    return probs / denom

class A2CTrainer:
    def __init__(self, net: ActorCritic, lr: float = 1e-4, gamma: float = 0.99,
                 ent_coef: float = 0.01, value_coef: float = 0.5, n_step: int = 5,
                 device: str = None):
        self.net = net
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.n_step = n_step
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def collect_episode(self, env: GraphEnv, max_steps: int):
        """
        Collect a single episode (or up to max_steps).
        Returns trajectory list of step dicts and final observation.
        Each step dict: feats (tensor), h (tensor), p, q, logprob (tensor), reward (float), done (bool)
        """
        obs = env.reset()
        feats_np, m1_mask_np, pair_mask_np, budget = obs
        feats = torch.from_numpy(feats_np).float().to(self.device)
        m1_mask = torch.from_numpy(m1_mask_np).to(self.device)
        pair_mask = torch.from_numpy(pair_mask_np).to(self.device)

        traj = []
        step = 0
        done = False
        while (not done) and step < max_steps:
            m = feats.shape[0]
            if m < 2 or env.budget <= 0:
                break

            logits1, h = self.net.stage1_logits(feats)  # (m,), (m,emb)
            probs1 = masked_softmax(logits1, m1_mask)
            dist1 = torch.distributions.Categorical(probs1)
            p = int(dist1.sample().item())

            logits2 = self.net.stage2_logits(feats, p)  # (m,)
            mask2 = pair_mask[p].to(self.device)
            probs2 = masked_softmax(logits2, mask2)
            dist2 = torch.distributions.Categorical(probs2)
            q = int(dist2.sample().item())

            logprob = dist1.log_prob(torch.tensor(p, device=self.device)) + dist2.log_prob(torch.tensor(q, device=self.device))
            value = self.net.value(h).detach()

            (next_feats_np, next_m1_mask_np, next_pair_mask_np, next_budget), reward, done, info = env.step((p,q))
            next_feats = torch.from_numpy(next_feats_np).float().to(self.device)
            next_m1_mask = torch.from_numpy(next_m1_mask_np).to(self.device)
            next_pair_mask = torch.from_numpy(next_pair_mask_np).to(self.device)

            traj.append({
                'feats': feats, 'h': h, 'p': p, 'q': q,
                'logprob': logprob, 'value': value, 'reward': float(reward), 'done': done,
                'm1_mask': m1_mask, 'pair_mask': pair_mask
            })

            feats = next_feats
            m1_mask = next_m1_mask
            pair_mask = next_pair_mask

            step += 1

        # return trajectory and last obs
        last_obs = (feats, m1_mask, pair_mask, env.budget)
        return traj, last_obs

    def compute_returns_and_advs(self, traj, last_obs):
        """
        Compute n-step returns and advantages for collected trajectory (on-policy).
        Returns lists Gs and advs (torch tensors on device).
        """
        if len(traj) == 0:
            return [], []

        # bootstrap value
        last_feats, _, _, _ = last_obs
        if last_feats.shape[0] == 0:
            last_value = torch.tensor(0.0, device=self.device)
        else:
            _, last_h = self.net.stage1_logits(last_feats)
            last_value = self.net.value(last_h).detach()

        returns = []
        R = last_value
        # compute returns in reverse
        for step in reversed(range(len(traj))):
            r = torch.tensor(traj[step]['reward'], device=self.device, dtype=torch.float32)
            done = traj[step]['done']
            if done:
                R = torch.tensor(0.0, device=self.device)
            R = r + self.gamma * R
            returns.insert(0, R)

        advs = []
        for idx, step in enumerate(traj):
            value = step['value']
            A = returns[idx] - value
            advs.append(A)
        return returns, advs

    def update(self, traj, returns, advs):
        """
        Update net parameters using collected trajectory and computed advantages.
        """
        if len(traj) == 0:
            return 0.0, 0.0

        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0

        for i, step in enumerate(traj):
            feats = step['feats']
            feats = feats.to(self.device)
            m1_mask = step['m1_mask'].to(self.device)
            pair_mask = step['pair_mask'].to(self.device)

            logits1, h = self.net.stage1_logits(feats)
            probs1 = masked_softmax(logits1, m1_mask)
            dist1 = torch.distributions.Categorical(probs1)
            p = step['p']
            logp1 = dist1.log_prob(torch.tensor(p, device=self.device))

            logits2 = self.net.stage2_logits(feats, p)
            mask2 = pair_mask[p].to(self.device)
            probs2 = masked_softmax(logits2, mask2)
            dist2 = torch.distributions.Categorical(probs2)
            q = step['q']
            logp2 = dist2.log_prob(torch.tensor(q, device=self.device))

            logprob = logp1 + logp2
            ent = dist1.entropy() + dist2.entropy()

            advantage = advs[i].detach()
            policy_loss = policy_loss - logprob * advantage
            value = self.net.value(h)
            value_loss = value_loss + F.mse_loss(value, returns[i])
            entropy_loss = entropy_loss - ent

        loss = policy_loss + self.value_coef * value_loss + self.ent_coef * entropy_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        return policy_loss.item(), value_loss.item()

    def train_epoch(self, envs: List[GraphEnv], max_steps_per_episode: int):
        """
        Run one epoch: collect one episode per env (sequentially) and update.
        envs: list of GraphEnv instances (can be single-element)
        Returns average losses
        """
        ploss_total = 0.0
        vloss_total = 0.0
        n = 0
        for env in envs:
            traj, last_obs = self.collect_episode(env, max_steps_per_episode)
            returns, advs = self.compute_returns_and_advs(traj, last_obs)
            ploss, vloss = self.update(traj, returns, advs)
            ploss_total += ploss; vloss_total += vloss
            n += 1
        return ploss_total / max(1,n), vloss_total / max(1,n)

    def infer_greedy(self, env: GraphEnv, device: str):
        """
        Deterministic inference: choose argmax at each stage until stop.
        Returns (q0, q_new, inc, added_edges)
        """
        env.reset()
        added_edges = []
        q0 = env.q0
        qinc = 0
        rem = env.budget
        # dynamic components list is env.comps
        while rem > 0 and len(env.comps) >= 2:
            feats_np, _, _, _ = env._get_state()
            feats = torch.from_numpy(feats_np).float().to(self.device)
            logits1, _ = self.net.stage1_logits(feats)
            mask1 = torch.tensor([c['size']<env.tau for c in env.comps], dtype=torch.bool, device=self.device)
            probs1 = masked_softmax(logits1, mask1)
            p = int(torch.argmax(probs1).item())

            logits2 = self.net.stage2_logits(feats, p)
            mask2 = torch.tensor([ (i!=p and env.comps[i]['size']<env.tau) for i in range(len(env.comps)) ], dtype=torch.bool, device=self.device)
            probs2 = masked_softmax(logits2, mask2)
            q = int(torch.argmax(probs2).item())

            size_p = env.comps[p]['size']; size_q = env.comps[q]['size']
            new_size = size_p + size_q
            if new_size >= env.tau:
                qinc += 1
            # record rep nodes if exist
            rp = env.comps[p]['nodes'][0] if env.comps[p]['nodes'] else None
            rq = env.comps[q]['nodes'][0] if env.comps[q]['nodes'] else None
            if rp is not None and rq is not None:
                added_edges.append((rp, rq))

            # apply merge
            env.step((p,q))
            rem = env.budget
        q_new = q0 + qinc
        return q0, q_new, qinc, added_edges
