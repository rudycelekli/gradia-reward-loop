"""From-scratch PPO: clipped surrogate + GAE(lambda) + value baseline + entropy bonus.

Linear-softmax policy over one-hot features with closed-form gradients in numpy -- no
autograd, no framework. This is the "I can implement the algorithm, not just call it"
artifact. The math it implements is written out in PROGRAM.md:
  * policy gradient through log pi(a|s), grad_z log pi = e_a - pi
  * PPO-clip objective  L = min( r*A , clip(r, 1-eps, 1+eps)*A ),  r = pi_new/pi_old
  * GAE(lambda):  A_t = sum_l (gamma*lambda)^l delta_{t+l},  delta_t = r_t + gamma V_{t+1} - V_t
  * entropy regulariser, grad_z H = -pi (log pi + H)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class PPO:
    n_features: int
    n_actions: int
    lr: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 6
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    seed: int = 0
    W: np.ndarray = field(init=False, repr=False)
    V: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.W = np.zeros((self.n_features, self.n_actions))
        self.V = np.zeros(self.n_features)

    def pi(self, phi: np.ndarray) -> np.ndarray:
        return _softmax(phi @ self.W)

    def value(self, phi: np.ndarray) -> np.ndarray:
        return phi @ self.V

    def act(self, phi: np.ndarray, rng: np.random.Generator):
        p = self.pi(phi[None])[0]
        a = int(rng.choice(self.n_actions, p=p))
        return a, float(np.log(p[a] + 1e-12))

    def collect(self, env, n_steps: int, rng: np.random.Generator) -> dict:
        phis, acts, logps, rews, vals, dones = [], [], [], [], [], []
        obs = env.reset()
        for _ in range(n_steps):
            a, logp = self.act(obs, rng)
            v = float(self.value(obs[None])[0])
            nobs, r, done = env.step(a)
            phis.append(obs); acts.append(a); logps.append(logp)
            rews.append(r); vals.append(v); dones.append(float(done))
            obs = env.reset() if done else nobs
        return dict(phi=np.array(phis), a=np.array(acts), logp_old=np.array(logps),
                    rew=np.array(rews), val=np.array(vals), done=np.array(dones))

    def gae(self, b: dict, last_value: float = 0.0) -> dict:
        rew, val, done = b["rew"], b["val"], b["done"]
        T = len(rew)
        adv = np.zeros(T)
        lastgae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - done[t]
            nextval = last_value if t == T - 1 else val[t + 1]
            delta = rew[t] + self.gamma * nextval * nonterminal - val[t]
            lastgae = delta + self.gamma * self.lam * nonterminal * lastgae
            adv[t] = lastgae
        b["adv"] = adv
        b["ret"] = adv + val
        return b

    def update(self, b: dict) -> dict:
        phi, a, logp_old = b["phi"], b["a"], b["logp_old"]
        ret = b["ret"]
        adv = b["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        T = len(a)
        idx = np.arange(T)
        eA = np.zeros((T, self.n_actions)); eA[idx, a] = 1.0
        last = {}
        for _ in range(self.epochs):
            P = self.pi(phi)
            logp_new = np.log(P[idx, a] + 1e-12)
            ratio = np.exp(logp_new - logp_old)
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
            use1 = (surr1 <= surr2).astype(float)          # grad flows only through the min branch
            coef = adv * ratio * use1
            gW_pi = phi.T @ (coef[:, None] * (eA - P))      # d/dW of the clipped surrogate
            Hrow = -(P * np.log(P + 1e-12)).sum(axis=1)
            dH_dz = -P * (np.log(P + 1e-12) + Hrow[:, None])
            gW_ent = phi.T @ dH_dz
            gW = (gW_pi + self.entropy_coef * gW_ent) / T
            Vpred = self.value(phi)
            gV = phi.T @ (Vpred - ret) / T
            self.W += self.lr * gW                          # ascent on the objective
            self.V -= self.lr * self.vf_coef * gV           # descent on value MSE
            last = dict(clip_frac=float((1.0 - use1).mean()),
                        approx_kl=float(np.mean(logp_old - logp_new)),
                        entropy=float(Hrow.mean()))
        return last


def eval_return(agent: PPO, env, rng: np.random.Generator, episodes: int = 20) -> float:
    tot = 0.0
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            p = agent.pi(obs[None])[0]
            a = int(np.argmax(p))                            # greedy eval
            obs, r, done = env.step(a)
            tot += r
    return tot / episodes


def train_ppo(env, iters: int = 150, steps_per_iter: int = 512, seed: int = 0, **kw):
    """Train PPO on `env`; return (agent, history) where history[i] is the greedy return."""
    rng = np.random.default_rng(seed)
    agent = PPO(n_features=env.n_states, n_actions=env.n_actions, seed=seed, **kw)
    hist = []
    for _ in range(iters):
        b = agent.collect(env, steps_per_iter, rng)
        agent.gae(b)
        agent.update(b)
        hist.append(eval_return(agent, env, rng))
    return agent, hist
