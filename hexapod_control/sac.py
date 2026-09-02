import copy
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import SAC_CONFIG, SACConfig


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: float) -> None:
        if not (np.isfinite(obs).all() and np.isfinite(action).all() and np.isfinite(reward) and np.isfinite(next_obs).all() and np.isfinite(done)):
            raise FloatingPointError("Refusing to store a non-finite transition")
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise RuntimeError(f"Replay buffer has {self.size} samples, needs {batch_size}")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(array[indices], dtype=torch.float32, device=device) for array in (self.obs, self.action, self.reward, self.next_obs, self.done))


def build_mlp(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim < 1:
            raise ValueError("hidden dimensions must be positive")
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = hidden_dim
    return nn.Sequential(*layers), current_dim


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int], log_std_min: float = -5.0, log_std_max: float = 1.0):
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone, feature_dim = build_mlp(obs_dim, hidden_dims)
        self.mu = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mu.bias)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.constant_(self.log_std.bias, -1.0)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        return self.mu(features), torch.clamp(self.log_std(features), self.log_std_min, self.log_std_max)

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        distribution = Normal(mu, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = (distribution.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        backbone, feature_dim = build_mlp(obs_dim + action_dim, hidden_dims)
        self.net = nn.Sequential(backbone, nn.Linear(feature_dim, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.net(torch.cat((obs, action), dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)


class SACAgent:
    CHECKPOINT_VERSION = 5

    def __init__(self, config: SACConfig = SAC_CONFIG):
        self.cfg = config
        device_name = "cuda" if config.device == "auto" and torch.cuda.is_available() else "cpu" if config.device == "auto" else config.device
        self.device = torch.device(device_name)
        self.actor = Actor(config.obs_dim, config.action_dim, config.hidden_dims, config.log_std_min, config.log_std_max).to(self.device)
        self.critic = TwinCritic(config.obs_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_alpha), dtype=torch.float32, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = float(-config.action_dim if config.target_entropy is None else config.target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action = self.actor.sample(obs_tensor)
        return (mean_action if deterministic else action).cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        obs, action, reward, next_obs, done = replay_buffer.sample(self.cfg.batch_size, self.device)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            bellman_target = self.cfg.reward_scale * reward + self.cfg.gamma * (1.0 - done) * target_q

        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, bellman_target) + F.mse_loss(current_q2, bellman_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.gradient_clip)
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        new_action, log_prob, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10.0, 2.0)
            for parameter, target_parameter in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_parameter.mul_(1.0 - self.cfg.tau).add_(parameter, alpha=self.cfg.tau)

        metrics = {
            "critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()), "alpha": float(self.alpha.item()),
            "q": float(torch.min(q1_pi, q2_pi).mean().item()), "entropy": float(-log_prob.mean().item()),
            "critic_grad_norm": float(critic_grad_norm.item()), "actor_grad_norm": float(actor_grad_norm.item()),
        }
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError(f"Non-finite SAC metrics: {metrics}")
        return metrics

    def save(self, path: str, step: int = 0, extra: Optional[Dict[str, Any]] = None) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": self.CHECKPOINT_VERSION, "step": int(step), "config": asdict(self.cfg),
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(), "log_alpha": self.log_alpha.detach().cpu(), "extra": extra or {},
        }, checkpoint_path)

    def load(self, path: str, load_optimizers: bool = True) -> Dict[str, Any]:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        saved_config = checkpoint.get("config", {})
        saved_obs_dim = int(saved_config.get("obs_dim", self.cfg.obs_dim))
        saved_action_dim = int(saved_config.get("action_dim", self.cfg.action_dim))
        if saved_obs_dim != self.cfg.obs_dim or saved_action_dim != self.cfg.action_dim:
            raise ValueError(f"Checkpoint dimensions ({saved_obs_dim}, {saved_action_dim}) do not match current config ({self.cfg.obs_dim}, {self.cfg.action_dim})")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        self.log_alpha.data.copy_(torch.as_tensor(checkpoint.get("log_alpha", math.log(self.cfg.initial_alpha)), dtype=torch.float32, device=self.device))
        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.critic_target.requires_grad_(False)
        return {"step": int(checkpoint.get("step", 0)), "extra": checkpoint.get("extra", {}), "version": int(checkpoint.get("version", 1))}
