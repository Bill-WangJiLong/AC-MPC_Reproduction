"""Train AC-MPC with PPO plus Model-Predictive Value Expansion.

This script implements the Phase 5 MPVE extension without changing the
baseline training entrypoint. Importing this file does not start training.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, NamedTuple, Optional

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF_MPC_DRONES = REPO_ROOT / "diff_mpc_drones"
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (REPO_ROOT, DIFF_MPC_DRONES, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from train_acmpc_gym import (
    CudaMemoryCsvCallback,
    EpisodeCsvCallback,
    build_arg_parser as build_base_arg_parser,
    linear_schedule,
    make_env,
    positive_int,
    save_run_config,
)


class MPVERolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    states: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    prediction_observations: th.Tensor
    prediction_rewards: th.Tensor
    prediction_valid: th.Tensor
    prediction_terminal: th.Tensor


class MPVERolloutBuffer(RolloutBuffer):
    """Rollout buffer extended with MPC prediction data for MPVE."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        state_space: spaces.Space,
        prediction_horizon: int,
        device: str | th.device = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.prediction_horizon = int(prediction_horizon)
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            state_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super().reset()
        pred_shape = (self.buffer_size, self.n_envs, self.prediction_horizon) + self.obs_shape
        self.prediction_observations = np.zeros(pred_shape, dtype=np.float32)
        self.prediction_rewards = np.zeros(
            (self.buffer_size, self.n_envs, self.prediction_horizon),
            dtype=np.float32,
        )
        self.prediction_valid = np.zeros(
            (self.buffer_size, self.n_envs, self.prediction_horizon),
            dtype=np.float32,
        )
        self.prediction_terminal = np.zeros(
            (self.buffer_size, self.n_envs, self.prediction_horizon),
            dtype=np.float32,
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        state: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        prediction_observations: np.ndarray,
        prediction_rewards: np.ndarray,
        prediction_valid: np.ndarray,
        prediction_terminal: np.ndarray,
    ) -> None:
        pos = self.pos
        super().add(obs, action, state, reward, episode_start, value, log_prob)
        self.prediction_observations[pos] = np.asarray(prediction_observations, dtype=np.float32).reshape(
            (self.n_envs, self.prediction_horizon) + self.obs_shape
        )
        self.prediction_rewards[pos] = np.asarray(prediction_rewards, dtype=np.float32).reshape(
            self.n_envs,
            self.prediction_horizon,
        )
        self.prediction_valid[pos] = np.asarray(prediction_valid, dtype=np.float32).reshape(
            self.n_envs,
            self.prediction_horizon,
        )
        self.prediction_terminal[pos] = np.asarray(prediction_terminal, dtype=np.float32).reshape(
            self.n_envs,
            self.prediction_horizon,
        )

    def get(self, batch_size: Optional[int] = None) -> Generator[MPVERolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            tensor_names = [
                "observations",
                "actions",
                "states",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "prediction_observations",
                "prediction_rewards",
                "prediction_valid",
                "prediction_terminal",
            ]
            for tensor in tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Any = None) -> MPVERolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.states[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.prediction_observations[batch_inds],
            self.prediction_rewards[batch_inds],
            self.prediction_valid[batch_inds],
            self.prediction_terminal[batch_inds],
        )
        return MPVERolloutBufferSamples(*tuple(map(self.to_torch, data)))


def compute_mpve_targets(
    prediction_rewards: th.Tensor,
    bootstrap_values: th.Tensor,
    prediction_valid: th.Tensor,
    prediction_terminal: th.Tensor,
    gamma: float,
    bootstrap: bool = True,
) -> th.Tensor:
    """Compute H-step MPVE targets with a TD-k style valid/terminal mask.

    All inputs are batch-first tensors with shape [batch, H]. The returned
    targets are detached by construction when this function is called under
    ``torch.no_grad()``.
    """
    rewards = prediction_rewards.float()
    valid = prediction_valid.float()
    terminal = prediction_terminal.float()
    bootstrap_values = bootstrap_values.float()

    batch_size, horizon = rewards.shape
    targets = th.zeros_like(rewards)

    for start in range(horizon):
        active = valid[:, start]
        returns = th.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
        discount = 1.0

        for step in range(start, horizon):
            step_active = active * valid[:, step]
            returns = returns + discount * step_active * rewards[:, step]
            active = step_active * (1.0 - terminal[:, step])
            discount *= gamma

        if bootstrap:
            returns = returns + discount * active * bootstrap_values[:, -1]
        targets[:, start] = returns

    return targets


class MPVEPPO(PPO):
    """PPO with an additional MPVE critic loss term."""

    def __init__(
        self,
        *args,
        mpve_coef: float = 1.0,
        mpve_horizon: Optional[int] = None,
        mpve_bootstrap: bool = True,
        mpve_valid_mask: bool = True,
        **kwargs,
    ):
        self.mpve_coef = float(mpve_coef)
        self.mpve_horizon = int(mpve_horizon or os.environ.get("ACMPC_T", 2))
        self.mpve_bootstrap = bool(mpve_bootstrap)
        self.mpve_valid_mask = bool(mpve_valid_mask)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        state_space = getattr(
            self.env,
            "state_space",
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        )
        self.rollout_buffer = MPVERolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            state_space,
            prediction_horizon=self.mpve_horizon,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer: MPVERolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                state_np = self._get_env_state(env)
                state_tensor = obs_as_tensor(state_np, self.device)
                actions, values, log_probs = self.policy(obs_tensor, state_tensor)
                mpc_predictions = self._policy_predictions_to_numpy(env.num_envs)

            prediction_data = self._evaluate_predictions(env, mpc_predictions)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                state_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                prediction_data["observations"],
                prediction_data["rewards"],
                prediction_data["valid"],
                prediction_data["terminal"],
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, td_value_losses, mpve_value_losses = [], [], [], []
        clip_fractions = []
        mpve_valid_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.states,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                td_value_loss = F.mse_loss(rollout_data.returns, values_pred)
                mpve_value_loss, mpve_valid_fraction = self._compute_mpve_value_loss(rollout_data)
                value_loss = td_value_loss + self.mpve_coef * mpve_value_loss

                td_value_losses.append(td_value_loss.item())
                mpve_value_losses.append(mpve_value_loss.item())
                mpve_valid_fractions.append(mpve_valid_fraction)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/td_value_loss", np.mean(td_value_losses))
        self.logger.record("train/mpve_value_loss", np.mean(mpve_value_losses))
        self.logger.record("train/mpve_valid_fraction", np.mean(mpve_valid_fractions))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _policy_predictions_to_numpy(self, n_envs: int) -> np.ndarray:
        predictions = getattr(self.policy.mlp_extractor, "predictions", None)
        if predictions is None:
            return np.zeros((n_envs, self.mpve_horizon, 14), dtype=np.float32)

        predictions_np = predictions.detach().cpu().numpy().astype(np.float32)
        if predictions_np.ndim != 3 or predictions_np.shape[0] != n_envs or predictions_np.shape[2] != 14:
            raise RuntimeError(
                "Unexpected MlpMpcPolicy prediction shape: "
                f"got {predictions_np.shape}, expected ({n_envs}, H, 14)"
            )
        if predictions_np.shape[1] < self.mpve_horizon:
            raise RuntimeError(
                f"MPVE horizon {self.mpve_horizon} exceeds prediction horizon {predictions_np.shape[1]}"
            )
        return predictions_np[:, : self.mpve_horizon, :]

    def _evaluate_predictions(self, env: VecEnv, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        n_envs = predictions.shape[0]
        obs_shape = self.observation_space.shape
        prediction_observations = np.zeros((n_envs, self.mpve_horizon) + obs_shape, dtype=np.float32)
        prediction_rewards = np.zeros((n_envs, self.mpve_horizon), dtype=np.float32)
        prediction_valid = np.zeros((n_envs, self.mpve_horizon), dtype=np.float32)
        prediction_terminal = np.zeros((n_envs, self.mpve_horizon), dtype=np.float32)

        if self.mpve_coef == 0.0:
            return {
                "observations": prediction_observations,
                "rewards": prediction_rewards,
                "valid": prediction_valid,
                "terminal": prediction_terminal,
            }

        for env_idx in range(n_envs):
            result = env.env_method(
                "compute_prediction_rollout",
                predictions[env_idx],
                indices=[env_idx],
            )[0]
            prediction_observations[env_idx] = np.asarray(result["observations"], dtype=np.float32)
            prediction_rewards[env_idx] = np.asarray(result["rewards"], dtype=np.float32)
            prediction_valid[env_idx] = np.asarray(result["valid"], dtype=np.float32)
            prediction_terminal[env_idx] = np.asarray(result["terminal"], dtype=np.float32)

        if hasattr(env, "normalize_obs"):
            flat_obs = prediction_observations.reshape((n_envs * self.mpve_horizon,) + obs_shape)
            flat_obs = env.normalize_obs(flat_obs)
            prediction_observations = flat_obs.reshape((n_envs, self.mpve_horizon) + obs_shape)

        return {
            "observations": prediction_observations,
            "rewards": prediction_rewards,
            "valid": prediction_valid,
            "terminal": prediction_terminal,
        }

    def _compute_mpve_value_loss(self, rollout_data: MPVERolloutBufferSamples) -> tuple[th.Tensor, float]:
        if self.mpve_coef == 0.0:
            return th.zeros((), device=self.device), 0.0

        prediction_obs = rollout_data.prediction_observations
        batch_size, horizon = prediction_obs.shape[:2]
        flat_obs = prediction_obs.reshape((batch_size * horizon,) + self.observation_space.shape)
        predicted_values = self.policy.predict_values(flat_obs).flatten().reshape(batch_size, horizon)

        if self.mpve_valid_mask:
            valid = rollout_data.prediction_valid.float()
        else:
            valid = th.ones_like(rollout_data.prediction_valid, dtype=th.float32)

        with th.no_grad():
            targets = compute_mpve_targets(
                prediction_rewards=rollout_data.prediction_rewards,
                bootstrap_values=predicted_values.detach(),
                prediction_valid=valid,
                prediction_terminal=rollout_data.prediction_terminal,
                gamma=self.gamma,
                bootstrap=self.mpve_bootstrap,
            )

        denom = valid.sum().clamp_min(1.0)
        loss = (((predicted_values - targets) ** 2) * valid).sum() / denom
        valid_fraction = float(valid.mean().detach().cpu().item())
        return loss, valid_fraction


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser()
    parser.description = "Train AC-MPC PPO with MPVE critic expansion on the Python Gym racing environment."
    parser.set_defaults(output_dir=REPO_ROOT / "runs" / "acmpc_gym_mpve")
    parser.add_argument("--mpve-coef", type=float, default=1.0, help="Weight for the MPVE critic loss.")
    parser.add_argument(
        "--mpve-horizon",
        type=positive_int,
        default=None,
        help="MPVE horizon. Defaults to ACMPC_T.",
    )
    parser.add_argument(
        "--no-mpve-bootstrap",
        action="store_true",
        help="Disable final predicted-state value bootstrap in MPVE targets.",
    )
    parser.add_argument(
        "--no-mpve-valid-mask",
        action="store_true",
        help="Use all predicted steps in MPVE loss, including invalid post-terminal steps.",
    )
    return parser


def make_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    horizon = args.mpve_horizon or args.acmpc_t
    run_name = args.run_name or f"{timestamp}_{args.track_name}_T{args.acmpc_t}_mpveH{horizon}"
    if args.single_gate:
        run_name += "_single_gate"
    return run_name


def build_ppo_config(args: argparse.Namespace, batch_size: int) -> Dict[str, Any]:
    return {
        "learning_rate_start": args.learning_rate_start,
        "learning_rate_end": args.learning_rate_end,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "log_std_init": args.log_std_init,
        "mpve_coef": args.mpve_coef,
        "mpve_horizon": args.mpve_horizon or args.acmpc_t,
        "mpve_bootstrap": not args.no_mpve_bootstrap,
        "mpve_valid_mask": not args.no_mpve_valid_mask,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    os.environ["ACMPC_T"] = str(args.acmpc_t)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    run_dir = args.output_dir / make_run_name(args)
    checkpoint_dir = run_dir / "checkpoints"
    sb3_log_dir = run_dir / "sb3"
    csv_dir = run_dir / "csv"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    env, env_config = make_env(args, run_dir)
    rollout_size = args.n_envs * args.n_steps
    batch_size = args.batch_size or min(25_000, rollout_size)
    if rollout_size % batch_size != 0:
        print(
            f"Warning: rollout_size={rollout_size} is not divisible by batch_size={batch_size}; "
            "SB3 will use a truncated final minibatch.",
            file=sys.stderr,
        )

    learning_rate = linear_schedule(args.learning_rate_start, args.learning_rate_end)
    ppo_config = build_ppo_config(args, batch_size)
    save_run_config(run_dir, args, env_config, ppo_config)

    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    common_model_kwargs = dict(
        mpve_coef=args.mpve_coef,
        mpve_horizon=args.mpve_horizon or args.acmpc_t,
        mpve_bootstrap=not args.no_mpve_bootstrap,
        mpve_valid_mask=not args.no_mpve_valid_mask,
        device=args.device,
    )

    if args.resume_model is not None:
        model = MPVEPPO.load(str(args.resume_model), env=env, **common_model_kwargs)
    else:
        model = MPVEPPO(
            MlpMpcPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=str(sb3_log_dir),
            policy_kwargs={"log_std_init": args.log_std_init},
            verbose=args.verbose,
            seed=args.seed,
            **common_model_kwargs,
        )

    model.set_logger(configure(str(sb3_log_dir), ["stdout", "csv", "tensorboard"]))

    checkpoint_save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    callback_items = [
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=str(checkpoint_dir),
            name_prefix="acmpc_gym_mpve",
            save_vecnormalize=True,
            verbose=args.verbose,
        ),
        EpisodeCsvCallback(csv_dir / "episodes.csv", verbose=args.verbose).as_sb3_callback(),
    ]
    if args.cuda_memory_log:
        callback_items.append(
            CudaMemoryCsvCallback(
                csv_dir / "cuda_memory.csv",
                log_freq_rollouts=args.cuda_memory_log_freq_rollouts,
                verbose=args.verbose,
            ).as_sb3_callback()
        )
    callbacks = CallbackList(callback_items)

    print(f"Run directory: {run_dir}")
    print(f"Track: {env_config.track_name}")
    print(f"ACMPC_T: {os.environ['ACMPC_T']}")
    print(
        "MPVE: "
        f"coef={args.mpve_coef}, horizon={args.mpve_horizon or args.acmpc_t}, "
        f"bootstrap={not args.no_mpve_bootstrap}, valid_mask={not args.no_mpve_valid_mask}"
    )
    print(f"n_envs={args.n_envs}, n_steps={args.n_steps}, rollout_size={rollout_size}, batch_size={batch_size}")
    print("Starting PPO+MPVE training...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=args.log_interval)
    finally:
        model.save(str(run_dir / "final_model.zip"))
        vec_norm = model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(str(run_dir / "vecnormalize.pkl"))
        env.close()
        print(f"Saved final model to: {run_dir / 'final_model.zip'}")
        if (run_dir / "vecnormalize.pkl").exists():
            print(f"Saved VecNormalize stats to: {run_dir / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
