import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_grad_list, compute_grad_norm, armijo_search

SelfSPMA = TypeVar("SelfSPMA", bound="SPMA")


class SPMA(OnPolicyAlgorithm):
    """
    Softmax Policy Mirror Ascent algorithm (SPMA)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        timesteps: int,
        learning_rate_actor: Union[float, Schedule] = 3e-4,
        learning_rate_critic: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        env_name: Optional[str] = None,
        use_armijo_actor: bool = False,
        use_armijo_critic: bool = False,
        alpha_max: float = 1e6,
        c_actor: float = 0.1,
        c_critic: float = 1e-6,
        eta: float = 1.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate_actor=learning_rate_actor,
            learning_rate_critic=learning_rate_critic,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.timesteps = timesteps
        self.env_name = env_name
        self.eta = eta
        self.use_armijo_actor = use_armijo_actor
        self.use_armijo_critic = use_armijo_critic
        self.alpha_max = alpha_max
        self.c_actor = c_actor
        self.c_critic = c_critic
        self.explained_var_old = 0
        self.softmax_rep = True

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        

    def compute_actor_loss(self, rollout_data, actions, advantages, old_log_prob, backwards=False, curr_loss=None):
        # skip recompuing the loss if you already have it and just want to do a backward pass.
        if curr_loss is None:
            _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
            log_ratio = log_prob - old_log_prob.detach()
            linear_loss = -log_ratio * advantages
            # using approx kl for bregman div from Schulman blog: http://joschu.net/blog/kl-approx.html
            # approx kl is an unbiased estimator of the true kl but results in less variance.
            # alternative approach is to use bregman_div_loss = -1/self.eta * log_ratio
            bregman_div_loss = 1/self.eta *((th.exp(log_ratio) - 1) - log_ratio)
            curr_loss = th.mean(linear_loss + bregman_div_loss)

        # backward pass
        if backwards:
            self.policy.optimizer_act.zero_grad()
            curr_loss.backward()
            
            return curr_loss, None, None
        
        return curr_loss, linear_loss, bregman_div_loss

    def compute_critic_loss(self, rollout_data, actions, backwards=False, curr_loss=None):
        if curr_loss is None and backwards:
            raise ValueError("Can not perform backward pass without curr_loss.")
            
        # skip recompuing the loss if you already have it and just want to do a backward pass.
        if curr_loss is None:
            # Re-sample the noise matrix because the log_std has changed
            values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # critic loss.
            curr_loss = F.mse_loss(rollout_data.returns, values)
        
        # backward pass.
        if backwards:
            self.policy.optimizer_critic.zero_grad()
            curr_loss.backward()

            return curr_loss, None, None

        return curr_loss, None, None

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self._update_learning_rate([('actor', self.policy.optimizer_act),
                                    ('critic', self.policy.optimizer_critic)])

        # record losses and number of times the surrogate loss is not decreasing in the inner loop.
        linear_losses = []
        bregman_div_losses = []

        alpha_max_actor = self.alpha_max
        alpha_max_critic = self.alpha_max
        alpha_actor = 1.0
        alpha_critic = 1.0
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for batch_idx, rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                old_log_prob = rollout_data.old_log_prob.clone()
    
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                

                # build a closure for the actor and critic.
                closure_actor = lambda backwards, curr_loss : self.compute_actor_loss(rollout_data, 
                                                                                      actions, 
                                                                                      advantages,
                                                                                      old_log_prob,
                                                                                      backwards=backwards, 
                                                                                      curr_loss=curr_loss)

                closure_critic = lambda backwards, curr_loss: self.compute_critic_loss(rollout_data,
                                                                                       actions,
                                                                                       backwards=backwards,
                                                                                       curr_loss=curr_loss)
                # compute the actor loss.
                loss_act, linear_loss, bregman_div_loss = closure_actor(backwards=False, curr_loss=None)

                # compute the critic loss.
                loss_critic, _, _ = closure_critic(backwards=False, curr_loss=None)

                # backward pass.
                closure_actor(backwards=True, curr_loss=loss_act)
                closure_critic(backwards=True, curr_loss=loss_critic)

                # using armijo vs adam for actor and critic.
                grad_list = get_grad_list(self.policy.params_act)
                grad_norm_actor = compute_grad_norm(grad_list)
                if self.use_armijo_actor:
                    # armijo for actor.
                    alpha_actor = armijo_search(closure_actor, self.policy.params_act, grad_list, 
                                                grad_norm_actor, alpha_max_actor, self.c_actor)
                    alpha_max_actor = alpha_actor * 1.8

                else:
                    self.policy.optimizer_act.step()
                    self.policy.optimizer_act.step()


                grad_list = get_grad_list(self.policy.params_critic)
                grad_norm_critic = compute_grad_norm(grad_list)
                if self.use_armijo_critic:
                    # armijo for critic.
                    alpha_critic = armijo_search(closure_critic, self.policy.params_critic, grad_list, 
                                                 grad_norm_critic, alpha_max_critic, self.c_critic)
                    alpha_max_critic = alpha_critic * 1.8
                else:
                    self.policy.optimizer_critic.step()
                
            self._n_updates += 1
        
        # Logging.
        linear_losses.append(linear_loss.mean().item())
        bregman_div_losses.append(bregman_div_loss.mean().item())

        # Logs
        self.logger.record("train/linear_loss", np.mean(linear_losses))
        self.logger.record("train/bregman_div_loss", np.mean(bregman_div_losses))
        self.logger.record("train/loss_actor", loss_act.item())
        self.logger.record("train/loss_critic", loss_critic.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


    def learn(
        self: SelfSPMA,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SPMA",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSPMA:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )