from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Self, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar, jaxtyped
from utils.flax_utils import ActorTrainState, RLTrainState, TrainState
import optax
from functools import partial


Array = jax.Array

@dataclass
class ChunkyFQLConfig:
    actor_chunksize: int
    single_action_v: bool
    n_critics: int
    distributional: bool
    update_ensemble_size: int
    only_fit_last_nstep: bool
    actor_kind: str
    apply_next_state_ent_bonus: bool
    simbav2: bool
    use_bnstats_from_live_net: bool

    def update_chunky_critic(
        self,
        gamma: Union[float, Scalar],
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        chunky_observations: Float[Array, "batch chunkyobs"],
        chunky_actions: Float[Array, "batch chunkyaction"],
        chunky_rewards: Float[Array, "batch chunk"],
        chunky_dones: Float[Array, "batch chunk"],
        chunky_truncated: Float[Array, "batch chunk"],
        chunky_next_observations: Float[Array, "batch chunkyobs"],
        key,
        sampler,
    ):
        batch_size, chunk_size = chunky_rewards.shape
        critic_chunksize = chunk_size
        assert chunk_size == critic_chunksize
        action_size = chunky_actions.shape[1] // chunk_size
        obs_size = chunky_next_observations.shape[1] // chunk_size

        key, next_action_key, current_dropout_key, next_dropout_key = jax.random.split(
            key, 4
        )

        first_observation = jnp.split(chunky_observations, chunk_size, axis=1)[0]
        chunky_actions = chunky_actions.reshape((batch_size, chunk_size, action_size))
        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        next_state_v_mask = jnp.logical_not(chunky_dones)
        next_state_v_scale = next_state_v_mask * gamma ** jnp.arange(
            1, critic_chunksize + 1
        )

        discounted_reward = gamma ** jnp.arange(chunk_size) * chunky_rewards

        discounted_accumulated_reward = jnp.cumsum(discounted_reward, axis=1)

        nonfinite_next_state_logprobs = 0
        chunky_next_observations = chunky_next_observations.reshape(
            (batch_size, critic_chunksize, obs_size)
        )
        if self.only_fit_last_nstep:
            chunky_next_observations = chunky_next_observations[:, -1, :].reshape(
                (batch_size, 1, obs_size)
            )
            next_state_v_scale = next_state_v_scale[:, -1].reshape((batch_size, 1))
            discounted_accumulated_reward = discounted_accumulated_reward[
                :, -1
            ].reshape((batch_size, 1))

        q_target = self.calculate_q_target(
            gamma,
            chunky_next_observations,
            next_state_v_scale,
            discounted_accumulated_reward,
            actor_state,
            qf_state,
            ent_coef_value if self.apply_next_state_ent_bonus else 0.0,
            next_action_key,
            self.use_bnstats_from_live_net,
            sampler,
            action_size,
        )
        assert isinstance(q_target, Array)

        if self.distributional: # TODO
            pass
        else:
            def mse_loss(params, batch_stats, dropout_key):
                nonlocal critic_chunksize
                nonlocal q_target
                q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    first_observation,
                    chunky_actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                if self.only_fit_last_nstep:
                    q_values = q_values[:, :, -1].reshape(
                        (self.n_critics, batch_size, 1)
                    )
                    critic_chunksize = 1  # <-- keep in mind for the following asserts
                else:
                    q_values = q_values[:, :, 1:]  # Drop V

                assert q_values.shape == (
                    self.n_critics,
                    batch_size,
                    critic_chunksize,
                )

                diff = q_target - q_values
                assert isinstance(q_target, Array)
                assert q_target.shape == (batch_size, critic_chunksize)
                assert diff.shape == (
                    self.n_critics,
                    batch_size,
                    critic_chunksize,
                )

                loss = (
                    (diff**2)
                    .mean(axis=2)  # over chunk
                    .mean(axis=1)  # over batch
                    .sum()  # over n_critics
                )
                assert loss.size == 1
                return loss, (state_updates, q_values[:, 1:], q_target)

            loss = mse_loss

        (
            qf_loss_value,
            (state_updates, current_q_values, next_q_values),
        ), grads = jax.value_and_grad(loss, has_aux=True)(
            qf_state.params, qf_state.batch_stats, current_dropout_key
        )

        qf_state = qf_state.apply_gradients(grads=nan_to_num_dict(grads))
        if self.simbav2:
            qf_state = qf_state.l2normalize()
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            "critic_grad_norm": optax.global_norm(grads),
            "critic_param_norm": optax.global_norm(qf_state.params),
            "critic_loss": qf_loss_value,
            "ent_coef": ent_coef_value,
            "current_q_values": current_q_values.mean(),
            "next_q_values": next_q_values.mean(),
            "nonfinite_next_state_logprobs": nonfinite_next_state_logprobs,
        }

        return (qf_state, metrics, key)

    #@staticmethod
    @partial(
        jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None, None, None, None, None)
        )  # batch
    @partial(
        jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None, None, None, None, None)
        )  # chunk
    def calculate_q_target(
        self,
        gamma: Union[float, Scalar],
        next_observation: Float[Array, " obs"],
        next_state_v_scale: Union[float, Scalar],
        reward: Union[float, Scalar],
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_value: Union[float, Scalar],
        key,
        use_bnstats_from_live_net: bool,
        sampler,
        action_dim: int,
    ):

        assert len(next_observation.shape) == 1
        next_observation = jnp.expand_dims(
            next_observation, 0
        )  # flax expects a batch dim

        # calculate next-state action and its (approximate) logprob

        next_action_key, key = jax.random.split(key, 2)
        if self.actor_kind == "DIFFUSION": # TODO (flow based?)
            pass
        else:
            # adapting to qc_fql’s flow actor 
            (
                next_chunky_action,
                next_chunky_logprobs,
                _,
                _,
            ), _ = actor_state.apply_fn(  # TODO: Update state
                {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                # unroll the observations for the actor
                next_observation,
                mutable=["batch_stats"],
                train=False,
                deterministic=False,
                key=next_action_key,
                single_action=self.single_action_v,
            )
            # discount = gamma ** jnp.arange(0, self.actor_chunksize)
            # next_action_logprob = jnp.sum(next_chunky_logprobs * discount)
            next_action_logprob = jnp.array(0.0)
        
        # calculate the q-value of the next-state action

        next_action = next_chunky_action.reshape(1, self.actor_chunksize, action_dim)
        if self.single_action_v:
            next_action = next_action[0, 0, :].reshape(1, 1, action_dim)
            self.actor_chunksize = 1

        dropout_key, key = jax.random.split(key, 2)
        q_value = qf_state.apply_fn(
            {
                "params": qf_state.target_params,
                "batch_stats": (
                    qf_state.target_batch_stats
                    if not use_bnstats_from_live_net
                    else qf_state.batch_stats
                ),
            },
            next_observation,
            next_action,
            rngs={"dropout": dropout_key},
            train=False,
        )
        q_value = jnp.squeeze(q_value, 1)  # remove batch axis

        if self.distributional: # TODO
            pass
        else:
            assert q_value.shape == (self.n_critics, self.actor_chunksize + 1)

        # TODO (path not adapted)
        # choose a RLPD-style random subset of the critics
        if self.n_critics != self.update_ensemble_size:
            subset_key, key = jax.random.split(key, 2)
            critic_i = jax.random.choice(
                key=subset_key,
                a=jnp.arange(self.n_critics),
                shape=(self.update_ensemble_size,),
                replace=False,
            )
            q_value = q_value[critic_i]
        
        if self.distributional: # TODO
            pass

        else:  # non-distributional critic
            assert q_value.shape == (
                self.update_ensemble_size,
                self.actor_chunksize + 1,
            )

            q_value = q_value.min(axis=0)
            assert q_value.shape == (1 + self.actor_chunksize,)
            # we estimate the next state value using the return of the whole chunk
            q_value = q_value[-1]

            next_state_v = q_value - ent_coef_value * next_action_logprob

            return reward + next_state_v_scale * next_state_v

def nan_to_num_dict(d: dict) -> dict:
    res = dict()
    for k, v in d.items():
        if type(v) is dict:
            res[k] = nan_to_num_dict(v)
        else:
            res[k] = jnp.nan_to_num(v)
    return res