import copy
from typing import Any

import flax
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.transformer_critic import EnsembleTransformerCritic
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, ActorTrainState, RLTrainState
from utils.networks import ActorVectorField, Value, EntropyCoef, ConstantEntropyCoef
from agents.chunky_fql import ChunkyFQLConfig

class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    qf_state: RLTrainState
    ent_coef_state: TrainState

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        if self.config["chunky_fql"]:
            raise ValueError("Not intended route. Something broken")
        
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            
            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)
            if not self.config["chunky_fql"]:
                qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
                q = jnp.mean(qs, axis=0)
                q_loss = -q.mean()
            else:
                # TODO explore alternatives
                q_loss = 0
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)
        
        def actor_apply(variables, obs, mutable=None, train=False, deterministic=False, key=None, single_action=False):
            # TODO
            act = agent.sample_actions(obs, rng=key)
            dummy_logp = jnp.zeros((agent.config["actor_chunksize"],))
            return (act, dummy_logp, None, None), {}

        if agent.config["chunky_fql"]:
            cfg = ChunkyFQLConfig(
                actor_chunksize=agent.config["actor_chunksize"],
                single_action_v=agent.config.get("single_action_v", False),
                n_critics=agent.config["num_qs"],
                distributional=False,
                update_ensemble_size=agent.config.get("update_ensemble_size", agent.config["num_qs"]),
                only_fit_last_nstep=agent.config.get("only_fit_last_nstep", False),
                actor_kind="flow",
                apply_next_state_ent_bonus=agent.config.get("apply_next_state_ent_bonus", True),
                simbav2=False,
                use_bnstats_from_live_net=False,
            )

            info = {}
            B, T, obs_dim = batch["full_observations"].shape
            _, _, act_dim = batch["actions"].shape

            chunky_observations = batch["full_observations"].reshape(B, T * obs_dim)
            chunky_next_observations = batch["next_observations"].reshape(B, T * obs_dim)
            chunky_actions = batch["actions"].reshape(B, T * act_dim)

            # reconstructing step rewards from cumulative discounted rewards
            discount = float(agent.config["discount"])
            disc_pows = discount ** jnp.arange(T)
            rewards_cum = batch["rewards"]  # (B, T)
            r0 = rewards_cum[:, :1]
            r_diff = rewards_cum[:, 1:] - rewards_cum[:, :-1]
            r_step = jnp.concatenate([r0, r_diff], axis=1) / disc_pows  # (B, T)
            term_cum = batch["terminals"]  # (B, T) cumulative max
            d0 = term_cum[:, :1]
            d_step = jnp.concatenate([d0, jnp.clip(term_cum[:, 1:] - term_cum[:, :-1], 0.0, 1.0)], axis=1)
            chunky_rewards = r_step
            chunky_dones = d_step
            chunky_truncated = jnp.zeros_like(chunky_dones)

            actor_state = ActorTrainState(
            step=0,
            apply_fn=actor_apply,
            model_def=None,
            params=FrozenDict(),
            tx=None,
            opt_state=None,
            batch_stats=FrozenDict(),
            old_params=FrozenDict(),
            )

            new_network = agent.network

            qf_state, metrics, rng = cfg.update_chunky_critic(
                gamma=float(agent.config["discount"]),
                actor_state=actor_state,
                qf_state=agent.qf_state,
                ent_coef_state=agent.ent_coef_state,
                chunky_observations=chunky_observations,
                chunky_actions=chunky_actions,
                chunky_rewards=chunky_rewards,
                chunky_dones=chunky_dones,
                chunky_truncated=chunky_truncated,
                chunky_next_observations=chunky_next_observations,
                key=rng,
                sampler=None,
            )

            agent = agent.replace(qf_state=qf_state)
            info.update({f"critic/{k}": v for k, v in metrics.items()})
        
        else:
            new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
            agent.target_update(new_network, 'critic')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            if self.config['chunky_fql']:
                raise ValueError("Chunky FQL not supported for best-of-n actor type")
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["ent_coef_init"] == "auto":
            ent_coef_module = EntropyCoef(config["ent_coef_init"])
        else:
            ent_coef_module = ConstantEntropyCoef(float(config["ent_coef_init"]))
        ent_key, rng = jax.random.split(rng)
        ent_coef_state = TrainState.create(
            model_def=ent_coef_module,
            params=ent_coef_module.init(ent_key)["params"],
            tx=optax.adam(learning_rate=config["lr"]),
        )
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        if config['chunky_fql']:
            critic_def = EnsembleTransformerCritic(
                n_critics=config["num_qs"],
                n_embed=config["critic_n_embed"],
                n_heads=config["critic_n_heads"],
                n_layer=config["critic_n_layer"],
                dropout_rate=config["critic_dropout_rate"],
                block_size=config["critic_chunksize"] + 2,
                relative_pos=True,
                norm=config["critic_norm"],
                weight_norm=config["critic_weight_norm"],
                distributional=config["critic_distributional"],
                n_atoms=config.get("critic_n_atoms", 0),
            )
        else:
            critic_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
            )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        if config['chunky_fql']:
            critic_key, rng = jax.random.split(rng)
            ex_obs_b = jnp.expand_dims(ex_observations, 0)
            ex_act_b = jnp.expand_dims(ex_actions, 0)
            ex_chunky_actions = jnp.tile(ex_act_b[:, None, :], (1, config["critic_chunksize"], 1))
            critic_vars = critic_def.init(critic_key, ex_obs_b, ex_chunky_actions, train=True)
            qf_params = critic_vars["params"]
            qf_batch_stats = critic_vars.get("batch_stats", FrozenDict())
            qf_state = RLTrainState.create(
                model_def=critic_def,
                params=qf_params,
                tx=optax.adam(config["lr"]),
                batch_stats=qf_batch_stats,
                target_params=qf_params,
                target_batch_stats=qf_batch_stats,
            )
        else:
            qf_state = None

        
        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
        )
        if not config['chunky_fql']:
            network_info["critic"]=(critic_def, (ex_observations, full_actions))
            network_info["target_critic"]=(copy.deepcopy(critic_def), (ex_observations, full_actions))
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        if not config["chunky_fql"]:
            params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, qf_state=qf_state, ent_coef_state=ent_coef_state, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values. Changed from mean
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            chunky_fql=True,
            ent_coef_init=1.0, # or 'auto'->not implemented yet
            critic_n_embed=128, #Try 512 later
            critic_n_heads=8,
            critic_n_layer=2,
            critic_dropout_rate=0.0,
            critic_norm="ln",
            critic_weight_norm=False,
            critic_distributional=False,
            critic_n_atoms=0,
            critic_chunksize=5,
            actor_chunksize=5,
        )
    )
    return config
