"""
This module was adapted from https://ravinkumar.com/GenAiGuidebook/deepdive/GPTFromScratchFlax.html
"""

from typing import Union, Optional
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.linen.attention import make_causal_mask
from flax.linen.normalization import BatchNorm, LayerNorm
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

class MultiHeadAttention(nn.Module):
    """
    A module to perform multi-head attention using Flax's linen library.
    This combines multiple attention heads into a single operation.
    """

    num_heads: int
    n_embed: int
    dropout_rate: float
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x, training):
        """
        Apply multi-head attention to the input tensor.

        Parameters:
            x (tensor): Input tensor.
            training (bool): Flag to indicate if the model is training (affects dropout).

        Returns:
            tensor: Output tensor after applying multi-head attention and a dense layer.
        """
        mask = make_causal_mask(x[..., 0])
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, mask=mask)
        if self.weight_norm:
            x = nn.WeightNorm(nn.Dense(self.n_embed))(x)
        else:
            x = nn.Dense(self.n_embed)(x)
        return x


class FeedForward(nn.Module):
    """
    A feedforward neural network module using Flax's Linen API with two dense layers
    and a dropout layer for regularization.
    """

    n_embed: int
    dropout_rate: float
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x, training):
        """
        Applies a sequence of layers to the input tensor.

        Parameters:
            x (tensor): Input tensor to the feedforward network.
            training (bool): Flag to indicate if the model is training.

        Returns:
            tensor: The output tensor after processing through dense and dropout layers.
        """
        if self.weight_norm:
            x = nn.Sequential(
                [
                    nn.WeightNorm(nn.Dense(4 * self.n_embed)),
                    nn.PReLU(),  # other repo uses gelu
                    nn.WeightNorm(nn.Dense(self.n_embed)),
                    nn.Dropout(self.dropout_rate, deterministic=not training),
                ]
            )(x)
        else:
            x = nn.Sequential(
                [
                    nn.Dense(4 * self.n_embed),
                    nn.PReLU(),  # other repo uses gelu
                    nn.Dense(self.n_embed),
                    nn.Dropout(self.dropout_rate, deterministic=not training),
                ]
            )(x)
        return x


class Block(nn.Module):
    """
    A transformer block module using Flax's linen API, which integrates multi-head attention
    and feedforward neural network layers.
    """

    n_embed: int
    n_heads: int
    dropout_rate: float
    norm: str
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        """
        Process the input tensor through the transformer block.

        Parameters:
            x (tensor): Input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            dict: A dictionary containing the output tensor and the training status.
        """
        # TODO: follow initialization as in Bruce's paper

        # Initialize the MultiHeadAttention and FeedForward modules
        sa = MultiHeadAttention(
            num_heads=self.n_heads,
            n_embed=self.n_embed,
            dropout_rate=self.dropout_rate,
            weight_norm=self.weight_norm,
        )
        ff = FeedForward(
            n_embed=self.n_embed,
            dropout_rate=self.dropout_rate,
            weight_norm=self.weight_norm,
        )

        # Apply self-attention and residual connection followed by layer normalization
        norm = get_norm(self.norm)
        x = x + sa(norm(x), training=training)

        # Apply feedforward network and residual connection followed by layer normalization
        norm = get_norm(self.norm)
        x = x + ff(norm(x), training=training)

        return x


class SeqQFunc(nn.Module):
    norm: str
    n_embed: int
    n_heads: int
    dropout_rate: float
    block_size: int
    n_layer: int
    relative_pos: bool
    weight_norm: bool
    distributional: bool
    out_dim: int = 1
    n_atoms: int = 0

    # removing typechecker for now
    #@jaxtyped(typechecker=typechecker) 
    @nn.compact
    def __call__(
        self,
        state: Float[Array, "batch obs"],
        actions: Float[Array, "batch chunk action"],
        training: bool,
    ) -> Union[Float[Array, "batch q_and_chunk"], Float[Array, "batch chunk n_atoms"]]:
        batch_size, critic_chunksize, _ = actions.shape
        idx_c = None
        idx_a = None
        # idx_c: time indices of current state
        # idx_a: time indices of actions
        if actions is None:
            assert idx_a is None
            t = 0
        else:
            t = actions.shape[-2]  # TODO: check dimensions
        assert t + 2 <= self.block_size

        # state embedding
        if self.weight_norm:
            state_embed = nn.WeightNorm(
                nn.Dense(self.n_embed, use_bias=False, name="StateEmbedding")
            )(state)
        else:
            state_embed = nn.Dense(self.n_embed, use_bias=False, name="StateEmbedding")(
                state
            )

        if actions is not None:
            # action embeding
            if self.weight_norm:
                action_embed = nn.WeightNorm(
                    nn.Dense(self.n_embed, use_bias=False, name="ActionEmbedding")
                )(actions)
            else:
                action_embed = nn.Dense(
                    self.n_embed, use_bias=False, name="ActionEmbedding"
                )(actions)
            assert isinstance(action_embed, Float[Array, "batch chunk n_embed"])  # type: ignore

            state_embed = jnp.expand_dims(state_embed, 1)
            assert isinstance(state_embed, Float[Array, "batch 1 n_embed"])  # type: ignore
            seq_embed = jnp.concatenate(
                [state_embed, action_embed], axis=1
            )  # TODO: Check axis
            assert self.relative_pos
            # seq_pos = jnp.concatenate(
            #    [idx_c, idx_a], axis=-1
            # )  # time indices of current staten and action
        else:
            seq_embed = state_embed
            seq_pos = idx_c

        if self.relative_pos:
            seq_pos = jnp.arange(0, 1 + t)  # TODO: First element should also be 1?
        # positional encoder
        pos_embed = nn.Embed(self.block_size, self.n_embed, name="PositionEmbedding")(
            seq_pos  # type: ignore[reportUnboundVariable]
        )

        # dropout
        # Bruce does dropout here but not sure how to do it here?
        x = seq_embed + pos_embed  # TODO: This should be a concatination? -- Max

        for _ in range(self.n_layer):
            x = Block(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                norm=self.norm,
                weight_norm=self.weight_norm,
            )(x, training=training)

        # layernorm
        norm = get_norm(self.norm)
        x = norm(x)

        # Print parameter count (only during initialization)
        if self.is_initializing():
            total_params = sum(
                p.size for p in jax.tree_util.tree_leaves(self.variables["params"])
            )
            print(f"Transformer Critic Parameters: {total_params:,}")

        # keep BatchNorm Code working
        _ = self.variable("batch_stats", "dummy", lambda _: jnp.zeros(1), 1)

        if self.distributional:
            n_atoms = self.n_atoms
            if self.weight_norm:
                x = nn.WeightNorm(nn.Dense(n_atoms))(x)
            else:
                x = nn.Dense(n_atoms)(x)
            x = x[:, 1:, :]  # drop v
            return jax.nn.softmax(x, axis=-1)
        else:
            assert self.out_dim == 1

            if self.weight_norm:
                x = nn.WeightNorm(nn.Dense(self.out_dim))(x)
            else:
                x = nn.Dense(self.out_dim)(x)
            # x = nn.Dense(1, kernel_init=nn.initializers.constant(1e-6),
            #                 bias_init=nn.initializers.constant(0.0))(x)
            return x.reshape((batch_size, critic_chunksize + 1))

# porting ensemble behaviour from jax_diff_rl/sac/policies.py
class EnsembleTransformerCritic(nn.Module):
    n_critics: int
    n_embed: int
    n_heads: int
    n_layer: int
    dropout_rate: float
    block_size: int
    relative_pos: bool
    norm: str
    weight_norm: bool
    distributional: bool
    n_atoms: int

    @nn.compact
    def __call__(
        self,
        obs: Float[Array, "batch obs"],
        actions: Float[Array, "batch chunk action"],
        train: bool = True,
    ):
        vmap_critic = nn.vmap(
            SeqQFunc,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,  # same inputs go to each critic
            out_axes=0,
            axis_size=self.n_critics,
        )

        q_values = vmap_critic(
            n_embed=self.n_embed,
            n_heads=self.n_heads,
            n_layer=self.n_layer,
            dropout_rate=self.dropout_rate,
            block_size=self.block_size,
            relative_pos=self.relative_pos,
            norm=self.norm,
            weight_norm=self.weight_norm,
            distributional=self.distributional,
            n_atoms=self.n_atoms,
        )(obs, actions, train)

        return q_values

def get_norm(norm) -> nn.Module:
    if norm == "ln":
        return LayerNorm()
    else:
        raise NotImplementedError