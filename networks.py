"""The two policy networks the agents can act with, chosen by --policy.

Both take the same observation features and expose the same heads: accept or reject, a Beta over the share
of the pool to ask for, and, with cheap talk, a message token. They differ only in the trunk:

    mlp   ActorCritic           two tanh layers on the current observation alone. Within a negotiation it
                                is memoryless: every offer is judged on what is on the table right now.
    gru   RecurrentActorCritic  a GRU that carries a hidden state along the agent's own decisions in the
                                episode, so the policy can condition on the whole exchange so far: what it
                                offered, what came back, what was said, and how many rounds have burned.

Both keep the actor and the critic in separate trunks, so switching between them changes the trunk and
nothing else. Attribute names are part of the checkpoint format: renaming one breaks saved runs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical


def layer(in_dim, out_dim, std=np.sqrt(2)):
    linear = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(linear.weight, std)
    nn.init.zeros_(linear.bias)
    return linear


def action_dists(response_head, offer_head, message_head, h):
    """The action distributions from a trunk's output: response, share and, with cheap talk, message."""
    # Concentrations >= 1 keep the Beta unimodal; the near-zero head init starts it broad around 50/50.
    alpha, beta = (F.softplus(offer_head(h)) + 1.0).unbind(-1)
    # Argument validation is slow relative to networks this small; shares are clamped into (0, 1) anyway.
    dists = {
        "response": Categorical(logits=response_head(h), validate_args=False),
        "share": Beta(alpha, beta, validate_args=False),
    }
    if message_head is not None:
        dists["message"] = Categorical(logits=message_head(h), validate_args=False)
    return dists


class ActorCritic(nn.Module):
    """Separate two-hidden-layer tanh MLPs for the policy and the value function."""

    recurrent = False

    def __init__(self, obs_dim, hidden=64, message_vocab=0):
        super().__init__()
        self.actor = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.response_head = layer(hidden, 2, std=0.01)
        self.offer_head = layer(hidden, 2, std=0.01)
        self.message_head = layer(hidden, message_vocab, std=0.01) if message_vocab else None
        self.critic = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, 1, std=1.0)
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def trunk(self, obs):
        """The actor's hidden activations, layer by layer; the demo dashboard draws these."""
        first = self.actor[:2](obs)
        return first, self.actor[2:](first)

    def policy(self, obs, hidden=None):
        h = self.actor(obs) if hidden is None else hidden
        return action_dists(self.response_head, self.offer_head, self.message_head, h)

    def step(self, obs, state=None, capture=False):
        """One decision for a batch of streams. Memoryless, so state passes straight through as None."""
        first, hidden = self.trunk(obs) if capture else (None, None)
        dists = self.policy(obs, hidden=hidden)
        activations = {"layer1": first, "layer2": hidden} if capture else None
        return dists, self.value(obs), None, activations

    def sequence(self, obs_seq):
        """Distributions and values for a padded (time, batch, features) block, flattened over time."""
        flat = obs_seq.reshape(-1, obs_seq.shape[-1])
        return self.policy(flat), self.value(flat)


class RecurrentActorCritic(nn.Module):
    """A GRU over the agent's own decisions in the episode, for the actor and for the critic."""

    recurrent = True

    def __init__(self, obs_dim, hidden=64, message_vocab=0):
        super().__init__()
        self.hidden = hidden
        self.actor_encoder = nn.Sequential(layer(obs_dim, hidden), nn.Tanh())
        self.critic_encoder = nn.Sequential(layer(obs_dim, hidden), nn.Tanh())
        self.actor_gru = nn.GRU(hidden, hidden)
        self.critic_gru = nn.GRU(hidden, hidden)
        for gru in (self.actor_gru, self.critic_gru):  # orthogonal recurrent weights keep long rollouts stable
            for name, parameter in gru.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(parameter)
                else:
                    nn.init.orthogonal_(parameter, 1.0)
        self.response_head = layer(hidden, 2, std=0.01)
        self.offer_head = layer(hidden, 2, std=0.01)
        self.message_head = layer(hidden, message_vocab, std=0.01) if message_vocab else None
        self.value_head = layer(hidden, 1, std=1.0)

    def initial_state(self, streams, device):
        zeros = torch.zeros(1, streams, self.hidden, device=device)
        return zeros, zeros.clone()

    def policy(self, h):
        return action_dists(self.response_head, self.offer_head, self.message_head, h)

    def step(self, obs, state=None, capture=False):
        """One decision for a batch of streams, carrying each stream's hidden state forward."""
        actor_state, critic_state = state if state is not None else (None, None)
        embedded = self.actor_encoder(obs)
        actor_out, actor_next = self.actor_gru(embedded.unsqueeze(0), actor_state)
        critic_out, critic_next = self.critic_gru(self.critic_encoder(obs).unsqueeze(0), critic_state)
        memory = actor_out.squeeze(0)
        value = self.value_head(critic_out.squeeze(0)).squeeze(-1)
        activations = {"layer1": embedded, "layer2": memory} if capture else None
        return self.policy(memory), value, (actor_next, critic_next), activations

    def sequence(self, obs_seq):
        """Distributions and values for a padded (time, batch, features) block, flattened over time.

        Every episode starts from a zero state, so a block can be replayed from the start; padded steps come
        after an episode ends and are dropped by the caller's mask.
        """
        time_steps, streams, _ = obs_seq.shape
        actor_out, _ = self.actor_gru(self.actor_encoder(obs_seq))
        critic_out, _ = self.critic_gru(self.critic_encoder(obs_seq))
        flat_actor = actor_out.reshape(time_steps * streams, -1)
        value = self.value_head(critic_out.reshape(time_steps * streams, -1)).squeeze(-1)
        return self.policy(flat_actor), value


NETWORKS = {"mlp": ActorCritic, "gru": RecurrentActorCritic}


def build_network(obs_dim, hidden, message_vocab, kind="mlp"):
    if kind not in NETWORKS:
        raise ValueError(f"unknown policy {kind!r}; choose from {', '.join(NETWORKS)}")
    return NETWORKS[kind](obs_dim, hidden, message_vocab)
