import copy
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModel, CLIPTextModel, T5EncoderModel

from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..factory import register_policy
from .configuration_affflow import AffFlowConfig


def _get_activation_fn(activation: str):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class _TimeNetwork(nn.Module):
    def __init__(
        self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000
    ) -> None:
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)
        super().__init__()

        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        assert len(t.shape) == 1, "assumes 1d input timestep array"
        t = t[:, None] * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create mlp
        self.mlp = nn.Sequential(
            self.linear1,
            self.activation,
            self.dropout2,
            self.linear2,
            self.dropout3,
        )

        # create modulation layers
        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        x3 = self.mlp_modulate(self.norm2(x), cond)
        x3 = self.mlp(x3)
        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

    def reset_parameters(self) -> None:
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, t, cond):
        x = src
        for layer in self.layers:
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        cond_dim,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        clip_sample=False,
        clip_sample_range=1.0,
    ) -> None:
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk

        # positional encoding blocks
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        # clip the output samples
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        print(
            f"Number of flow params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    def forward(self, noisy_actions, time, global_cond):
        c = self.cond_proj(global_cond)
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noisy_actions)  # [B, T, adim] -> [B, T, hidden_dim]
        ac_tokens = ac_tokens.transpose(
            0, 1
        )  # [B, T, hidden_dim] -> [T, B, hidden_dim]

        # Allow variable length action chunks
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]  # [T, B, hidden_dim]

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, c)

        # apply final epsilon prediction layer
        eps_out = self.eps_out(
            dec_out, time_enc, c
        )  # [T, B, hidden_dim] -> [T, B, adim]
        return eps_out.transpose(0, 1)  # [T, B, adim] -> [B, T, adim]

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        timesteps: int = 100,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # Use Euler integration to solve the ODE.
        batch_size, device = condition.shape[0], condition.device
        x_0 = self.sample_noise(batch_size, device, generator)
        dt = 1.0 / timesteps
        t_all = (
            torch.arange(timesteps, device=device)
            .float()
            .unsqueeze(0)
            .expand(batch_size, timesteps)
            / timesteps
        )

        for k in range(timesteps):
            t = t_all[:, k]
            x_0 = x_0 + dt * self.forward(x_0, t, condition)
            if self.clip_sample:
                x_0 = torch.clamp(x_0, -self.clip_sample_range, self.clip_sample_range)
        return x_0

    def sample_noise(
        self, batch_size: int, device, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.randn(
            batch_size, self.ac_chunk, self.ac_dim, device=device, generator=generator
        )


@register_policy(policy_type="affflow")
class AffFlowPolicy(PreTrainedPolicy):
    """Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = AffFlowConfig
    name = "affflow"

    def __init__(
        self,
        config: AffFlowConfig,
    ) -> None:
        """Args:
        config: Policy configuration class instance or None, in which case the default instantiation of
            the configuration class is used.
        """
        super().__init__(config)

        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.dit_flow = DiTFlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.dit_flow.parameters()

    def reset(self) -> None:
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            for key in self.config.image_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        for key in batch:
            if key in self._queues:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)

        # batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.dit_flow.generate_actions(batch)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "action" in batch:
            batch.pop("action")  # remove action if present in the input batch

        batch = dict(batch)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        loss = self.dit_flow.compute_loss(batch)
        return loss, None


class DiTFlowModel(nn.Module):
    def __init__(self, config: AffFlowConfig) -> None:
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        # If image_only, don't include state features in conditioning
        global_cond_dim = (
            0 if self.config.image_only else self.config.robot_state_feature.shape[0]
        )
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [
                    DiffusionRgbEncoder(config, feature_key=key)
                    for key in self.config.image_features
                ]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += sum(enc.feature_dim for enc in encoders)
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        if self.config.use_text_conditioning:
            if "clip" in config.tokenizer_name.lower():
                self.text_encoder = CLIPTextModel.from_pretrained(config.tokenizer_name)
            elif "t5" in config.tokenizer_name.lower():
                self.text_encoder = T5EncoderModel.from_pretrained(
                    config.tokenizer_name
                )
            else:
                self.text_encoder = AutoModel.from_pretrained(config.tokenizer_name)

            if config.freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

            # Project CLIP embedding to desired dimension if needed,
            # though usually text_embedding_dim in config should match CLIP's output
            # For CLIP ViT-B/32 output is 512.
            # If we wanted to map to something else we could add a projection here.
            # self.text_proj = nn.Linear(config.text_embedding_dim, config.text_embedding_dim)

            encoded_dim = getattr(
                self.text_encoder.config, "hidden_size", None
            ) or getattr(self.text_encoder.config, "d_model", None)
            if encoded_dim != config.text_embedding_dim:
                raise ValueError(
                    f"Config text_embedding_dim {config.text_embedding_dim} does not match Encoder hidden size {encoded_dim}"
                )

            global_cond_dim += config.text_embedding_dim

        self.velocity_net = _DiTNoiseNet(
            ac_dim=config.action_feature.shape[0],
            ac_chunk=config.horizon,
            cond_dim=global_cond_dim * config.n_obs_steps,
            time_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

        self.num_inference_steps = config.num_inference_steps or 100
        self.training_noise_sampling = config.training_noise_sampling
        if config.training_noise_sampling == "uniform":
            self.noise_distribution = torch.distributions.Uniform(
                low=0,
                high=1,
            )
        elif config.training_noise_sampling == "beta":
            # From the Pi0 paper, https://www.physicalintelligence.company/download/pi0.pdf Appendix B.
            # There, they say the PDF for the distribution they use is the following:
            # $p(t) = Beta((s-t) / s; 1.5, 1)$
            # So, we first figure out the distribution over $t'$ and then transform it to $t = s - s * t'$.
            s = 0.999  # constant from the paper
            beta_dist = torch.distributions.Beta(
                concentration1=1.5,  # alpha
                concentration0=1.0,  # beta
            )
            affine_transform = torch.distributions.transforms.AffineTransform(
                loc=s, scale=-s
            )
            self.noise_distribution = torch.distributions.TransformedDistribution(
                beta_dist, [affine_transform]
            )

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Expand global conditioning to the batch size.
        if global_cond is not None:
            global_cond = global_cond.expand(batch_size, -1).to(
                device=device, dtype=dtype
            )

        # Sample prior.
        sample = self.velocity_net.sample(
            global_cond, timesteps=self.num_inference_steps, generator=generator
        )
        return sample

    def _prepare_global_conditioning(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        # If image_only, don't include state features in conditioning
        global_cond_feats = [] if self.config.image_only else [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            img_features_list = []
            if self.config.use_separate_rgb_encoder_per_camera:
                for i, key in enumerate(self.config.image_features):
                    img_features_list.append(
                        self.rgb_encoder[i](
                            einops.rearrange(batch[key], "b s ... -> (b s) ...")
                        )
                    )
            else:
                for key in self.config.image_features:
                    img_features_list.append(
                        self.rgb_encoder(
                            einops.rearrange(batch[key], "b s ... -> (b s) ...")
                        )
                    )
            # Concatenate all features along the feature dimension
            img_features = torch.cat(img_features_list, dim=-1)
            # Restore batch and sequence dimensions
            img_features = einops.rearrange(
                img_features, "(b s) ... -> b s ...", b=batch_size, s=n_obs_steps
            )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        if self.config.use_text_conditioning:
            tokens = batch[OBS_LANGUAGE_TOKENS]
            mask = batch[OBS_LANGUAGE_ATTENTION_MASK]

            # Handle both (B, L) and (B, S, L) cases
            flat_input = False
            if tokens.ndim == 2:
                # (B, L)
                tokens_flat = tokens
                mask_flat = mask
                flat_input = True
            else:
                # (B, S, L)
                b, s, l = tokens.shape
                tokens_flat = tokens.view(b * s, l)
                mask_flat = mask.view(b * s, l)

            outputs = self.text_encoder(input_ids=tokens_flat, attention_mask=mask_flat)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                text_feat = outputs.pooler_output
            else:
                # Mean pooling
                last_hidden_state = outputs.last_hidden_state
                input_mask_expanded = (
                    mask_flat.unsqueeze(-1).expand(last_hidden_state.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                text_feat = sum_embeddings / sum_mask

            if flat_input:
                # (B, D) -> (B, n_obs_steps, D)
                text_feat = text_feat.unsqueeze(1).expand(-1, n_obs_steps, -1)
            else:
                text_feat = text_feat.view(b, s, -1)  # (B, S, D)

            # Note: We removed the learnable projection self.text_proj as we use pretrained embeddings directly.
            # If fine-tuning adaptation is needed, a projection could be added back.

            global_cond_feats.append(text_feat)

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim).

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim).

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        has_images = self.config.image_features and all(
            key in batch for key in self.config.image_features
        )
        assert has_images or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        noise = self.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = self.noise_distribution.sample((trajectory.shape[0],)).to(
            trajectory.device
        )
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[
            :, None, None
        ] * trajectory

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.velocity_net(
            noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond
        )
        target = trajectory - noise
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
