import copy
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from transformers import CLIPTextModel

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
from .configuration_mopsflow import MopsFlowConfig


class SpatialRgbEncoder(DiffusionRgbEncoder):
    def forward_with_spatial_and_aux(
        self, x: torch.Tensor, aux: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Same as forward but returns spatial features and cropped auxiliary masks.

        Args:
            x: (B, C, H, W) image tensor.
            aux: (B, C_aux, H, W) auxiliary tensor (e.g. segmentation masks) to be cropped identically.

        Returns:
            pooled_features: (B, D)
            spatial_features: (B, C_backbone, H', W')
            aux_cropped: (B, C_aux, H', W') or None
        """
        # Preprocess: maybe crop
        if self.do_crop:
            if self.training and self.maybe_random_crop != self.center_crop:
                # Manual random crop to synchronize x and aux
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                    x, output_size=self.maybe_random_crop.size
                )
                x = torchvision.transforms.functional.crop(x, i, j, h, w)
                if aux is not None:
                    aux = torchvision.transforms.functional.crop(aux, i, j, h, w)
            else:
                # Deterministic center crop
                x = self.center_crop(x)
                if aux is not None:
                    aux = self.center_crop(aux)

        # Extract backbone feature.
        spatial_features = self.backbone(x)

        # Standard encoding path
        flat = torch.flatten(self.pool(spatial_features), start_dim=1)
        flat = self.relu(self.out(flat))

        return flat, spatial_features, aux


def _get_activation_fn(activation: str):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SegmentationDecoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_dim=128, num_layers=2, nhead=4
    ) -> None:
        super().__init__()
        # Projection to hidden dimension
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Learnable positional encoding
        # Optimized for 7x7 feature maps (standard ResNet18/34/50 output for 224px img)
        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim, 7, 7) * 0.02)

        # Transformer Encoder Bottleneck
        # Captures global context before upsampling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Lightweight Decoder (Upsample 32x)
        # 5 stages: 7 -> 14 -> 28 -> 56 -> 112 -> 224
        # Uses GroupNorm for batch-size independence
        self.decoder_layers = nn.ModuleList(
            [
                self._make_layer(hidden_dim, 64),
                self._make_layer(64, 64),
                self._make_layer(64, 32),
                self._make_layer(32, 32),
                self._make_layer(32, 16),
            ]
        )

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.input_proj(x)

        # Add position embedding (interpolate if resolution varies)
        pos = self.pos_embed
        if pos.shape[-2:] != (h, w):
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        x = x + pos

        # Apply Transformer
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        # Apply Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        return self.final_conv(x)


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


@register_policy(policy_type="mopsflow")
class MopsFlowPolicy(PreTrainedPolicy):
    config_class = MopsFlowConfig
    name = "mopsflow"

    def __init__(
        self,
        config: MopsFlowConfig,
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

        self.mops_flow = MopsFlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.mops_flow.parameters()

    def reset(self) -> None:
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            image_keys = [
                k for k in self.config.image_features if "affordance" not in k
            ]
            for key in image_keys:
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
        actions = self.mops_flow.generate_actions(batch)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "action" in batch:
            batch.pop("action")  # remove action if present in the input batch

        batch = dict(batch)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        loss, loss_dict = self.mops_flow.compute_loss(batch)
        return loss, loss_dict


class MopsFlowModel(nn.Module):
    def __init__(self, config: MopsFlowConfig) -> None:
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        # If image_only, don't include state features in conditioning
        global_cond_dim = (
            0 if self.config.image_only else self.config.robot_state_feature.shape[0]
        )
        if self.config.image_features:
            # Filter out affordance keys
            image_keys = [
                k for k in self.config.image_features if "affordance" not in k
            ]
            num_images = len(image_keys)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [
                    SpatialRgbEncoder(config, feature_key=key) for key in image_keys
                ]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += sum(enc.feature_dim for enc in encoders)
            else:
                self.rgb_encoder = SpatialRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        if self.config.use_text_conditioning:
            self.text_encoder = CLIPTextModel.from_pretrained(config.tokenizer_name)
            if config.freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

            # Project CLIP embedding to desired dimension if needed,
            # though usually text_embedding_dim in config should match CLIP's output
            # For CLIP ViT-B/32 output is 512.
            # If we wanted to map to something else we could add a projection here.
            # self.text_proj = nn.Linear(config.text_embedding_dim, config.text_embedding_dim)
            if self.text_encoder.config.hidden_size != config.text_embedding_dim:
                raise ValueError(
                    f"Config text_embedding_dim {config.text_embedding_dim} does not match CLIP hidden size {self.text_encoder.config.hidden_size}"
                )

            global_cond_dim += config.text_embedding_dim

        if self.config.segmentation_prediction_loss_weight > 0:
            # Determine backbone output channels by running a dummy pass
            if self.config.use_separate_rgb_encoder_per_camera:
                encoder = self.rgb_encoder[0]
            else:
                encoder = self.rgb_encoder

            # Use crop_shape or default 224
            h, w = config.crop_shape if config.crop_shape else (224, 224)
            dummy_image = torch.zeros(1, 3, h, w)

            with torch.no_grad():
                _, spatial, _ = encoder.forward_with_spatial_and_aux(dummy_image)

            in_channels = spatial.shape[1]
            out_channels = config.num_affordance_classes
            self.segmentation_decoder = SegmentationDecoder(in_channels, out_channels)
            self.segmentation_loss_fn = FocalLoss()

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
        self, batch: dict[str, torch.Tensor], compute_segmentation_loss: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        # If image_only, don't include state features in conditioning
        global_cond_feats = [] if self.config.image_only else [batch[OBS_STATE]]

        seg_loss = torch.tensor(0.0, device=batch[OBS_STATE].device)

        # Extract image features.
        if self.config.image_features:
            # Filter out affordance features
            image_keys = [
                k for k in self.config.image_features if "affordance" not in k
            ]

            def get_aux(idx):
                if (
                    compute_segmentation_loss
                    and self.config.affordance_keys
                    and idx < len(self.config.affordance_keys)
                ):
                    key = self.config.affordance_keys[idx]
                    if key in batch:
                        return batch[key]
                return None

            img_features_list = []
            for cam_idx, key in enumerate(image_keys):
                cam_images = batch[key]
                cam_images_flat = einops.rearrange(cam_images, "b s ... -> (b s) ...")

                if self.config.use_separate_rgb_encoder_per_camera:
                    encoder = self.rgb_encoder[cam_idx]
                else:
                    encoder = self.rgb_encoder

                aux = get_aux(cam_idx)
                aux_flat = None
                if aux is not None:
                    aux_flat = einops.rearrange(aux, "b s ... -> (b s) ...")

                feat, spatial, aux_cropped = encoder.forward_with_spatial_and_aux(
                    cam_images_flat, aux_flat
                )
                img_features_list.append(feat)

                if aux_cropped is not None:
                    pred = self.segmentation_decoder(spatial)
                    seg_loss += self.segmentation_loss_fn(pred, aux_cropped.float())

            # Concatenate all features along the feature dimension
            img_features_combined = torch.cat(img_features_list, dim=-1)
            img_features = einops.rearrange(
                img_features_combined,
                "(b s) d -> b s d",
                b=batch_size,
                s=n_obs_steps,
            )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        if self.config.use_text_conditioning:
            tokens = batch[OBS_LANGUAGE_TOKENS]
            mask = batch[OBS_LANGUAGE_ATTENTION_MASK]

            # Handle both (B, L) and (B, S, L) cases
            if tokens.ndim == 2:
                # (B, L)
                outputs = self.text_encoder(input_ids=tokens, attention_mask=mask)
                text_feat = outputs.pooler_output  # (B, D)
                # Expand to (B, n_obs_steps, D)
                text_feat = text_feat.unsqueeze(1).expand(-1, n_obs_steps, -1)
            else:
                # (B, S, L)
                b, s, l = tokens.shape
                tokens_flat = tokens.view(b * s, l)
                mask_flat = mask.view(b * s, l)

                outputs = self.text_encoder(
                    input_ids=tokens_flat, attention_mask=mask_flat
                )
                text_feat = outputs.pooler_output  # (B*S, D)
                text_feat = text_feat.view(b, s, -1)  # (B, S, D)

            # Note: We removed the learnable projection self.text_proj as we use pretrained embeddings directly.
            # If fine-tuning adaptation is needed, a projection could be added back.

            global_cond_feats.append(text_feat)

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1), seg_loss

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
        global_cond, _ = self._prepare_global_conditioning(
            batch, compute_segmentation_loss=False
        )  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
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

        non_aff_keys = []
        if self.config.image_features:
            non_aff_keys = [
                k for k in self.config.image_features if "affordance" not in k
            ]
        has_images = non_aff_keys and all(key in batch for key in non_aff_keys)
        assert has_images or "observation.environment_state" in batch

        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, seg_loss = self._prepare_global_conditioning(
            batch, compute_segmentation_loss=True
        )  # (B, global_cond_dim)

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

        flow_loss = loss.mean()
        weighted_seg_loss = seg_loss * self.config.segmentation_prediction_loss_weight
        total_loss = flow_loss + weighted_seg_loss

        loss_dict = {
            "flow_loss": flow_loss.item(),
            "segmentation_loss": seg_loss.item(),
            "loss": total_loss.item(),
        }

        return total_loss, loss_dict
