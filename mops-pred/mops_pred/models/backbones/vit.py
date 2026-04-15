import torch
import torchvision

from .backbone_factory import BackboneNonSeqABC, register_backbone


class VisionCompLayer(torch.nn.Module):
    def forward(self, x):
        patch_tokens = x[:, 1:]
        batch_size = patch_tokens.size(0)
        reshaped_tokens = patch_tokens.view(batch_size, 14, 14, 768)
        reshaped_tokens = reshaped_tokens.permute(
            0, 3, 1, 2
        )  # Change to (batch_size, dim, height, width)

        return reshaped_tokens


class Tokenize(torch.nn.Module):
    def __init__(self, vit_backbone) -> None:
        super().__init__()
        self.vit_backbone = vit_backbone

    def forward(self, x):
        n = x.shape[0]
        x = self.vit_backbone._process_input(x)
        batch_class_token = self.vit_backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x


@register_backbone(name="vit")
class ViTBackbone(BackboneNonSeqABC):
    def __init__(self) -> None:
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        vit_backbone = torchvision.models.vit_b_16(weights=weights)

        modules = torch.nn.Sequential(
            Tokenize(vit_backbone),
            vit_backbone.encoder,
            VisionCompLayer(),
        )

        super().__init__(weights, modules, channels=768, width=14, height=14)
