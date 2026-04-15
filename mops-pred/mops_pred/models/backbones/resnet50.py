import torchvision

from .backbone_factory import BackboneABC, register_backbone


@register_backbone(name="resnet50")
class ResNetBackbone(BackboneABC):
    def __init__(self) -> None:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet = torchvision.models.resnet50(weights=weights)
        modules = list(resnet.children())[:-2]

        super().__init__(weights, modules, channels=2048, width=7, height=7)
