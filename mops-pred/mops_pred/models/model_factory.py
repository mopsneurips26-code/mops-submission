from __future__ import annotations

import dataclasses

from mops_pred.config import ModelConfig

from .backbones import backbone_factory

_MODEL_REPOSITORY = {}


def register_model(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _MODEL_REPOSITORY:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODEL_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create_model(model_cfg: ModelConfig):
    """Instantiate a registered model from a ModelConfig.

    The ``name`` field selects the model class; remaining non-None fields are
    forwarded as constructor arguments.  If ``backbone`` is set it is
    instantiated first via ``backbone_factory`` and injected as the first
    positional argument.
    """
    kwargs = {
        k: v
        for k, v in dataclasses.asdict(model_cfg).items()
        if k not in ("name", "backbone") and v is not None
    }

    cls = _MODEL_REPOSITORY[model_cfg.name]
    if model_cfg.backbone is not None:
        backbone = backbone_factory.create_backbone(model_cfg.backbone)
        return cls(backbone, **kwargs)
    return cls(**kwargs)
