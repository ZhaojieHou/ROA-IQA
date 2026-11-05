
"""Load the YOTO-NR network with the original Swin-T 22K initializer.

The author's constructor uses ``strict=False`` for backbone initialization.
This adapter preserves the original tensor-key and tensor-value checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


WEIGHT_NAME = "swin_tiny_patch4_window7_224_22k.pth"
EXPECTED_MISSING = {f"norm{i}.{part}" for i in range(4) for part in ("weight", "bias")}
EXPECTED_UNEXPECTED = {
    "layers.0.blocks.1.attn_mask",
    "layers.1.blocks.1.attn_mask",
    "layers.2.blocks.1.attn_mask",
    "layers.2.blocks.3.attn_mask",
    "layers.2.blocks.5.attn_mask",
    "norm.weight", "norm.bias", "head.weight", "head.bias",
}


def _source_and_weight(asset_root: Path) -> tuple[Path, Path]:
    source = asset_root / "source" / "YOTO_official_snapshot"
    weight = asset_root / "models" / "pretrained" / WEIGHT_NAME
    for required in (
        source / "models" / "__init__.py",
        source / "models" / "model.py",
        source / "models" / "swin.py",
        source / "timm" / "__init__.py",
        weight,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required original YOTO asset is missing: {required}"
            )
    return source, weight


def _load_author_net_module(source: Path):
    """Import the frozen model under a private package name, with vendored timm."""
    alias = "_yoto_frozen_models_" + hashlib.sha256(
        str(source).encode("utf-8")
    ).hexdigest()[:12]
    name = alias + ".model"
    if name in sys.modules:
        return sys.modules[name]
    vendored_timm = (source / "timm" / "__init__.py").resolve()
    if "timm" in sys.modules:
        loaded = Path(
            getattr(sys.modules["timm"], "__file__", "")
        ).resolve()
        if loaded != vendored_timm:
            raise RuntimeError(
                f"A different timm is already imported: {loaded}"
            )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    package_dir = source / "models"
    package_spec = importlib.util.spec_from_file_location(
        alias,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError("Cannot load frozen YOTO models package")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[alias] = package
    package_spec.loader.exec_module(package)
    model_spec = importlib.util.spec_from_file_location(
        name, package_dir / "model.py"
    )
    if model_spec is None or model_spec.loader is None:
        raise ImportError("Cannot load frozen YOTO model.py")
    module = importlib.util.module_from_spec(model_spec)
    sys.modules[name] = module
    model_spec.loader.exec_module(module)
    import timm

    if Path(timm.__file__).resolve() != vendored_timm:
        raise RuntimeError(
            "Frozen YOTO model imported a non-vendored timm"
        )
    return module


def make_model(
    device,
    asset_root: Path | str | None = None,
    *,
    release_root: Path | str | None = None,
):
    """Return author's NR model initialized with Swin-T 22K tensors.

    Training should call `model(x, x, mode="NR")`. The model is left in its
    constructor's training state; the caller controls `.train()` and `.eval()`.
    """
    import torch

    if (asset_root is None) == (release_root is None):
        raise ValueError(
            "Pass exactly one explicit asset_root or release_root"
        )
    root = Path(
        asset_root if asset_root is not None else release_root
    ).resolve()
    source, weight = _source_and_weight(root)
    payload = torch.load(
        weight, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or set(payload) != {"model"}:
        raise ValueError(
            "Swin-T checkpoint must contain exactly the 'model' state dict"
        )
    initializer = payload["model"]
    if (
        not isinstance(initializer, dict)
        or len(initializer) != 190
        or not all(
            isinstance(tensor, torch.Tensor)
            for tensor in initializer.values()
        )
        or tuple(initializer["head.weight"].shape) != (21841, 768)
        or tuple(initializer["head.bias"].shape) != (21841,)
        or tuple(initializer["patch_embed.proj.weight"].shape)
        != (96, 3, 4, 4)
    ):
        raise ValueError(
            "Swin-T checkpoint tensor structure is not "
            "the official 22K initializer"
        )
    author_model = _load_author_net_module(source)
    model = author_model.Net(
        SimpleNamespace(root_dir=str(root)), device=str(device)
    )
    actual = model.encoder.state_dict()
    missing = set(actual) - set(initializer)
    unexpected = set(initializer) - set(actual)
    common = set(actual) & set(initializer)
    shape_mismatch = {
        key
        for key in common
        if actual[key].shape != initializer[key].shape
    }
    matched = common - shape_mismatch
    if (
        len(actual) != 189
        or len(matched) != 181
        or missing != EXPECTED_MISSING
        or unexpected != EXPECTED_UNEXPECTED
        or shape_mismatch
    ):
        raise ValueError(
            "Swin-T/YOTO key or shape mismatch: "
            f"matched={len(matched)}, missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, "
            f"shape_mismatch={sorted(shape_mismatch)}"
        )
    unloaded = [
        key
        for key in sorted(matched)
        if not torch.equal(
            actual[key].cpu(), initializer[key].cpu()
        )
    ]
    if unloaded:
        raise ValueError(
            f"Swin-T shared tensors were not loaded: {unloaded}"
        )
    return model.to(device)
