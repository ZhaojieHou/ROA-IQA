
"""Load the original YOTO-NR network for inference."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SWIN_WEIGHT = "swin_tiny_patch4_window7_224_22k.pth"


def _load_author_model(source: Path):
    """Import YOTO's original model and its bundled timm package."""
    package_name = "_roa_yoto_models"
    module_name = package_name + ".model"
    if module_name in sys.modules:
        return sys.modules[module_name]

    models_dir = source / "models"
    bundled_timm = (source / "timm" / "__init__.py").resolve()

    if "timm" in sys.modules:
        loaded = Path(
            getattr(sys.modules["timm"], "__file__", "") or ""
        ).resolve()
        if loaded != bundled_timm:
            raise RuntimeError(
                "A different timm package is already loaded. "
                "Start a fresh Python process."
            )

    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    package_spec = importlib.util.spec_from_file_location(
        package_name,
        models_dir / "__init__.py",
        submodule_search_locations=[str(models_dir)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError("Cannot import the original YOTO models package")

    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)

    model_spec = importlib.util.spec_from_file_location(
        module_name, models_dir / "model.py"
    )
    if model_spec is None or model_spec.loader is None:
        raise ImportError("Cannot import the original YOTO model")

    module = importlib.util.module_from_spec(model_spec)
    sys.modules[module_name] = module
    model_spec.loader.exec_module(module)

    import timm

    if Path(timm.__file__).resolve() != bundled_timm:
        raise RuntimeError(
            "YOTO must use the timm package bundled "
            "with its original source"
        )
    return module


def make_model(device, asset_root: Path | str):
    """Construct YOTO-NR using the author's model implementation."""
    root = Path(asset_root).resolve()
    source = root / "source" / "YOTO_official_snapshot"
    initializer = root / "models" / "pretrained" / SWIN_WEIGHT

    for required in (
        source / "models" / "__init__.py",
        source / "models" / "model.py",
        source / "models" / "swin.py",
        source / "timm" / "__init__.py",
        initializer,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required YOTO asset not found: {required}"
            )

    module = _load_author_model(source)
    model = module.Net(
        SimpleNamespace(root_dir=str(root)),
        device=str(device),
    )
    return model.to(device)
