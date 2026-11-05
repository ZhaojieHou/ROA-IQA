"""Arrange user-obtained YOTO source and Swin initializer for the model loader."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--swin', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    args = parser.parse_args()
    for relative in ['models/__init__.py', 'models/model.py', 'models/swin.py', 'timm/__init__.py']:
        if not (args.source_dir / relative).is_file():
            raise FileNotFoundError(args.source_dir / relative)
    if not args.swin.is_file():
        raise FileNotFoundError(args.swin)
    if args.asset_root.exists():
        raise FileExistsError('Choose a new asset root; existing assets are not overwritten')
    target = args.asset_root / 'source' / 'YOTO_official_snapshot'
    shutil.copytree(args.source_dir, target,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    weights = args.asset_root / 'models' / 'pretrained'
    weights.mkdir(parents=True)
    shutil.copyfile(args.swin, weights / 'swin_tiny_patch4_window7_224_22k.pth')
    print('Assets prepared. Tensor structure and strict model loading are checked during inference.')


if __name__ == '__main__':
    main()
