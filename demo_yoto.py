
"""Run Native20 and ROA-B7 on one image with a frozen YOTO-NR model."""

import argparse
import json
from pathlib import Path

from yoto_v7.sampling import native_coordinates, sobol_coordinates
from yoto_v7.scoring import ImageSpec, YotoScorer, image_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = json.loads(
        (root / "configs" / "BUDGET_LOCK.json").read_text(encoding="utf-8")
    )
    budget = int(config["B_fixed"])
    if int(config["native_budget"]) != 20 or not 1 <= budget <= 19:
        raise ValueError("Expected YOTO Native20 and a valid ROA budget")

    scorer = YotoScorer(
        dataset="KONIQ",
        asset_root=args.asset_root,
        data_root=args.image.resolve().parent,
        checkpoint=args.checkpoint,
        device=args.device,
        crop_batch_size=8,
    )
    image = scorer.load_image(ImageSpec(args.image.name, args.image.name))
    height, width = image.shape[1:]

    native_positions, seed = native_coordinates(
        height, width, "KONIQ", "S01", "FORMAL_TEST_NATIVE", args.image.name, 0
    )
    methods = {}
    for name, positions in (
        ("Native20", native_positions),
        ("ROA", sobol_coordinates(height, width, budget)),
    ):
        scored = scorer.score_coordinates(image, positions)
        methods[name] = {
            **scored,
            "coordinates": positions,
            "prediction": image_mean(scored["scores"]),
        }

    result = {
        "image_id": args.image.name,
        "native_seed": seed,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Native20: {methods['Native20']['prediction']:.10f}")
    print(f"ROA-B{budget}: {methods['ROA']['prediction']:.10f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
