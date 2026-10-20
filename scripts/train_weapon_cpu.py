"""Compatibility entry point for a CPU-only SmartVision training run."""

import sys

from scripts.train_weapon_improved import build_parser, run_training


def main():
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.device == "auto":
        arguments.device = "cpu"
    try:
        run_training(arguments)
    except Exception as exc:
        print(
            f"CPU training failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
