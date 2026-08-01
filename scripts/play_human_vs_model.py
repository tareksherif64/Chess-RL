"""Play a game against a trained checkpoint in the desktop GUI.

Usage:
    python scripts/play_human_vs_model.py --checkpoint checkpoints_overnight/iter_000012.pt
    python scripts/play_human_vs_model.py --checkpoint path/to/ckpt.pt --simulations 400
    python scripts/play_human_vs_model.py --checkpoint path/to/ckpt.pt --cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.device import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a .pt checkpoint saved by training/checkpoint.py")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per agent move")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--cpu", action="store_true", help="allow CPU fallback (default requires CUDA)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    device = resolve_device(require_cuda=not args.cpu)
    print(f"device: {device}")

    from gui.human_vs_model_app import HumanVsModelApp

    app = HumanVsModelApp(
        checkpoint_path=args.checkpoint,
        device=device,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
