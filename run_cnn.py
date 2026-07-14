import os
import argparse
from cnn import train_cnn

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train CNN PPO agent (server)')
    ap.add_argument('--episodes',             type=int, default=20000)
    ap.add_argument('--checkpoint-dir',       default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'cnn'))
    ap.add_argument('--checkpoint-interval',  type=int, default=500,
                    help='Save checkpoint every N episodes (0 = off)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    train_cnn(
        episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
