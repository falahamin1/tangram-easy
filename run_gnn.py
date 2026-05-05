import os
import argparse
from graphrep import train_graph_rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train GNN PPO agent — easy tangram (server)')
    ap.add_argument('--episodes',            type=int, default=20000)
    ap.add_argument('--checkpoint-dir',      default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'gnn'))
    ap.add_argument('--checkpoint-interval', type=int, default=500)
    args = ap.parse_args()
    train_graph_rep(episodes=args.episodes,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval)
