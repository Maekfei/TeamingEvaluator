#!/usr/bin/env python3
"""
Example
-------
python inference.py \
   --ckpt runs/20250217_121314/best_model_epoch050_male0.4321_team.pt \
   --snapshots "data/raw/G_{}.pt" \
   --year 2005 \
   --authors 123456 789012 345678 \
   --topic_emb path/to/topic_vec.npy \
   --device cuda:0
"""
import argparse, pickle, torch
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      required=True)
    p.add_argument('--snapshots', required=True,
                   help='Pattern with one {} placeholder for the year, e.g. '
                        '"data/raw/G_{}.pt"')
    p.add_argument('--year', type=int, required=True,
                   help='Publication year of the queried paper')
    p.add_argument('--authors', nargs='+', required=True,
                   help='List of author IDs (same format as in the data set)')
    p.add_argument('--topic_emb', required=True,
                   help='Path to a NumPy file holding the topic embedding')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available()
                                             else 'cpu')
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # load checkpoint – provides both the model hyper-params and weights
    # ------------------------------------------------------------------ #
    ckpt = torch.load(args.ckpt, map_location=args.device)
    saved_args = ckpt['args']                 # training arguments namespace

    # -------- mapping tables (author <→ idx) ---------------------------
    META_CACHE = "data/yearly_snapshots/mappings.pkl"
    with open(META_CACHE, "rb") as fh:
        _, AUT2IDX, _ = pickle.load(fh)

    idx2aut = [None] * len(AUT2IDX)
    for a, i in AUT2IDX.items():
        idx2aut[i] = a

    # -------- load 5-year window of snapshots -------------------------
    years_needed = list(range(args.year - 4, args.year + 1))
    snapshots = load_snapshots(args.snapshots, years_needed)
    snapshots = [g.to(args.device) for g in snapshots]

    # must match the order used during training
    metadata = snapshots[0].metadata()
    in_dims  = {
        "author": snapshots[0]["author"].x.size(-1),
        "paper":  snapshots[0]["paper"].x_title_emb.size(-1),
        "venue":  snapshots[0]["venue"].x.size(-1),
    }

    model = ImpactModel(metadata,
                        in_dims,
                        hidden_dim = saved_args.hidden_dim,
                        beta       = saved_args.beta,
                        cold_start_prob = 0.0,
                        aut2idx     = AUT2IDX,
                        idx2aut     = idx2aut).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    # topic embedding ---------------------------------------------------
    topic_vec = torch.from_numpy(
        __import__('numpy').load(args.topic_emb)
    ).float().to(args.device)                # [hidden_dim]

    # the index of the "current" year within the `snapshots` list
    current_year_idx = len(snapshots) - 1

    with torch.no_grad():
        yearly = model.predict_team(args.authors, topic_vec,
                                    snapshots, current_year_idx)

    print(f"Predicted yearly citations (years +1 … +5):")
    print(yearly.cpu().round(2).tolist())

if __name__ == '__main__':
    main()