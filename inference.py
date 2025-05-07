#!/usr/bin/env python3
"""
Example
-------
python inference.py \
   --ckpt runs/20250507_170412_team_resumedFrom_best_model_epoch005_male0.3637_team/evaluated_model_epoch020_male0_0.3730_male1_0.6771_male2_0.7959_male3_0.8540_male4_0.8855_team.pt \
   --snapshots "data/raw/G_{}.pt" \
   --year 2024 \
   --authors 6052561 6052561 \
   --topic_emb path/to/topic_vec.npy \
   --device cuda:1
"""

import argparse, os, pickle, sys, torch
import numpy as np
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel


# --------------------------------------------------------------------------- #
# helper: load exactly the years we can actually find on disk
# --------------------------------------------------------------------------- #
def load_available_snapshots(pattern: str, years: list[int], device: str):
    paths = [pattern.format(y) for y in years]
    missing = [y for y, p in zip(years, paths) if not os.path.isfile(p)]
    if missing:
        print(f"[warning] No graph file for years: {missing} – "
              f"these years will be skipped.", file=sys.stderr)
    years_kept = [y for y in years if y not in missing]
    if not years_kept:
        raise RuntimeError("None of the required yearly graph files exist.")
    snaps = load_snapshots(pattern, years_kept)
    return [g.to(device) for g in snaps], years_kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      required=True)
    p.add_argument('--snapshots', required=True,
                   help='Filename pattern with one {} placeholder for the year')
    p.add_argument('--year', type=int, required=True,
                   help='Publication year of the queried paper')
    p.add_argument('--authors', nargs='+', required=True,
                   help='List of author IDs (same string format as in the data)')
    p.add_argument('--topic_emb', required=True,
                   help='NumPy .npy file with the 256-d paper/topic embedding')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available()
                                             else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------ #
    # 1) checkpoint  (contains the full model state + the CLI arguments)
    # ------------------------------------------------------------------ #
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt['args']          # namespace that was passed to train.py

    # ------------------------------------------------------------------ #
    # 2) author–index mapping                                           #
    # ------------------------------------------------------------------ #
    META_CACHE = "data/yearly_snapshots/mappings.pkl"
    with open(META_CACHE, "rb") as fh:
        _, AUT2IDX, _ = pickle.load(fh)

    idx2aut = [None] * len(AUT2IDX)
    for a, i in AUT2IDX.items():
        idx2aut[i] = a

    # ------------------------------------------------------------------ #
    # 3) load the 5 (or fewer if missing) required yearly snapshots
    # ------------------------------------------------------------------ #
    years_needed = list(range(args.year - 4, args.year + 1))
    snapshots, yrs_kept = load_available_snapshots(args.snapshots,
                                                   years_needed, device)

    # the *index* of the current/publication year inside `snapshots`
    try:
        current_year_idx = yrs_kept.index(args.year)
    except ValueError:
        raise RuntimeError(f"The graph for the publication year {args.year} "
                           f"is missing – cannot run inference.")

    # ------------------------------------------------------------------ #
    # 4) rebuild the model skeleton and load weights
    # ------------------------------------------------------------------ #
    metadata = snapshots[0].metadata()
    in_dims  = {
        "author": snapshots[0]["author"].x.size(-1),
        "paper":  snapshots[0]["paper"].x_title_emb.size(-1),
        "venue":  snapshots[0]["venue"].x.size(-1),
    }

    model = ImpactModel(metadata,
                        in_dims,
                        hidden_dim       = saved_args.hidden_dim,
                        beta             = saved_args.beta,
                        cold_start_prob  = 0.0,   # no augmentation at inference
                        aut2idx          = AUT2IDX,
                        idx2aut          = idx2aut).to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    # ------------------------------------------------------------------ #
    # 5) prepare the query paper’s topic vector
    # ------------------------------------------------------------------ #
    topic_np = np.load(args.topic_emb)          # [256] or [1,256]
    # do a dummy random topic embedding here for testing
    # topic_np = np.random.rand(1, 256)           # [1, 256]
    topic_np = topic_np.squeeze().astype(np.float32)
    if topic_np.ndim != 1 or topic_np.size != in_dims['paper']:
        raise ValueError(f"Expected a flat vector with {in_dims['paper']} "
                         f"values, got shape {topic_np.shape}")
    topic_vec = torch.from_numpy(topic_np).to(device)        # [256]

    # If the network was trained with hidden_dim < original dim,
    # the encoder contains a learned Linear projection – reuse it.
    if topic_vec.size(0) != saved_args.hidden_dim:
        if 'paper' not in model.encoder.lins:
            raise RuntimeError("Mismatch in topic embedding size but no "
                               "projection layer found in the encoder.")
        with torch.no_grad():
            topic_vec = model.encoder.lins['paper'](topic_vec)   # → [H]
    # final sanity check
    assert topic_vec.size(0) == saved_args.hidden_dim

    # ------------------------------------------------------------------ #
    # 6) run the prediction
    # ------------------------------------------------------------------ #
    #  author IDs must be *strings*  (that is how they are stored in AUT2IDX)
    author_ids = [str(a) for a in args.authors]

    with torch.no_grad():
        yearly_cits = model.predict_team(author_ids,
                                         topic_vec,
                                         snapshots,
                                         current_year_idx)    # Tensor [5]

    print("Predicted yearly citations (years +1 … +5):")
    print([round(float(x), 2) for x in yearly_cits.cpu()])


if __name__ == '__main__':
    main()