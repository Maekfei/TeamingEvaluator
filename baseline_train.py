#!/usr/bin/env python3
"""
Unified CLI for classical & DeepCas baselines.

Examples
--------
# GBM on years 2005-2013
python baseline_train.py \
    --model gbm \
    --train_years 2005 2013 \
    --test_years 2016 2019

# DeepCas (needs your implementation) on GPU 0
python baseline_train.py \
    --model deepcas \
    --train_years 2005 2013 \
    --test_years 2016 2019 \
    --device cuda:0
"""
import argparse, os, time, torch, pickle
from utils.data_utils import load_snapshots
from baselines.gbm_model import GBMBaseline
from baselines.deepcas_model import DeepCasModel
from rich.console import Console

console = Console()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gbm", "deepcas"], required=True)
    p.add_argument("--train_years", nargs=2, type=int, required=True)
    p.add_argument("--test_years",  nargs=2, type=int, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    train_years = list(range(args.train_years[0], args.train_years[1] + 1))
    test_years  = list(range(args.test_years [0], args.test_years [1] + 1))
    console.print(f"[bold]Train years:[/bold] {train_years}")
    console.print(f"[bold]Test years:[/bold]  {test_years}")

    # ------------------------------------------------------------------
    snapshots = load_snapshots("data/raw/G_{}.pt", train_years + test_years)
    snapshots = [g.to(args.device) for g in snapshots]

    # ------------------------------------------------------------------
    run_dir = os.path.join("runs_baseline", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    if args.model == "gbm":
        model = GBMBaseline(device=args.device)
        console.print("Fitting GBM …")
        model.fit(snapshots, list(range(len(train_years))))
        male, rmsle = model.evaluate(
            snapshots,
            list(range(len(train_years), len(train_years) + len(test_years)))
        )

    elif args.model == "deepcas":
        device = torch.device(args.device)
        model = DeepCasModel(hidden_dim=64).to(device)

        # TODO: build DataLoader for random-walk batches
        train_loader, test_loader = None, None
        raise NotImplementedError("DeepCas data loader not yet implemented")

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, 51):
            loss = model.train_one_epoch(train_loader, optim, device)
            console.log(f"Epoch {epoch:03d}  loss {loss:.4f}")
        male, rmsle = model.evaluate(test_loader, device)

    console.print(f"[green]Finished.[/green]  MALE {male.tolist()}  "
                  f"RMSLE {rmsle.tolist()}")

    # save metrics
    with open(os.path.join(run_dir, f"{args.model}_metrics.pkl"), "wb") as fh:
        pickle.dump({"male": male, "rmsle": rmsle}, fh)

if __name__ == "__main__":
    main()