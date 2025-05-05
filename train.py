import argparse, os, time
import torch
from torch.optim import Adam
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=2, type=int, required=True,
                        help="e.g. 1995 2004 inclusive")
    parser.add_argument("--test_years", nargs=2, type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)  # unused in v1
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=.5) # regularization parameter (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_years = list(range(args.train_years[0], args.train_years[1] + 1))
    test_years = list(range(args.test_years[0], args.test_years[1] + 1))

    console.print(f"[bold]Train years:[/bold] {train_years}")
    console.print(f"[bold]Test years:[/bold]  {test_years}")

    snapshots = load_snapshots("data/raw/G_{}.pt", train_years + test_years)
    # snapshots = [g.to(args.device) for g in snapshots]
    for g in snapshots:            # keep graphs on CPU
        g.pin_memory()             # fast CPU→GPU copy
    
    metadata = snapshots[0].metadata() #  returns a tuple containing information about the graph's structure, specifically the node types and the edge types (including their source and target node types
    in_dims = {
        "author": snapshots[0]["author"].x.size(-1),
        "paper":  snapshots[0]["paper"].x_title_emb.size(-1),
        "venue":  snapshots[0]["venue"].x.size(-1),
    }

    model = ImpactModel(metadata, # metadata of the graph
                        in_dims, #
                        hidden_dim=args.hidden_dim, # hidden_dim, the larger the more complex the model
                        beta=args.beta).to(args.device) # beta, a regularization parameter for the model (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
    optimizer = Adam(model.parameters(), lr=args.lr) # adapts the learning rates for each parameter individually

    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Training loop ---------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss, log = model(snapshots, list(range(len(train_years))))
        loss.backward() # compute gradients
        optimizer.step()

        console.log(f"Epoch {epoch:03d}  " +
                    "  ".join(f"{k}:{v:.4f}" for k, v in log.items()))

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                male, rmsle = model.evaluate(
                    snapshots,
                    list(range(len(train_years), len(train_years) + len(test_years)))
                )
            console.print(f"[green]Eval[/green] MALE {male.tolist()}  RMSLE {rmsle.tolist()}")

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    console.print(f"Done. Model saved to {run_dir}/model.pt")


if __name__ == "__main__":
    main()