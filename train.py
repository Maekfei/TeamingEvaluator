import argparse, time
import torch
from torch.optim import Adam
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel
from rich.console import Console

console = Console()

# example
# python train.py \
#   --train_years 1995 1995 \
#   --test_years 1996 1996 \
#   --hidden_dim 32 --epochs 1 \
#   --cold_start_prob 0.5 \
#   --eval_mode team \
#   --device cuda:0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=2, type=int, required=True,
                        help="e.g. 1995 2004 inclusive, the yeas to train on")
    parser.add_argument("--test_years", nargs=2, type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)  # unused in v1
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=.5) # regularization parameter (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cold_start_prob", type=float, default=0.3,
                    help="probability that a training paper is treated as "
                         "venue/reference-free (cold-start calibration)") # 1.0 = all papers are cold-start, 0.0 = no papers are cold-start
    parser.add_argument("--eval_mode", choices=["paper", "team"], default="paper",
                    help="'paper' = original evaluation  |  'team' = counter-factual") # when team, the input is a list of authors plus a topic; when paper, the input is a list of authors plus a paper topic, the paper's venue,  the paper's reference paper.
    args = parser.parse_args()

    train_years = list(range(args.train_years[0], args.train_years[1] + 1))
    test_years = list(range(args.test_years[0], args.test_years[1] + 1))

    console.print(f"[bold]Train years:[/bold] {train_years}")
    console.print(f"[bold]Test years:[/bold]  {test_years}")

    # read all snapshots (years) into a list
    snapshots = load_snapshots("data/raw/G_{}.pt", train_years + test_years)
    snapshots = [g.to(args.device) for g in snapshots]
    
    # ------------------------------------------------------------------ #
    #  load cached mapping tables without pulling the big embedding file #
    # ------------------------------------------------------------------ #
    import pickle, os
    META_CACHE = "data/yearly_snapshots/mappings.pkl"

    with open(META_CACHE, "rb") as fh:
        _, AUT2IDX, _ = pickle.load(fh)           # we only need authors

    idx2aut = [None] * len(AUT2IDX)
    for a, i in AUT2IDX.items():
        idx2aut[i] = a


    metadata = snapshots[0].metadata() #  returns a tuple containing information about the graph's structure, specifically the node types and the edge types (including their source and target node types
    in_dims = {
        "author": snapshots[0]["author"].x.size(-1),
        "paper":  snapshots[0]["paper"].x_title_emb.size(-1),
        "venue":  snapshots[0]["venue"].x.size(-1),
    }

    model = ImpactModel(metadata, # metadata of the graph
                        in_dims, #
                        hidden_dim=args.hidden_dim, # hidden_dim, the larger the more complex the model
                        beta=args.beta, # beta, a regularization parameter for the model (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
                        cold_start_prob=args.cold_start_prob,
                        aut2idx=AUT2IDX,
                        idx2aut=idx2aut).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr) # adapts the learning rates for each parameter individually

    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Initialize to track the best model based on MALE (lower is better)
    best_male_metric = float('inf')
    best_epoch = 0
    best_model_path = "" # To store the path of the best model

    console.print(f"Run directory: {run_dir}")
    # Save args to the run directory for reproducibility
    if hasattr(args, '__dict__'):
        with open(os.path.join(run_dir, "args.txt"), 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
    else:
        console.print("[yellow]Warning: 'args' object does not have __dict__, cannot save arguments easily.[/yellow]")


    # Training loop ---------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss, log = model(snapshots, list(range(len(train_years)))) # training on the first len(train_years) snapshots
        loss.backward() # compute gradients
        optimizer.step()

        # Constructing the log message
        log_items_str = "  ".join(f"{k}:{v:.4f}" for k, v in log.items())
        console.log(f"Epoch {epoch:03d}  Loss: {loss.item():.4f}  {log_items_str}") # Added loss.item() for clarity

        if epoch % 5 == 0 or epoch == args.epochs: # Using args.eval_every if available, else default
            model.eval()
            with torch.no_grad():
                current_male_values = None # Initialize
                current_rmsle_values = None

                if args.eval_mode == "paper":
                    male, rmsle = model.evaluate(
                        snapshots,
                        list(range(len(train_years), len(train_years)+len(test_years)))
                    ) # use test set for evaluation
                else:   # counter-factual
                    male, rmsle = model.evaluate_team(
                        snapshots,
                        list(range(len(train_years), len(train_years)+len(test_years)))
                    ) # use test set for evaluation
                
                current_male_values = male.tolist()
                current_rmsle_values = rmsle.tolist()

                console.print(f"[green]Eval Epoch {epoch:03d} ({args.eval_mode})[/green] "
                            f"MALE {current_male_values}  RMSLE {current_rmsle_values}")

                # --- Improvement: Save best model based on the first MALE value ---
                # Assuming lower MALE is better and we use the first MALE value.
                # Adjust if your MALE is a single value or you want to track another element/metric.
                if current_male_values: # Ensure MALE values are available
                    # Let's assume the first MALE value is the primary one to track
                    # If male is already a scalar, male.tolist() might not be needed or might error.
                    # Adjust accordingly. For this example, assuming male is a tensor/list.
                    metric_to_track = current_male_values[0] if isinstance(current_male_values, list) and current_male_values else float('inf')

                    if metric_to_track < best_male_metric:
                        best_male_metric = metric_to_track
                        best_epoch = epoch
                        if best_model_path and os.path.exists(best_model_path):
                            try:
                                os.remove(best_model_path)
                            except OSError as e:
                                console.print(f"[yellow]Could not remove old best model: {e}[/yellow]")
                                
                        best_model_filename = f"best_model_epoch{epoch:03d}_male{best_male_metric:.4f}_{args.eval_mode}.pt"
                        best_model_path = os.path.join(run_dir, best_model_filename)
                        
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            'eval_male': current_male_values,
                            'eval_rmsle': current_rmsle_values,
                            'args': args # Save training arguments
                        }, best_model_path)
                        console.print(f"[blue]New best model saved: {best_model_path} "
                                    f"(MALE: {best_male_metric:.4f})[/blue]")
                # --- End of improvement ---

    # Save the final model with a descriptive name and more info
    final_model_filename = f"final_model_epoch{args.epochs:03d}_{args.eval_mode}.pt"
    final_model_path = os.path.join(run_dir, final_model_filename)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss_val': loss.item(), # Last training loss value
        'final_train_log': log, # Last logged training metrics dict
        'args': args
    }, final_model_path)

    console.print(f"Done. Final model saved to {final_model_path}")
    if best_model_path:
        console.print(f"Best performing model (Epoch {best_epoch}) retained at: {best_model_path} "
                    f"with MALE: {best_male_metric:.4f}")
    else:
        console.print("[yellow]No best model was saved based on MALE metric during evaluation steps.[/yellow]")



if __name__ == "__main__":
    main()