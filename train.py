import argparse, time
import torch
from torch.optim import Adam
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel
from rich.console import Console
import os
from utils.plotting import plot_pred_true_distributions_with_ci, plot_yearly_aggregates
import numpy as np

# (1) Function to store each evaluated model checkpoint
def save_evaluated_model_checkpoint(model, optimizer, epoch, current_male_values, current_rmsle_values, args, training_loss, run_dir, console): 
    """
    Saves a checkpoint of the model, optimizer, and metrics after an evaluation step.
    The filename includes the epoch and up to the first 5 MALE values (rounded).
    """
    if current_male_values is None:
        console.print("[yellow]Skipping saving evaluated model: MALE values are not available.[/yellow]")
        return

    male_components = []
    # Ensure current_male_values is a list. It should be after processing in main.
    num_male_values_to_include = min(5, len(current_male_values))

    for i in range(num_male_values_to_include):
        try:
            # Round to 4 decimal places for the filename
            male_components.append(f"male{i}_{round(float(current_male_values[i]), 4):.4f}")
        except (ValueError, TypeError) as e:
            console.print(f"[yellow]Warning: Could not process MALE value at index {i} for filename component: '{current_male_values[i]}'. Error: {e}[/yellow]")
            male_components.append(f"male{i}_error")

    if not male_components and current_male_values:
            male_str = "male_values_present_but_error_in_formatting"
    elif not male_components:
        male_str = "no_male_values"
    else:
        male_str = "_".join(male_components)
        
    # Construct a descriptive filename
    eval_model_filename = f"evaluated_model_epoch{epoch:03d}_{male_str}_{args.eval_mode}.pt"
    eval_model_path = os.path.join(run_dir, eval_model_filename)

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss, # Training loss at the time of this epoch's training step
            'eval_male': current_male_values,
            'eval_rmsle': current_rmsle_values,
            'args': args # Save training arguments for this specific model
        }, eval_model_path)
        console.print(f"[cyan]Evaluated model checkpoint saved: {eval_model_path}[/cyan]")
    except Exception as e:
        console.print(f"[red]Error saving evaluated model checkpoint: {e}[/red]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=2, type=int, required=True,
                        help="e.g. 1995 2004 inclusive, the yeas to train on")
    parser.add_argument("--test_years", nargs=2, type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150, help="Total number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=256)  # unused in v1
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=.5) # regularization parameter
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cold_start_prob", type=float, default=0.3,
                        help="probability that a training paper is treated as "
                             "venue/reference-free (cold-start calibration)")
    parser.add_argument("--eval_mode", choices=["paper", "team"], default="paper",
                        help="'paper' = original evaluation  |  'team' = counter-factual")

    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint file to load model and optimizer states for continuing training.")
    parser.add_argument("--training_off", type=int, default=0,
                    help="Path to a .pt checkpoint file to load model and optimizer states for continuing training.")
    # ['all features', 'drop toic']
    # choose one of the two
    parser.add_argument("--input_feature_model", choices=['all features', 'drop topic'], default='all features',
                        help="Input feature model to use. Choose one of the two: 'all features' or 'drop topic'.")
    parser.add_argument("--inference_time_author_dropping", type=str, default=False,
                        help="Whether to drop authors from the training set. Default is False.")
    parser.add_argument("--inference_time_num_author_dropping_k", type=int, default=0,
                        help="Number of authors to drop from the training set. Default is 0.")
    args = parser.parse_args()

    run_dir_suffix = f"_{args.eval_mode}"
    if args.load_checkpoint: # Add suffix if resuming to distinguish run directory
        run_dir_suffix += f"_resumedFrom_{os.path.splitext(os.path.basename(args.load_checkpoint))[0]}"
    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + run_dir_suffix)
    os.makedirs(run_dir, exist_ok=True)

    log_file_path = os.path.join(run_dir, "training_log.log")

    console = Console(record=True)


    try: # Wrap the main logic in try...finally to ensure log file is closed
        console.print(f"Training log will be saved to: {log_file_path}")
        console.print(f"Run directory: {run_dir}")
        console.print(f"Arguments: {args}")
        console.print(f"CUDA available: {torch.cuda.is_available()}")
        
        train_years = list(range(args.train_years[0], args.train_years[1] + 1))
        test_years = list(range(args.test_years[0], args.test_years[1] + 1))

        console.print(f"[bold]Train years:[/bold] {train_years}")
        console.print(f"[bold]Test years:[/bold]  {test_years}")
        console.print(f"[bold]Device:[/bold] {args.device}")
        
        snapshots = load_snapshots("data/yearly_snapshots_specter2/G_{}.pt", train_years + test_years)
        snapshots = [g.to(args.device) for g in snapshots]
        
        author_raw_ids = snapshots[-1]['author'].raw_ids          # list[str]
        AUT2IDX = {aid: i for i, aid in enumerate(author_raw_ids)}
        idx2aut = author_raw_ids                                  # same order

        metadata = snapshots[0].metadata()
        in_dims = {
            "author": snapshots[0]["author"].x.size(-1),
            "paper":  snapshots[0]["paper"].x_title_emb.size(-1),
            "venue":  snapshots[0]["venue"].x.size(-1),
        }
        model = ImpactModel(metadata,
                            in_dims,
                            hidden_dim=args.hidden_dim,
                            beta=args.beta,
                            cold_start_prob=args.cold_start_prob,
                            aut2idx=AUT2IDX,
                            idx2aut=idx2aut,
                            input_feature_model=args.input_feature_model,
                            args=args,
                            ).to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)

        start_epoch = 1
        loaded_checkpoint_args_info = "None"  

        if args.load_checkpoint:
            if os.path.exists(args.load_checkpoint):
                try:
                    console.print(f"[cyan]Attempting to load checkpoint from: {args.load_checkpoint}[/cyan]")
                    checkpoint = torch.load(args.load_checkpoint, map_location=args.device, weights_only=False)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    loaded_epoch = checkpoint.get('epoch', 0)
                    start_epoch = loaded_epoch + 1 
                    
                    if 'args' in checkpoint:
                        loaded_checkpoint_args_info = str(vars(checkpoint['args']))
                        console.print(f"[magenta]Arguments from loaded checkpoint (for reference): {loaded_checkpoint_args_info}[/magenta]")
                    
                    console.print(f"[green]Checkpoint loaded successfully. Model and optimizer states restored. "
                                  f"Resuming training from epoch {start_epoch}. Current total epochs set to {args.epochs}.[/green]")

                    if start_epoch > args.epochs:
                        console.print(f"[yellow]Warning: Loaded checkpoint from epoch {loaded_epoch}. "
                                      f"The new start epoch {start_epoch} is greater than the total configured epochs {args.epochs}. "
                                      f"Training will not run unless --epochs is increased beyond {loaded_epoch}.[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error loading checkpoint '{args.load_checkpoint}': {e}. Starting training from scratch (epoch 1).[/red]")
                    start_epoch = 1  
            else:
                console.print(f"[yellow]Checkpoint file not found: {args.load_checkpoint}. Starting training from scratch (epoch 1).[/yellow]")
                start_epoch = 1


        best_male_metric = float('inf')
        best_epoch_val = 0  
        best_model_path = ""

        
        current_args_path = os.path.join(run_dir, "current_run_args.txt")
        with open(current_args_path, 'w') as f:
            if hasattr(args, '__dict__'):
                for arg_name, arg_val in vars(args).items():
                    f.write(f"{arg_name}: {arg_val}\n")
            else:
                    f.write(str(args))
        console.print(f"Current run arguments saved to {current_args_path}")

        if args.load_checkpoint and loaded_checkpoint_args_info != "None":
            loaded_args_path = os.path.join(run_dir, "loaded_checkpoint_args_info.txt")
            with open(loaded_args_path, 'w') as f:
                f.write(loaded_checkpoint_args_info)
            console.print(f"Arguments info from loaded checkpoint saved to {loaded_args_path}")

        if start_epoch > args.epochs:
            console.print(f"[yellow]Training skipped: start_epoch ({start_epoch}) > total epochs ({args.epochs}).[/yellow]")
        else:
            console.print(f"Starting training from epoch {start_epoch} to {args.epochs}.")

        if args.training_off:
            console.print(f"[yellow]Training is turned off (training_off={args.training_off}). No training will be performed.[/yellow]")
            model.eval()
            console.print(f"Evaluating model using ckpt: {args.load_checkpoint} ")
            with torch.no_grad():
                current_male_values = None
                current_rmsle_values = None

                if args.eval_mode == "paper":
                    male, rmsle, mape = model.evaluate(
                        snapshots,
                        list(range(len(train_years), len(train_years)+len(test_years))),
                        start_year=train_years[0]
                    )
                else:  # counter-factual
                    male, rmsle, mape, y_true, y_predict = model.evaluate_team(
                        snapshots,
                        list(range(len(train_years), len(train_years)+len(test_years))),
                        start_year=train_years[0],
                        return_raw=True
                    )
                
                
                console.print(f"[green]Eval ({args.eval_mode})[/green] "
                                f"MALE {male}  RMSLE {rmsle} MAPE {mape}")
                
                plot_pred_true_distributions_with_ci(
                    y_true.numpy(),                   # expects numpy
                    y_predict.numpy(),
                    horizons=[f"Year {i}" for i in range(5)],
                    bins=100,
                    plot_type="hist",
                    save_path="./figs/ours_dist_all_years_ci_hist.png",
                    show=False)
                
                plot_yearly_aggregates(
                    y_true.numpy(),
                    y_predict.numpy(),
                    horizons=[f"Year {i}" for i in range(5)],
                    agg_fn=np.median,
                    show_iqr=True,
                    save_path="./figs/median_iqr.png",
                    show=False)

            return 0

        loss = None
        log = {}

        for epoch in range(start_epoch, args.epochs + 1):
            console.save_text(log_file_path, clear=False)
            model.train()
            optimizer.zero_grad()
            # for the forward pass, we need to pass the snapshots for the training years, not only the idx, but the specific year, so add the first specific year as one element
            loss, log = model(snapshots, list(range(len(train_years))), train_years[0])
            loss.backward()
            optimizer.step()

            log_items_str = "  ".join(f"{k}:{v:.4f}" for k, v in log.items())
            console.log(f"Epoch {epoch:03d}  Loss: {loss.item():.4f}  {log_items_str}")

            if epoch % 20 == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    current_male_values = None
                    current_rmsle_values = None

                    if args.eval_mode == "paper":
                        male, rmsle, mape = model.evaluate(
                            snapshots,
                            list(range(len(train_years), len(train_years)+len(test_years))),
                            start_year=train_years[0]
                        )
                    else:  # counter-factual
                        male, rmsle, mape = model.evaluate_team(
                            snapshots,
                            list(range(len(train_years), len(train_years)+len(test_years))),
                            start_year=train_years[0]
                        )
                    
                    current_male_values = male.tolist() if hasattr(male, 'tolist') else male
                    current_rmsle_values = rmsle.tolist() if hasattr(rmsle, 'tolist') else rmsle

                    if not isinstance(current_male_values, list): current_male_values = [current_male_values]
                    if not isinstance(current_rmsle_values, list): current_rmsle_values = [current_rmsle_values]
                    
                    console.print(f"[green]Eval Epoch {epoch:03d} ({args.eval_mode})[/green] "
                                  f"MALE {current_male_values}  RMSLE {current_rmsle_values} MAPE {mape}")

                    if loss is not None: 
                        save_evaluated_model_checkpoint(model, optimizer, epoch, current_male_values, current_rmsle_values, args, loss.item(), run_dir, console)

                    if current_male_values and len(current_male_values) > 0:
                        metric_to_track = float(current_male_values[0])  
                        
                        if metric_to_track < best_male_metric:
                            best_male_metric = metric_to_track
                            best_epoch_val = epoch
                            
                            if best_model_path and os.path.exists(best_model_path):
                                try:
                                    os.remove(best_model_path)
                                    console.print(f"[grey50]Removed old best model: {best_model_path}[/grey50]")
                                except OSError as e:
                                    console.print(f"[yellow]Could not remove old best model: {e}[/yellow]")
                                    
                            best_model_filename = f"best_model_epoch{epoch:03d}_male{best_male_metric:.4f}_{args.eval_mode}.pt"
                            best_model_path = os.path.join(run_dir, best_model_filename)
                            
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss.item() if loss is not None else None,
                                'eval_male': current_male_values,
                                'eval_rmsle': current_rmsle_values,
                                'args': args
                            }, best_model_path)
                            console.print(f"[blue]New best model saved: {best_model_path} "
                                          f"(MALE: {best_male_metric:.4f} at epoch {best_epoch_val})[/blue]")

        final_model_filename = f"final_model_epoch{args.epochs:03d}_{args.eval_mode}.pt" 
        actual_last_epoch = epoch if 'epoch' in locals() and start_epoch <= args.epochs else args.epochs

        final_model_path = os.path.join(run_dir, final_model_filename)
        
        if start_epoch <= args.epochs and loss is not None: 
            torch.save({
                'epoch': actual_last_epoch,  
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'final_train_loss_val': loss.item(),  
                'final_train_log': log,  
                'args': args
            }, final_model_path)
            console.print(f"Done. Final model for epoch {actual_last_epoch} saved to {final_model_path}")
        elif start_epoch <= args.epochs and loss is None: 
            console.print(f"[yellow]Training loop may have had issues; loss not defined. Final model not saved.[/yellow]")
        else: 
            console.print(f"No training performed in this run. Final model not saved. (start_epoch: {start_epoch}, args.epochs: {args.epochs})")

        if best_model_path:
            console.print(f"Best performing model (Epoch {best_epoch_val}) from this run retained at: {best_model_path} "
                          f"with MALE: {best_male_metric:.4f}")
        else:
            console.print("[yellow]No best model was saved based on MALE metric during evaluation steps for this run.[/yellow]")

    finally:
        console.save_text(log_file_path, clear=False)

if __name__ == "__main__":
    main()