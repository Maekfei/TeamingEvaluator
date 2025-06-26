import torch
import torch.nn as nn
from .rgcn_encoder import RGCNEncoder
from .imputer import WeightedImputer
from .impact_rnn import ImpactRNN
from utils.metrics import rmsle_vec, male_vec, mape_vec


class ImpactModel(nn.Module):
    """
    Puts all three modules together and computes joint loss:
        J = L_pred  +  β Σ_t L_time(t,t+1)
    where L_pred is the prediction loss and L_time(t,t+1) is the temporal
    smoothing regulariser.
    β  controls the degree of temporal smoothing. # Beta is 0 in current implementation.
    """
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim=32,
        beta=0, # regularization parameter (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
        horizons=(1, 2, 3, 4, 5), # [1, 2, 3, 4, 5] yearly citation counts
        meta_types: tuple[str, ...] = ("author", "venue", "paper"),
        cold_start_prob=0.0,
        aut2idx=None,
        idx2aut=None,
        input_feature_model=None,
        args=None,
    ):
        super().__init__()
        self.encoder = RGCNEncoder(metadata, in_dims, hidden_dim) # output: keys ('author', 'paper', 'venue'), values: embeddings. (num_nodes_of_that_type, hidden_dim)
        self.imputer = WeightedImputer(meta_types, hidden_dim) # impute to update the new paper embedding.
        self.generator = ImpactRNN(hidden_dim, rnn_layers=1)
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.cold_p = cold_start_prob
        self.aut2idx = aut2idx
        self.args = args
        self.idx2aut = idx2aut
        self.input_feature_model = input_feature_model
        self.history   = len(horizons)      # number of years fed to the GRU
        self.horizons = torch.tensor(horizons, dtype=torch.float32)
        self.eps = 1.0                    # +1 for zero-citation papers

    # -----------------------------------------------------------------
    def forward(self, snapshots, years_train, start_year):
        """
        Train forward pass iterating over years in years_train.
        snapshots : list[HeteroData] aligned with chronological order.
        years_train: iterable of indices (ints), starting from 5, as we are using 5 years before the training set for the imputer.
        start_year : int, first year in the training set.
        Returns:
            loss  (scalar)
            log   (dict for printing)
        """
        device = next(self.parameters()).device
        self.horizons = self.horizons.to(device, non_blocking=True)


        # 1) encode every snapshot once, paper node is pretrained.
        embeddings = []
        for data in snapshots: # need multiple years for the imputer
            embeddings.append(self.encoder(data))
        # -------- temporal smoothing regulariser -------------------- # make sure the same papers, authors, venues are not too different in the two consecutive years
        train_idxs = set([0, 1, 2, 3, 4] + years_train) # 0, 1, 2, 3, 4 are the 5 years before the training set, we need to regularize them.
        # we are not using this regularization for now, as it is not very helpful.
        l_time = []
        node_types_to_regularize = ['paper', 'author', 'venue']  # Add other node types if needed
        for ntype in node_types_to_regularize:
            l_time_ntype = []
            for t in [0, 1, 2, 3, 4] + years_train:            
                if (t - 1) not in train_idxs:   
                    continue
                emb_t = embeddings[t]    [ntype]
                emb_tp1 = embeddings[t-1][ntype].detach()  # ← detach: no grad to past, so there will be no data leakage.

                common = torch.arange(
                    min(emb_t.size(0), emb_tp1.size(0)),
                    device=emb_t.device,
                )
                diff = emb_t[common] - emb_tp1[common]
                l_time_ntype.append((diff ** 2).sum(1).mean())

            if l_time_ntype:
                l_time.append(torch.stack(l_time_ntype).mean())

        # Handle l_time properly
        if l_time:  # If we have any temporal losses
            l_time = torch.stack(l_time).mean()
        else:
            l_time = torch.tensor(0.0, device=embeddings[0]['paper'].device)

        # -------- prediction loss over all new papers ----------------
        l_pred = []
        n_samples = []
        for t in years_train: # year t. start from 5, as we are using 5 years before the training set for the imputer.
            data = snapshots[t]
            year_actual = start_year + t - 5 # 5 is the first year of the training data
            mask = (data['paper'].y_year == year_actual).to(device) # only use current year's papers for training
            if mask.sum() == 0:           # nothing to train / evaluate for that year
                continue

            paper_ids = mask.nonzero(as_tuple=False).view(-1) # get the paper ids of the current year's papers
            neigh_cache = [
                self.imputer.collect_neighbours(data, int(pid), device)
                for pid in paper_ids
            ] # get the papers' authors, references, and venue.
            
            y_true    = data['paper'].y_citations[paper_ids].float()
            if self.input_feature_model == 'drop topic': # here can be problematic by setting the topic vector to zero.
                topic_all = torch.zeros(y_true.size(0), self.hidden_dim, device=device)
            elif self.input_feature_model == 'drop authors':
                for nc in neigh_cache:
                    nc.pop('author', None)
                topic_all = embeddings[t]['paper'][paper_ids]
            else:
                topic_all = embeddings[t]['paper'][paper_ids]

            N = y_true.size(0) # number of (core) papers in the current year

            # ----------------------------------------------------------
            # 3.1 cache neighbours of every paper in its publication yr
            # ----------------------------------------------------------


            # -------- cold-start augmentation --------------------------
            if self.cold_p > 0.0:
                for nc in neigh_cache:                          # nc is a dict
                    if torch.rand(1).item() < self.cold_p:
                        nc.pop('venue',  None)                  # drop venue
                        nc.pop('paper',  None)                  # drop refs
            # ----------------------------------------------------------------

            # ----------------------------------------------------------
            # 3.2 build sequence  [t-1, …, t-history]
            # ----------------------------------------------------------

            seq_steps = []
            for k in range(self.history + 1): # k = 0 … 5
                yr = t - k                                        # t … t-5
                seq_k = torch.stack([
                                    self.imputer(
                                        None,                        # paper_id not needed
                                        yr,
                                        snapshots,
                                        embeddings,
                                        predefined_neigh = neigh_cache[i],
                                        topic_vec        = topic_all[i]
                                    )
                                    for i in range(N)
                                ], dim=0)                                            # [N,H], each paper has a embedding of size H.
                seq_steps.append(seq_k) # seq_k is a tensor of shape [N,H], from year t back to year t -5, from now to 5 years ago
            seq_steps = seq_steps[::-1] # reverse the sequence, in a chronological order.
            V_p = torch.stack(seq_steps, dim=1) # [N,6,H]
            # 3.3 generate citation distribution -----------------------
            eta, mu, sigma = self.generator(V_p) 
            y_hat_cum = self.generator.predict_cumulative(
                self.horizons.to(device), eta, mu, sigma
            )                                           # [N, L]

            # convert cumulative → yearly counts; standard time series prediction -----------------------
            y_hat = torch.cat(
                [y_hat_cum[:, :1], y_hat_cum[:, 1:] - y_hat_cum[:, :-1]],
                dim=1,
            )

            # Debug prints for loss calculation
            if torch.isnan(y_true).any() or torch.isnan(y_hat).any():
                print("NaN detected in loss calculation:")
                print(f"y_true stats: min={y_true.min().item():.3f}, max={y_true.max().item():.3f}, mean={y_true.mean().item():.3f}")
                print(f"y_hat stats: min={y_hat.min().item():.3f}, max={y_hat.max().item():.3f}, mean={y_hat.mean().item():.3f}")
                print(f"log1p(y_true) stats: min={torch.log1p(y_true).min().item():.3f}, max={torch.log1p(y_true).max().item():.3f}")
                print(f"log1p(y_hat) stats: min={torch.log1p(y_hat).min().item():.3f}, max={torch.log1p(y_hat).max().item():.3f}")

            # Calculate year-wise losses
            year_losses = (torch.log1p(y_true) - torch.log1p(y_hat)) ** 2  # [N, 5]
            
            # Calculate mean loss per year
            mean_loss_per_year = year_losses.mean(dim=0)  # [5]
            
            # Calculate how much each year's loss deviates from the mean
            year_deviation = torch.abs(year_losses - mean_loss_per_year)  # [N, 5]
            
            # Add penalty for large deviations from mean loss
            deviation_penalty = year_deviation.mean()
            
            # Combine the main loss with the deviation penalty
            loss = year_losses.mean() + 0.1 * deviation_penalty
            
            if torch.isnan(loss):
                print("NaN detected in final loss:")
                print(f"loss value: {loss.item()}")
                print(f"y_true shape: {y_true.shape}, y_hat shape: {y_hat.shape}")
            
            l_pred.append(loss * y_true.size(0))  # weight by number of samples
            n_samples.append(y_true.size(0))

        if n_samples:
            l_pred = torch.stack(l_pred).sum() / sum(n_samples)
        else:
            l_pred = torch.tensor(0.0, device=device)

        loss   = l_pred + self.beta * l_time

        log = dict(L_pred=l_pred.item(),
                   L_time=l_time.item(),
                   Loss  =loss.item())
        return loss, log

    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, snapshots, years_test, start_year):
        """
        Standard evaluation (real papers, full neighbourhood).
        Uses the *same* sequence construction as in training – one true
        topic vector at k=0 and one imputed vector for every past year.
        """
        device = next(self.parameters()).device
        horizons = self.horizons.to(device)

        # 1) encode all snapshots once
        embeddings = [self.encoder(g) for g in snapshots]

        males, rmsles, mapes = [], [] ,[]
        for t in years_test:
            data      = snapshots[t]
            year_actual = start_year + t - 5 # 5 is the first year of the training data
            mask = (data['paper'].y_year == year_actual).to(device)
            if mask.sum() == 0:           # nothing to train / evaluate for that year
                continue

            paper_ids = mask.nonzero(as_tuple=False).view(-1)
            y_true    = data['paper'].y_citations[paper_ids].float()
            if self.input_feature_model == 'drop topic':
                topic_all = torch.zeros(y_true.size(0), self.hidden_dim, device=device)
            else:
                topic_all = embeddings[t]['paper'][paper_ids]
            N = y_true.size(0)

            neigh_cache = [
                self.imputer.collect_neighbours(data, int(pid), device)
                for pid in paper_ids
            ]

            # 3) build 5-step sequence  (k = 0 … 4)
            seq_steps = [] # below also need updates.
            for k in range(self.history + 1):
                yr = t - k
                seq_k = torch.stack([
                                            self.imputer(
                                                None,                        # paper_id not needed
                                                yr,
                                                snapshots,
                                                embeddings,
                                                predefined_neigh = neigh_cache[i],
                                                topic_vec        = topic_all[i]
                                            )
                                            for i in range(N)
                                        ], dim=0)                               # [N,H]
                seq_steps.append(seq_k)
            seq_steps = seq_steps[::-1]
            V_p = torch.stack(seq_steps, dim=1)        # [N,6,H]

            # 4) predict citation distribution
            eta, mu, sigma = self.generator(V_p)
            y_hat_cum = self.generator.predict_cumulative(
                            horizons, eta, mu, sigma)              # [N,6]
            y_hat = torch.cat([y_hat_cum[:, :1],
                               y_hat_cum[:, 1:] - y_hat_cum[:, :-1]], 1)

            males .append(male_vec (y_true, y_hat))
            rmsles.append(rmsle_vec(y_true, y_hat))
            mapes.append(mape_vec(y_true, y_hat))
            

        male  = torch.stack(males ).mean(0)    # [5]
        rmsle = torch.stack(rmsles).mean(0)
        mape  = torch.stack(mapes ).mean(0)
        return male.cpu(), rmsle.cpu(), mape.cpu()

    # -----------------------------------------------------------------
    @torch.no_grad()
    def predict_team(
        self,
        author_ids: list[str],
        topic_vec: torch.Tensor,
        snapshots: list,
        current_year_idx: int,
    ):
        """
        Counter-factual prediction that is conditioned on the topic vector
        plus the embeddings of the *same* authors in the previous years.
        """
        device = next(self.parameters()).device
        if self.input_feature_model == 'drop topic':
            topic_vec = torch.zeros(self.hidden_dim, device=device)
        else:
            topic_vec = topic_vec.to(device)

        # --- author selection logic for inference-time author dropping ---
        mode = getattr(self.args, 'inference_time_author_dropping', None)
        if mode is not None:
            if mode == 'drop_first' and len(author_ids) >= 1:
                author_ids = author_ids[1:]
            elif mode == 'drop_last' and len(author_ids) >= 1:
                author_ids = author_ids[:-1]
            elif mode == 'drop_first_and_last' and len(author_ids) >= 2:
                author_ids = author_ids[1:-1]
            elif mode == 'keep_first' and len(author_ids) >= 1:
                author_ids = [author_ids[0]]
            elif mode == 'keep_last' and len(author_ids) >= 1:
                author_ids = [author_ids[-1]]
            elif mode == 'no author':
                author_ids = []

        # Pre-encode all snapshots once for efficiency
        embeddings = [self.encoder(g) for g in snapshots]

        # Build sequence using the same approach as forward method
        seq = []
        for offset in range(self.history, -1, -1):  # 5, 4, 3, 2, 1, 0
            yr = current_year_idx - offset
            
            # Get author indices for this year
            author_indices = self._get_author_indices_for_year(author_ids, yr, snapshots)
            
            # Use imputer to get the embedding for this year
            v_k = self.imputer(
                None,  # paper_id not needed
                yr,
                snapshots,
                embeddings,
                predefined_neigh={'author': author_indices},
                topic_vec=topic_vec
            )
            seq.append(v_k)

        V_p = torch.stack(seq, 0).unsqueeze(0)  # [1, 6, H]
        eta, mu, sigma = self.generator(V_p)
        cum = self.generator.predict_cumulative(
            self.horizons.to(device), eta, mu, sigma
        )
        yearly = torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1]], 1)
        return yearly.squeeze(0)  # [6]

    @torch.no_grad()
    def evaluate_team(self, snapshots, years_test, start_year, return_raw=False, author_drop_fn=None):
        """
        Evaluate in the counter-factual setting:
        use only authors + topic of each paper in test years.
        including inference-time author dropping.
        author_drop_fn: optional function(list[str]) -> list[str], to drop authors for ablation
        """
        device = next(self.parameters()).device
        horizons = self.horizons.to(device)

        # Pre-encode all snapshots once
        print("Pre-encoding all snapshots...")
        encs = [self.encoder(g) for g in snapshots]
        print("Done encoding snapshots")

        y_true_all, y_pred_all = [], []
        males, rmsles, mapes = [], [], []

        for t in years_test:
            data = snapshots[t]
            year_actual = start_year + t - 5  # 5 is the first year of the training data
            mask = (data['paper'].y_year == year_actual).to(device)
            if mask.sum() == 0:  # nothing to train / evaluate for that year
                continue

            paper_ids = mask.nonzero(as_tuple=False).view(-1)
            y_true = data['paper'].y_citations[paper_ids].float()
            if self.input_feature_model == 'drop topic':
                topic_all = torch.zeros(y_true.size(0), self.hidden_dim, device=device)
            else:
                topic_all = encs[t]['paper'][paper_ids]
            N = y_true.size(0)

            # Get all author indices for all papers at once
            neigh_cache = [
                self.imputer.collect_neighbours(data, int(pid), device)
                for pid in paper_ids
            ]
            for d in neigh_cache:
                d.pop('venue', None)
                d.pop('paper', None)  # keep only authors

            # Process papers in batches using predict_teams
            batch_size = 100  # Process 100 papers at a time
            preds = []
            for i in range(0, N, batch_size):
                end_idx = min(i + batch_size, N)
                batch_teams = []
                
                # Prepare batch of (author_ids, topic_vec) pairs
                for j in range(i, end_idx):
                    d = neigh_cache[j]
                    a_local = d.get('author', torch.empty(0, device=device, dtype=torch.long))
                    if a_local.numel() == 0:
                        # For papers with no authors, use empty list and zero topic
                        batch_teams.append(([], topic_all[j]))
                    else:
                        au_raw = [self.idx2aut[int(k)] for k in a_local]
                        # Apply ablation function if provided
                        if author_drop_fn is not None: # 
                            au_raw = author_drop_fn(au_raw)
                        batch_teams.append((au_raw, topic_all[j]))

                # Process batch
                batch_preds = self.predict_teams(batch_teams, snapshots, t, author_drop_fn=None)  # already applied
                preds.append(batch_preds)

            y_hat = torch.cat(preds, 0)

            males.append(male_vec(y_true, y_hat))
            rmsles.append(rmsle_vec(y_true, y_hat))
            mapes.append(mape_vec(y_true, y_hat))

            if return_raw:
                y_true_all.append(y_true.cpu())
                y_pred_all.append(y_hat.cpu())

        male = torch.stack(males).mean(0)
        rmsle = torch.stack(rmsles).mean(0)
        mape = torch.stack(mapes).mean(0)

        if return_raw:
            y_true_all = torch.cat(y_true_all, 0)
            y_pred_all = torch.cat(y_pred_all, 0)
            return male.cpu(), rmsle.cpu(), mape.cpu(), y_true_all, y_pred_all
        else:
            return male.cpu(), rmsle.cpu(), mape.cpu()

    # -----------------------------------------------------------------
    @torch.no_grad()
    def predict_teams(
        self,
        teams: list[tuple[list[str], torch.Tensor]],
        snapshots: list,
        current_year_idx: int,
        author_drop_fn=None,
    ) -> torch.Tensor:
        """
        Vectorised counter-factual inference for many teams.
        author_drop_fn: optional function(list[str]) -> list[str], to drop authors for ablation
        """
        device  = next(self.parameters()).device
        H       = self.hidden_dim
        N       = len(teams)
        history = self.history

        # Pre-encode all snapshots once for efficiency
        embeddings = [self.encoder(g) for g in snapshots]

        # --------------------------------------------------------------
        # Build sequence using the same approach as forward method
        # --------------------------------------------------------------
        seq_steps = []
        for k in range(history + 1):          # 0, 1, 2, 3, 4, 5
            yr = current_year_idx - k
            
            # Build sequence step for all teams at once
            seq_k = torch.stack([
                self.imputer(
                    None,  # paper_id not needed
                    yr,
                    snapshots,
                    embeddings,
                    predefined_neigh={'author': self._get_author_indices_for_year(teams[i][0], yr, snapshots)},
                    topic_vec=teams[i][1]
                )
                for i in range(N)
            ], dim=0)  # [N, H]
            
            seq_steps.append(seq_k)

        seq_steps = seq_steps[::-1]  # reverse to get chronological order
        V_p = torch.stack(seq_steps, dim=1)  # [N, 6, H]

        # --------------------------------------------------------------
        # Predict with ImpactRNN
        # --------------------------------------------------------------
        eta, mu, sigma = self.generator(V_p)
        horizons = self.horizons.to(device)
        cum = self.generator.predict_cumulative(horizons, eta, mu, sigma)
        yearly = torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1]], 1)  # [N, 5]
        return yearly

    def _get_author_indices_for_year(self, author_ids: list[str], year: int, snapshots: list) -> torch.Tensor:
        """
        Helper method to get author indices for a specific year.
        Returns empty tensor if authors don't exist in that year.
        """
        device = next(self.parameters()).device
        
        if not author_ids:
            return torch.empty(0, dtype=torch.long, device=device)
        
        # Get raw author IDs for the specified year
        raw_ids = snapshots[year]['author'].raw_ids
        raw2row = {aid: i for i, aid in enumerate(raw_ids)}
        
        # Find which authors exist in this year
        rows = [raw2row[a] for a in author_ids if a in raw2row]
        
        if not rows:
            return torch.empty(0, dtype=torch.long, device=device)
        
        return torch.tensor(rows, dtype=torch.long, device=device)