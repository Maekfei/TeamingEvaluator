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
    β  controls the degree of temporal smoothing.
    """
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim=128,
        beta=.5, # regularization parameter (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
        horizons=(1, 2, 3, 4, 5), # [1, 2, 3, 4, 5] yearly citation counts
        meta_types: tuple[str, ...] = ("author", "venue", "paper"),
        cold_start_prob=0.0,
        aut2idx=None,
        idx2aut=None
    ):
        super().__init__()
        self.encoder = RGCNEncoder(metadata, in_dims, hidden_dim) # output: keys ('author', 'paper', 'venue'), values: embeddings. (num_nodes_of_that_type, hidden_dim)
        self.imputer = WeightedImputer(meta_types) # impute to update the new paper embedding.
        self.generator = ImpactRNN(hidden_dim)
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.cold_p = cold_start_prob
        self.aut2idx = aut2idx
        self.idx2aut = idx2aut

        self.history   = len(horizons)      # number of years fed to the GRU
        self.horizons = torch.tensor(horizons, dtype=torch.float32)
        self.eps = 1.0                    # +1 for zero-citation papers

    # -----------------------------------------------------------------
    def forward(self, snapshots, years_train, start_year):
        """
        Train forward pass iterating over years in years_train.
        snapshots : list[HeteroData] aligned with chronological order.
        years_train: iterable of indices (ints)
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
        train_idxs = set(years_train)
        l_time = []
        node_types_to_regularize = ['paper', 'author', 'venue']  # Add other node types if needed

        for ntype in node_types_to_regularize:
            l_time_ntype = []
            for t in years_train:            
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

        l_time = (torch.stack(l_time).mean()
                if l_time else torch.tensor(0.0, device=embeddings[0]['paper'].device))

        # -------- prediction loss over all new papers ----------------
        l_pred = []
        for t in years_train: # year t.
            data = snapshots[t]
            year_actual = start_year + t
            mask = (data['paper'].y_year == year_actual).to(device)
            if mask.sum() == 0:           # nothing to train / evaluate for that year
                continue

            paper_ids = mask.nonzero(as_tuple=False).view(-1)
            y_true    = data['paper'].y_citations[paper_ids].float()
            topic_all = embeddings[t]['paper'][paper_ids]
            N = y_true.size(0)

            # ----------------------------------------------------------
            # 3.1 cache neighbours of every paper in its publication yr
            # ----------------------------------------------------------
            neigh_cache = [
                self.imputer.collect_neighbours(data, int(pid), device)
                for pid in paper_ids
            ]

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
            for k in range(self.history): # k = 0 … 4
                if k == 0:
                    # step-0  : the current year, topic only  (no neighbours ⇒ no leakage)
                    seq_k = topic_all                             # [N,H]
                else:
                    yr = t - k                                        # t-1 … t- 4
                    if yr < 0:                                        # before data starts
                        seq_k = torch.zeros(N, self.hidden_dim, device=device)
                    else:
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
                                        ], dim=0)                                            # [N,H]
                seq_steps.append(seq_k) # seq_k is a tensor of shape [N,H], from year t back to year t -4, from now to 4 years ago
            seq_steps = seq_steps[::-1]
            V_p = torch.stack(seq_steps, dim=1)    # [N,5,H]

            # 3.3 generate citation distribution -----------------------
            eta, mu, sigma = self.generator(V_p) 
            y_hat_cum = self.generator.predict_cumulative(
                self.horizons.to(device), eta, mu, sigma
            )                                           # [N, L]

            # convert cumulative → yearly counts -----------------------
            y_hat = torch.cat(
                [y_hat_cum[:, :1], y_hat_cum[:, 1:] - y_hat_cum[:, :-1]],
                dim=1,
            )

            loss = ((torch.log1p(y_true + self.eps) -
                     torch.log1p(y_hat)) ** 2).mean()
            l_pred.append(loss)

        l_pred = torch.stack(l_pred).mean()
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

        males, rmsles = [], []
        for t in years_test:
            data      = snapshots[t]
            year_actual = start_year + t
            mask = (data['paper'].y_year == year_actual).to(device)
            if mask.sum() == 0:           # nothing to train / evaluate for that year
                continue

            paper_ids = mask.nonzero(as_tuple=False).view(-1)
            y_true    = data['paper'].y_citations[paper_ids].float()
            topic_all = embeddings[t]['paper'][paper_ids]
            N = y_true.size(0)

            neigh_cache = [
                self.imputer.collect_neighbours(data, int(pid), device)
                for pid in paper_ids
            ]

            # 3) build 5-step sequence  (k = 0 … 4)
            seq_steps = [] # below also need updates.
            for k in range(self.history):
                if k == 0:                             # publication day
                    seq_k = topic_all                  # [N,H]
                else:
                    yr = t - k
                    if yr < 0:                         # before data starts
                        seq_k = torch.zeros(N, self.hidden_dim, device=device)
                    else:
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
            V_p = torch.stack(seq_steps, dim=1)        # [N,5,H]

            # 4) predict citation distribution
            eta, mu, sigma = self.generator(V_p)
            y_hat_cum = self.generator.predict_cumulative(
                            horizons, eta, mu, sigma)              # [N,5]
            y_hat = torch.cat([y_hat_cum[:, :1],
                               y_hat_cum[:, 1:] - y_hat_cum[:, :-1]], 1)

            males .append(male_vec (y_true, y_hat))
            rmsles.append(rmsle_vec(y_true, y_hat))

        male  = torch.stack(males ).mean(0)    # [5]
        rmsle = torch.stack(rmsles).mean(0)
        return male.cpu(), rmsle.cpu()

    # -----------------------------------------------------------------
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

        author_ids        : list of raw author IDs (strings)
        topic_vec         : H-dim tensor of the paper topic
        snapshots         : list[HeteroData]  (all yearly graphs)
        current_year_idx  : index of the snapshot in which the paper occurs
        """
        device    = next(self.parameters()).device
        topic_vec = topic_vec.to(device)

        # ------------- helper to translate raw ids to row indices -----
        idx_cache: dict[int, torch.Tensor] = {}   # year -> LongTensor indices

        def _author_indices(year: int) -> torch.Tensor:
            """return LongTensor with valid row indices for <year> snapshot"""
            if year in idx_cache:
                return idx_cache[year]            # cached

            raw_ids = snapshots[year]['author'].raw_ids      # list[str]
            raw2row = {aid: i for i, aid in enumerate(raw_ids)}

            rows = [raw2row[a] for a in author_ids if a in raw2row]
            if len(rows) == 0:
                out = torch.empty(0, dtype=torch.long, device=device)
            else:
                out = torch.tensor(rows, dtype=torch.long, device=device)

            idx_cache[year] = out
            return out
        # --------------------------------------------------------------

        seq = []          # will hold 5 tensors (oldest → newest)
        for offset in range(self.history - 1, -1, -1):
            yr = current_year_idx - offset

            if offset == 0:                      # publication year
                auth_emb = torch.zeros(self.hidden_dim, device=device)

            elif yr >= 0:                        # one of the past 4 yrs
                ids = _author_indices(yr)
                if ids.numel() == 0:
                    auth_emb = torch.zeros(self.hidden_dim, device=device)
                else:
                    enc_dict = self.encoder(snapshots[yr])
                    auth_emb = enc_dict['author'][ids].mean(0)

            else:                                # before the first snapshot
                auth_emb = torch.zeros(self.hidden_dim, device=device)

            v_k = ( self.imputer.w['author'] * auth_emb +
                    self.imputer.w['self']   * topic_vec )
            seq.append(v_k)

        V_p  = torch.stack(seq, 0).unsqueeze(0)        # [1,5,H]
        eta, mu, sigma = self.generator(V_p)
        cum = self.generator.predict_cumulative(
                  self.horizons.to(device), eta, mu, sigma)    # [1,5]
        yearly = torch.cat([cum[:, :1], cum[:,1:] - cum[:,:-1]], 1)
        return yearly.squeeze(0)                               # [5]


    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate_team(self, snapshots, years_test, start_year, return_raw=False):
        """
        Evaluate in the counter-factual setting:
        use only authors + topic of each paper in test years.
        """
        device    = next(self.parameters()).device
        horizons  = self.horizons.to(device)

        encs = [self.encoder(g) for g in snapshots]       # testing years
        y_true_all, y_pred_all = [], []

        males, rmsles, mapes = [], [] ,[]
        for t in years_test:
            data = snapshots[t]
            year_actual = start_year + t
            mask = (data['paper'].y_year == year_actual).to(device)
            if mask.sum() == 0:           # nothing to train / evaluate for that year
                continue

            paper_ids = mask.nonzero(as_tuple=False).view(-1)
            y_true    = data['paper'].y_citations[paper_ids].float()
            topic_all = encs[t]['paper'][paper_ids]
            N = y_true.size(0)


            neigh_cache = [
                self.imputer.collect_neighbours(data, int(pid), device)
                for pid in paper_ids
            ]
            for d in neigh_cache:
                d.pop('venue',  None)
                d.pop('paper',  None)   # keep only authors

            preds = []
            for i, d in enumerate(neigh_cache):
                a_local = d.get('author', torch.empty(0, device=device, dtype=torch.long))
                if a_local.numel() == 0:
                    preds.append(torch.zeros(5, device=device))
                    continue
                au_raw = [ self.idx2aut[int(j)] for j in a_local ]
                p = self.predict_team(au_raw, topic_all[i], snapshots, t)
                preds.append(p)
            y_hat = torch.stack(preds, 0)

            males .append(male_vec (y_true, y_hat))
            rmsles.append(rmsle_vec(y_true, y_hat))
            mapes.append(mape_vec(y_true, y_hat))


            # store raw arrays -------------------------------------------------
            if return_raw:
                y_true_all.append(y_true.cpu())
                y_pred_all.append(y_hat.cpu())

        male  = torch.stack(males ).mean(0)
        rmsle = torch.stack(rmsles).mean(0)
        mape  = torch.stack(mapes ).mean(0)

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
    ) -> torch.Tensor:
        """
        Vectorised counter-factual inference for many teams.

        Parameters
        ----------
        teams            : list of (author_id_list, topic_vec) tuples
                           • author_id_list  –  raw author ids (strings)
                           • topic_vec       –  torch.FloatTensor [H]
        snapshots        : list of yearly HeteroData graphs
        current_year_idx : index of the snapshot that contains the topic
                           vectors (= "today")

        Returns
        -------
        Tensor  [N, 5]  – yearly citation predictions for every team
        """
        device  = next(self.parameters()).device
        H       = self.hidden_dim
        N       = len(teams)
        history = self.history

        # --------------------------------------------------------------
        # helper : translate  raw-author-ids  →  row indices *per year*
        # --------------------------------------------------------------
        raw2row_cache: dict[int, dict[str, int]] = {}   # year → {raw:row}

        def _rows_of(year: int, author_ids: list[str]) -> torch.LongTensor:
            if year not in raw2row_cache:
                raw_ids = snapshots[year]['author'].raw_ids
                raw2row_cache[year] = {aid: i for i, aid in enumerate(raw_ids)}
            mapping = raw2row_cache[year]
            rows = [mapping[a] for a in author_ids if a in mapping]
            if len(rows) == 0:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.tensor(rows, dtype=torch.long, device=device)

        # --------------------------------------------------------------
        # build the 5-step sequence for every team  →  [N, 5, H]
        # --------------------------------------------------------------
        seq_steps = []
        for offset in range(history - 1, -1, -1):          # 4 … 0
            yr = current_year_idx - offset

            # (i) author embedding
            author_emb = torch.zeros(N, H, device=device)
            if offset != 0 and yr >= 0:                     # past year
                enc_dict = self.encoder(snapshots[yr])
                for n, (au_list, _) in enumerate(teams):
                    row_idx = _rows_of(yr, au_list)
                    if row_idx.numel() > 0:
                        author_emb[n] = enc_dict['author'][row_idx].mean(0)

            # (ii) topic vector (k = 0 receives weight; past steps do not)
            topic_vec_batch = torch.stack([t[1] for t in teams]).to(device)

            v_k = ( self.imputer.w['author'] * author_emb +
                    self.imputer.w['self']   * topic_vec_batch )
            seq_steps.append(v_k)

        V_p = torch.stack(seq_steps, dim=1)                 # [N, 5, H]

        # --------------------------------------------------------------
        # Predict with ImpactRNN
        # --------------------------------------------------------------
        eta, mu, sigma = self.generator(V_p)
        horizons = self.horizons.to(device)
        cum = self.generator.predict_cumulative(horizons, eta, mu, sigma)
        yearly = torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1]], 1) # [N,5]
        return yearly