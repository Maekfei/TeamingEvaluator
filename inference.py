import torch

from utils.data_utils import load_snapshots
from models.full_model import ImpactModel

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------
# 1) load snapshots up to the year you want to condition on
# ---------------------------------------------------------------------
this_year = 2024                                        # example
snapshots = load_snapshots("data/yearly_snapshots_specter2/G_{}.pt",
                           list(range(2020, this_year+1)))
snapshots = [g.to(device) for g in snapshots]

# ---------------------------------------------------------------------
# 2) load the trained model checkpoint
# ---------------------------------------------------------------------
ckpt = torch.load("/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/best_ckpt/best_model_epoch120_male0.4242_team.pt", map_location=device, weights_only=False)
model_args   = ckpt['args']
metadata     = snapshots[0].metadata()
in_dims = {
    "author": snapshots[0]["author"].x.size(-1),
    "paper":  snapshots[0]["paper"].x_title_emb.size(-1),
    "venue":  snapshots[0]["venue"].x.size(-1),
}
model = ImpactModel(metadata,
                    in_dims,
                    hidden_dim=model_args.hidden_dim,
                    beta=model_args.beta,
                    cold_start_prob=0.0,
                    aut2idx={}, idx2aut=[])              # not used here
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

# ---------------------------------------------------------------------
# 3) prepare the list of candidate teams
# ---------------------------------------------------------------------
H = model.hidden_dim
teams = []                                              # list of (authors, topic)
for _ in range(2):
    author_ids = ['100000093', '6052561']               # <- replace
    topic_vec  = torch.randn(H)                         # <- replace with SPECTER2
    teams.append((author_ids, topic_vec))

# ---------------------------------------------------------------------
# 4) run inference
# ---------------------------------------------------------------------
pred = model.predict_teams(teams,
                           snapshots=snapshots,
                           current_year_idx=len(snapshots)-1)   # 2015
print(pred.shape)      # torch.Size([6000, 5])
print(pred[:3])        # first three teams – yearly counts for years 1…5
print(pred.mean(dim=1))