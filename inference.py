import torch
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel

# ------------------------------------------------------------------
# 0) device
# ------------------------------------------------------------------
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------
# 1) load snapshots up to the year you want to condition on
# ------------------------------------------------------------------
this_year  = 2024                           # the publication year
years      = list(range(1993, this_year))   # strictly < this_year
snapshots  = load_snapshots(
                "data/yearly_snapshots_specter2/G_{}.pt",
                years
             )
snapshots  = [g.to(device) for g in snapshots]

# ------------------------------------------------------------------
# 2) load the trained model checkpoint
# ------------------------------------------------------------------
ckpt_path = (
   "/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/"
   "best_ckpt/best_model_epoch120_male0.4242_team.pt"
)
ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
model_args  = ckpt["args"]
metadata    = snapshots[0].metadata()
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
                    aut2idx={}, idx2aut=[])          # not used here
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(device)

# ------------------------------------------------------------------
# 3) prepare the list of candidate teams
# ------------------------------------------------------------------
H      = model.hidden_dim                       # e.g. 128
teams  = []                                     # list of (authors, topic)

def project_topic(vec: torch.Tensor) -> torch.Tensor:
    """
    Projects a raw SPECTER(2) vector (768-d) to the model’s hidden
    dimension, *exactly* the same way the model does for paper nodes.
    """
    if vec.dim() == 1:           # (D,)  →  (1, D)   for Linear(...)
        vec = vec.unsqueeze(0)

    if vec.size(-1) == H:
        return vec.squeeze(0)                   # already projected

    if "paper" in model.encoder.lins:
        with torch.no_grad():
            vec = model.encoder.lins["paper"](vec)
    else:                                       # very unlikely
        if not hasattr(model, "_topic_proj"):
            model._topic_proj = torch.nn.Linear(
                vec.size(-1), H, bias=False
            ).to(device)
        with torch.no_grad():
            vec = model._topic_proj(vec)

    return vec.squeeze(0)                       # (H,)

# ------------------------------------------------------------------
#  Example: build two teams
# ------------------------------------------------------------------
for _ in range(2):
    author_ids = ['100000093', '6052561']       # ← replace
    raw_topic  = torch.randn(768)               # ← replace with SPECTER2
    topic_vec  = project_topic(raw_topic.to(device))
    teams.append((author_ids, topic_vec))

# ------------------------------------------------------------------
# 4) run inference
# ------------------------------------------------------------------
pred = model.predict_teams(
            teams,
            snapshots=snapshots,
            current_year_idx=len(snapshots) - 1   # index of 2023
       )

print(pred.shape)           # torch.Size([len(teams), 5])
print(pred[:3])             # yearly counts for first teams
print(pred.sum(dim=1))      # total 5-year impact per team