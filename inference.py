from utils.data_utils import load_snapshots
from models.full_model import ImpactModel
import torch, numpy as np

# ---------- load model -------------------------------------------------
snapshots = load_snapshots("data/raw/G_{}.pt", list(range(1995, 2025)))
device    = "cuda:0"
for s in snapshots: s.to(device)

metadata = snapshots[0].metadata()
in_dims  = dict(author = snapshots[0]['author'].x.size(-1),
                paper  = snapshots[0]['paper'].x_title_emb.size(-1),
                venue  = snapshots[0]['venue' ].x.size(-1))
model = ImpactModel(metadata, in_dims, hidden_dim=50).to(device)
model.load_state_dict(torch.load("runs/…/model.pt"))
model.eval()

# ---------- counter-factual query -------------------------------------
authors = ["A123456", "B987654", "C…", "D…", "E…"]      # raw IDs
topic_vec = torch.from_numpy(np.random.randn(768).astype('float32'))

pred = model.predict_team(authors, topic_vec, snapshots, current_year_idx=len(snapshots)-1)
print("expected yearly citations:", pred.tolist())