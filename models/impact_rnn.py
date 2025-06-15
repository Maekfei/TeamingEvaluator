import torch
import torch.nn as nn
import math


class ImpactRNN(nn.Module):
    """
    GRU encodes the imputed pre-publication sequence.
    Three identical MLP heads (20 → 8 → 1) output   η, μ, σ   per paper.
    """
    def __init__(self, hidden_dim: int, rnn_layers: int = 1):
        super().__init__()

        # ------------------------------------------------------------------
        # 1) temporal encoder
        # ------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # 2) helper to create a 3-layer MLP  (H → 20 → 8 → 1)
        # ------------------------------------------------------------------
        def _make_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, 20),
                nn.ReLU(inplace=True),
                nn.Linear(20, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
            )

        # three separate heads share the same architecture
        self.head_eta   = _make_head()
        self.head_mu    = _make_head()
        self.head_sigma = _make_head()

        # global α  (detached buffer)
        self.register_buffer("alpha", torch.tensor(math.e))   # ≈ 2.718
        
    # ----------------------------------------------------------------------
    def forward(self, seq):               # seq: [B, T, hidden_dim]
        _, h_n = self.gru(seq)            # h_n: [num_layers, B, H]
        z = h_n[-1]                       # take last-layer hidden   [B, H]

        eta   = self.head_eta(z).squeeze(-1)                   # [B]
        mu    = self.head_mu(z).squeeze(-1)                    # [B]
        sigma = torch.relu(self.head_sigma(z)).squeeze(-1) + 1e-6  # >0

        # Debug prints
        if torch.isnan(eta).any() or torch.isnan(mu).any() or torch.isnan(sigma).any():
            print("NaN detected in ImpactRNN forward pass:")
            print(f"eta stats: min={eta.min().item():.3f}, max={eta.max().item():.3f}, mean={eta.mean().item():.3f}")
            print(f"mu stats: min={mu.min().item():.3f}, max={mu.max().item():.3f}, mean={mu.mean().item():.3f}")
            print(f"sigma stats: min={sigma.min().item():.3f}, max={sigma.max().item():.3f}, mean={sigma.mean().item():.3f}")
            print(f"z stats: min={z.min().item():.3f}, max={z.max().item():.3f}, mean={z.mean().item():.3f}")

        return eta, mu, sigma

    # ----------------------------------------------------------------------
    def _phi(self, x):
        """Standard normal CDF implemented with torch.erf."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def predict_cumulative(self, l, eta, mu, sigma):
        """
        Ĉ_p^l = α · [ exp( η · Φ((ln l − μ)/σ) ) − 1 ]
        Args
        ----
        l     : scalar or [L] tensor of horizons
        eta,
        mu,
        sigma : [B]
        Returns
        -------
        Tensor [B, L]  – cumulative citations up to each horizon
        """
        L = l.numel()
        B = eta.size(0)
        x = (torch.log(l).view(1, L) - mu.view(B, 1)) / sigma.view(B, 1)
        phi = self._phi(x)
        out = self.alpha * (torch.exp(eta.view(B, 1) * phi) - 1.0)

        # Debug prints
        if torch.isnan(out).any():
            print("NaN detected in predict_cumulative:")
            print(f"x stats: min={x.min().item():.3f}, max={x.max().item():.3f}, mean={x.mean().item():.3f}")
            print(f"phi stats: min={phi.min().item():.3f}, max={phi.max().item():.3f}, mean={phi.mean().item():.3f}")
            print(f"eta*phi stats: min={(eta.view(B, 1) * phi).min().item():.3f}, max={(eta.view(B, 1) * phi).max().item():.3f}")
            print(f"exp(eta*phi) stats: min={torch.exp(eta.view(B, 1) * phi).min().item():.3f}, max={torch.exp(eta.view(B, 1) * phi).max().item():.3f}")
            print(f"out stats: min={out.min().item():.3f}, max={out.max().item():.3f}, mean={out.mean().item():.3f}")

        return out