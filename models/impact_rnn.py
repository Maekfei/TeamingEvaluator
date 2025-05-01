import torch
import torch.nn as nn
import math


class ImpactRNN(nn.Module):
    """
    GRU encodes the imputed sequence  (years before publication).
    Three MLP heads produce   η, μ, σ    (paper-specific).
    """
    def __init__(self, hidden_dim: int, rnn_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.head_eta = nn.Linear(hidden_dim, 1)
        self.head_mu  = nn.Linear(hidden_dim, 1)
        self.head_sigma = nn.Linear(hidden_dim, 1)

        # global α is registered as a (detached) buffer for convenience
        self.register_buffer("alpha", torch.tensor(math.e))  # ≈ 2.718

    def forward(self, seq):            # seq: [B, T, hidden_dim]
        _, h_n = self.gru(seq)         # h_n: [num_layers, B, H]
        z = h_n[-1]                    # [B, H]

        eta   = self.head_eta(z).squeeze(-1)     # [B]
        mu    = self.head_mu(z).squeeze(-1)
        sigma = torch.relu(self.head_sigma(z)).squeeze(-1) + 1e-6  # >0

        return eta, mu, sigma

    # --- helpers ----------------------------------------------------
    def _phi(self, x):
        """
        Φ(x) – CDF of standard normal distribution using torch.erf
        """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def predict_cumulative(self, l, eta, mu, sigma):
        """
        Eq.(*)   Ĉ_p^l = α [ exp( η · Φ( (ln l – μ)/σ ) ) – 1 ]
        Args:
            l     : tensor scalar or [L] of horizons
            eta, mu, sigma : [B]
        Returns:
            C_hat : [B, L]
        """
        L = l.numel()
        B = eta.size(0)
        x = (torch.log(l).view(1, L) - mu.view(B, 1)) / sigma.view(B, 1)
        phi = self._phi(x)
        out = self.alpha * (torch.exp(eta.view(B, 1) * phi) - 1.0)
        return out                       # [B, L]