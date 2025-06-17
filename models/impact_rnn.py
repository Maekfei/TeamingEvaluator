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
        self.dropout = nn.Dropout(0.1)
        
        # Add layer normalization to stabilize GRU
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
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

        # Initialize weights with adaptive gain based on hidden dimension
        self._init_weights(hidden_dim)

        # global α  (detached buffer)
        self.register_buffer("alpha", torch.tensor(math.e))   # ≈ 2.718
        
    def _init_weights(self, hidden_dim):
        """Initialize weights with adaptive gain based on hidden dimension."""
        # Adjust gain based on hidden dimension size
        gru_gain = min(0.1, 1.0 / math.sqrt(hidden_dim))
        mlp_gain = min(0.1, 1.0 / math.sqrt(hidden_dim))
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=gru_gain)
            elif 'bias' in name:
                # Initialize biases to small values
                nn.init.constant_(param, 0.01)
        
        # Initialize MLP weights
        for name, param in self.named_parameters():
            if 'head' in name and 'weight' in name:
                nn.init.xavier_uniform_(param, gain=mlp_gain)
            elif 'head' in name and 'bias' in name:
                nn.init.zeros_(param)
        
    # ----------------------------------------------------------------------
    def forward(self, seq):               # seq: [B, T, hidden_dim]
        # Debug input sequence only if NaNs are detected
        if torch.isnan(seq).any():
            print("NaN detected in input sequence:")
            print(f"seq stats: min={seq.min().item():.3f}, max={seq.max().item():.3f}, mean={seq.mean().item():.3f}")
            # Replace NaNs with zeros
            seq = torch.nan_to_num(seq, nan=0.0)
        
        # Apply layer normalization to input sequence
        seq = self.layer_norm(seq)
        
        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, seq.size(0), self.gru.hidden_size, device=seq.device)
        
        # Forward pass through GRU with gradient clipping
        with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision to avoid potential issues
            _, h_n = self.gru(seq, h0)            # h_n: [num_layers, B, H]
        
        z = h_n[-1]                       # take last-layer hidden   [B, H]
        z = self.dropout(z)               # apply dropout to the hidden state
        z = self.layer_norm(z)            # normalize hidden state

        # Debug hidden state only if NaNs are detected
        if torch.isnan(z).any():
            print("NaN detected in hidden state after GRU:")
            print(f"z stats: min={z.min().item():.3f}, max={z.max().item():.3f}, mean={z.mean().item():.3f}")
            # Replace NaNs with zeros
            z = torch.nan_to_num(z, nan=0.0)
            z = self.layer_norm(z)  # Re-normalize after NaN replacement

        # Get outputs with adaptive bounds based on hidden dimension
        max_val = 8.0 / math.sqrt(self.gru.hidden_size / 8.0)  # Scale bounds with hidden dimension
        eta   = torch.clamp(self.head_eta(z).squeeze(-1), min=-max_val, max=max_val)                   # [B]
        mu    = torch.clamp(self.head_mu(z).squeeze(-1), min=-max_val, max=max_val)                    # [B]
        sigma = torch.clamp(torch.relu(self.head_sigma(z)).squeeze(-1) + 1e-4, min=1e-4, max=max_val)  # >0

        # Debug outputs only if NaNs are detected
        if torch.isnan(eta).any() or torch.isnan(mu).any() or torch.isnan(sigma).any():
            print("NaN detected in ImpactRNN forward pass:")
            print(f"eta stats: min={eta.min().item():.3f}, max={eta.max().item():.3f}, mean={eta.mean().item():.3f}")
            print(f"mu stats: min={mu.min().item():.3f}, max={mu.max().item():.3f}, mean={mu.mean().item():.3f}")
            print(f"sigma stats: min={sigma.min().item():.3f}, max={sigma.max().item():.3f}, mean={sigma.mean().item():.3f}")
            print(f"z stats: min={z.min().item():.3f}, max={z.max().item():.3f}, mean={z.mean().item():.3f}")
            raise RuntimeError("NaN detected in model outputs - stopping experiment")

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
        
        # Add small epsilon to prevent numerical issues
        l = l + 1e-4
        sigma = sigma + 1e-4
        
        x = (torch.log(l).view(1, L) - mu.view(B, 1)) / sigma.view(B, 1)
        phi = self._phi(x)
        
        # Clamp the exponent to prevent overflow
        max_val = 8.0 / math.sqrt(self.gru.hidden_size / 8.0)  # Scale bounds with hidden dimension
        exp_term = torch.exp(torch.clamp(eta.view(B, 1) * phi, min=-max_val, max=max_val))
        out = self.alpha * (exp_term - 1.0)

        # Debug output only if NaNs are detected
        if torch.isnan(out).any():
            print("NaN detected in predict_cumulative:")
            print(f"x stats: min={x.min().item():.3f}, max={x.max().item():.3f}, mean={x.mean().item():.3f}")
            print(f"phi stats: min={phi.min().item():.3f}, max={phi.max().item():.3f}, mean={phi.mean().item():.3f}")
            print(f"eta*phi stats: min={(eta.view(B, 1) * phi).min().item():.3f}, max={(eta.view(B, 1) * phi).max().item():.3f}")
            print(f"exp(eta*phi) stats: min={exp_term.min().item():.3f}, max={exp_term.max().item():.3f}")
            print(f"out stats: min={out.min().item():.3f}, max={out.max().item():.3f}, mean={out.mean().item():.3f}")
            raise RuntimeError("NaN detected in predictions - stopping experiment")

        return out