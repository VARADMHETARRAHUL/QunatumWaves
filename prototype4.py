import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1) Vectorized Wave Encoder (FAST + CLEAN)
# ============================================================

class WaveEncoder(nn.Module):
    """
    Converts tokens → wave representation
    ψ_i(t) = Σ_m a_m * embedding_i * exp(i(k_i - ω_m t))
    """

    def __init__(self, vocab_size, d_model, num_frequencies=64):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies

        self.embedding = nn.Embedding(vocab_size, d_model)

        # (num_frequencies, d_model)
        self.amplitudes: torch.Tensor = nn.Parameter(
            torch.randn(num_frequencies, d_model) * 0.01
        )

        # Frequencies (num_frequencies)
        self.register_buffer("frequencies", torch.linspace(0.1, 10, num_frequencies))

    def forward(self, tokens, t=0.0):
        batch, seq_len = tokens.shape
        emb = self.embedding(tokens)  # (B, L, D)

        positions = torch.arange(seq_len, device=tokens.device).float()  # (L)
        k = 2 * np.pi * positions / seq_len  # (L)

        # Vectorize:
        # phase shape → (1, L, F)
        omega = 2 * np.pi * self.frequencies  # (F)
        phase = k.unsqueeze(-1) - omega * t   # (L, F)

        cos_p = torch.cos(phase).unsqueeze(0)  # (1, L, F)
        sin_p = torch.sin(phase).unsqueeze(0)  # (1, L, F)

        # amplitudes: (F, D)
        A = self.amplitudes.unsqueeze(0).unsqueeze(0)  # (1,1,F,D)

        # wave_real/wave_imag shape: (B, L, D)
        wave_real = torch.sum(A * cos_p.unsqueeze(-1) * emb.unsqueeze(2), dim=2)
        wave_imag = torch.sum(A * sin_p.unsqueeze(-1) * emb.unsqueeze(2), dim=2)

        return wave_real, wave_imag


# ============================================================
# 2) Correct Quantum Attention (FIXED SHAPE BUG)
# ============================================================

class QuantumAttention(nn.Module):
    """
    Quantum-inspired attention:
    A_ij = | <ψ_Q^i | ψ_K^j> |^2
    """

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Orthogonal initialization
        nn.init.orthogonal_(self.W_Q.weight)
        nn.init.orthogonal_(self.W_K.weight)
        nn.init.orthogonal_(self.W_V.weight)

    def forward(self, wave_real, wave_imag, mask=None):
        B, L, D = wave_real.shape
        H = self.num_heads
        Dh = self.d_head

        # Project
        Qr = self.W_Q(wave_real)
        Qi = self.W_Q(wave_imag)

        Kr = self.W_K(wave_real)
        Ki = self.W_K(wave_imag)

        Vr = self.W_V(wave_real)
        Vi = self.W_V(wave_imag)

        # Reshape -> (B, H, L, Dh)
        Qr = Qr.view(B, L, H, Dh).transpose(1, 2)
        Qi = Qi.view(B, L, H, Dh).transpose(1, 2)
        Kr = Kr.view(B, L, H, Dh).transpose(1, 2)
        Ki = Ki.view(B, L, H, Dh).transpose(1, 2)
        Vr = Vr.view(B, L, H, Dh).transpose(1, 2)
        Vi = Vi.view(B, L, H, Dh).transpose(1, 2)

        # Complex inner product
        inner_real = torch.matmul(Qr, Kr.transpose(-2, -1)) + torch.matmul(Qi, Ki.transpose(-2, -1))
        inner_imag = torch.matmul(Qr, Ki.transpose(-2, -1)) - torch.matmul(Qi, Kr.transpose(-2, -1))

        # magnitude squared
        attention = (inner_real**2 + inner_imag**2) / np.sqrt(Dh)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = F.softmax(attention, dim=-1)  # (B, H, L, L)

        # Phase
        phase = torch.atan2(inner_imag, inner_real)  # (B,H,L,L)
        cos_p = torch.cos(phase).unsqueeze(-1)  # (B,H,L,L,1)
        sin_p = torch.sin(phase).unsqueeze(-1)

        # Values rotated: shape → (B,H,L,L,Dh)
        Vr = Vr.unsqueeze(3)  # (B,H,L,1,Dh)
        Vi = Vi.unsqueeze(3)

        V_rot_real = Vr * cos_p - Vi * sin_p
        V_rot_imag = Vr * sin_p + Vi * cos_p

        att_expanded = attention.unsqueeze(-1)  # (B,H,L,L,1)

        # SUM over L dimension (keys)
        out_real = torch.sum(att_expanded * V_rot_real, dim=3)  # (B,H,L,Dh)
        out_imag = torch.sum(att_expanded * V_rot_imag, dim=3)

        # Merge heads
        out_real = out_real.transpose(1, 2).contiguous().view(B, L, D)
        out_imag = out_imag.transpose(1, 2).contiguous().view(B, L, D)

        # Final projection
        return self.W_O(out_real), self.W_O(out_imag), attention


# ============================================================
# 3) Stable Hamiltonian Evolution
# ============================================================

class HamiltonianEvolution(nn.Module):
    def __init__(self, d_model, dt=0.1):
        super().__init__()
        self.dt = dt

        self.H_sem = nn.Linear(d_model, d_model, bias=False)
        self.H_syn = nn.Linear(d_model, d_model, bias=False)
        self.H_rea = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.H_sem.weight)
        nn.init.xavier_uniform_(self.H_syn.weight)
        nn.init.xavier_uniform_(self.H_rea.weight)

    def forward(self, real, imag):

        H_real = self.H_sem(real) + self.H_syn(real) + self.H_rea(real)
        H_imag = self.H_sem(imag) + self.H_syn(imag) + self.H_rea(imag)

        evolved_real = real + self.dt * H_imag
        evolved_imag = imag - self.dt * H_real

        return evolved_real, evolved_imag


# ============================================================
# 4) Transformer Layer
# ============================================================

class WaveQuantumTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dt=0.1):
        super().__init__()
        self.attn = QuantumAttention(d_model, num_heads)
        self.H = HamiltonianEvolution(d_model, dt)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_i = nn.LayerNorm(d_model)

    def forward(self, r, i, mask=None):
        ar, ai, att = self.attn(r, i, mask)
        r = self.norm_r(r + ar)
        i = self.norm_i(i + ai)

        er, ei = self.H(r, i)
        r = self.norm_r(r + er)
        i = self.norm_i(i + ei)

        return r, i, att


# ============================================================
# 5) Full Model
# ============================================================

class WaveQuantumTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, num_frequencies=64, dt=0.1):
        super().__init__()

        self.encoder = WaveEncoder(vocab_size, d_model, num_frequencies)
        self.layers = nn.ModuleList([
            WaveQuantumTransformerLayer(d_model, num_heads, dt)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, t=0.0, mask=None):
        r, i = self.encoder(tokens, t)

        attentions = []
        for layer in self.layers:
            r, i, att = layer(r, i, mask)
            attentions.append(att)

        mags = torch.sqrt(r*r + i*i)
        logits = self.out(mags)

        return logits, attentions, (r, i)


# ============================================================
# 6) Reasoning Path Extraction
# ============================================================

def extract_reasoning_paths(wave_real, wave_imag, top_k=3):
    wave_complex = torch.complex(wave_real, wave_imag)
    freq = torch.fft.fft(wave_complex, dim=1)
    amps = torch.abs(freq).mean(dim=-1)

    top_vals, top_idx = torch.topk(amps, k=top_k, dim=1)

    paths = []
    B, L = amps.shape
    for b in range(B):
        p = []
        for k in range(top_k):
            fi = top_idx[b, k].item()
            amp = top_vals[b, k].item()
            freq_norm = fi / L

            if freq_norm < 0.2:
                mode = "semantic"
            elif freq_norm < 0.5:
                mode = "syntactic"
            else:
                mode = "detail"

            p.append({"frequency": freq_norm, "amplitude": amp, "mode": mode})

        paths.append(p)
    return paths


# ============================================================
# 7) TESTING
# ============================================================

if __name__ == "__main__":
    vocab = 1000
    model = WaveQuantumTransformer(
        vocab_size=vocab,
        d_model=256,
        num_layers=4,
        num_heads=8,
        num_frequencies=32,
    )

    tokens = torch.randint(0, vocab, (2, 10))

    print("Running forward pass...")
    logits, att, (wr, wi) = model(tokens)

    print("Logits:", logits.shape)
    print("Wave:", wr.shape)

    rp = extract_reasoning_paths(wr, wi, 3)
    print("\nReasoning paths for batch 0:")
    print(rp[0])
