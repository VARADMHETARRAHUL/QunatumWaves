#!/usr/bin/env python3
"""
Quantum Wave Transformer – Fixed version
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import re
from collections import Counter

device = torch.device("cpu")

# ------------------------------------------------------------------
# 0.  small helpers
# ------------------------------------------------------------------
def ensure_complex(x):
    if x.dtype in (torch.float32, torch.float64):
        return torch.complex(x, torch.zeros_like(x))
    return x

def complex_gelu(z):
    return torch.complex(F.gelu(z.real), F.gelu(z.imag))

def complex_to_real(z):
    """(...,D) complex -> (...,2D) real"""
    return torch.view_as_real(z).reshape(*z.shape[:-1], -1)

# ------------------------------------------------------------------
# 1.  tokenizer (unchanged API)
# ------------------------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab_size=20_000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, texts):
        c = Counter()
        for t in texts:
            t = t.lower()
            c.update(re.findall(r"[A-Za-z0-9]+|\S", t))
        vocab = [w for w, f in c.items() if f >= self.min_freq][: self.vocab_size - 4]
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        idx = 4
        for w in vocab:
            self.word2id[w] = idx
            idx += 1
        self.id2word = {i: w for w, i in self.word2id.items()}
        print(f"[Tokenizer] vocab = {len(self.word2id)}")

    def encode(self, text, max_len=128):
        text = text.lower()
        words = re.findall(r"[A-Za-z0-9]+|\S", text)[: max_len - 2]
        ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in words]
        ids = [self.word2id["<BOS>"]] + ids + [self.word2id["<EOS>"]]
        while len(ids) < max_len:
            ids.append(self.word2id["<PAD>"])
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.id2word.get(int(i), "<UNK>")
            if w in {"<PAD>", "<BOS>", "<EOS>"}:
                continue
            words.append(w)
        return " ".join(words)

# ------------------------------------------------------------------
# 2.  complex layers
# ------------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.Wi = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.br = nn.Parameter(torch.zeros(out_dim))
        self.bi = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = ensure_complex(x)
        xr, xi = x.real, x.imag
        real = F.linear(xr, self.Wr, self.br) - F.linear(xi, self.Wi)
        imag = F.linear(xr, self.Wi, self.bi) + F.linear(xi, self.Wr)
        return torch.complex(real, imag)

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.complex64))

    def forward(self, x):
        x = ensure_complex(x)
        var = x.abs().pow(2).mean(dim=-1, keepdim=True)
        return self.gamma * x / torch.sqrt(var + self.eps)

class ComplexDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = ensure_complex(x)
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x.real) > self.p).float() / (1 - self.p)
        return torch.complex(x.real * mask, x.imag * mask)

# ------------------------------------------------------------------
# 3.  attention modules
# ------------------------------------------------------------------
class QuantumAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.scale = 1 / math.sqrt(self.dh)
        self.to_q = ComplexLinear(dim, dim)
        self.to_k = ComplexLinear(dim, dim)
        self.to_v = ComplexLinear(dim, dim)
        self.to_out = ComplexLinear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = ensure_complex(x)
        B, N, D = x.shape
        q = self.to_q(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        sim = torch.einsum("bhid,bhjd->bhij", q, k.conj()) * self.scale
        attn = F.softmax(sim.abs(), dim=-1)
        attn = self.drop(attn)
        attn = torch.complex(attn, torch.zeros_like(attn))
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        return self.to_out(out)

class FFTAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.to_q = ComplexLinear(dim, dim)
        self.to_k = ComplexLinear(dim, dim)
        self.to_v = ComplexLinear(dim, dim)
        self.to_out = ComplexLinear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = ensure_complex(x)
        B, N, D = x.shape
        q = self.to_q(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        qf = torch.fft.fft(q, dim=2, norm="ortho")
        kf = torch.fft.fft(k, dim=2, norm="ortho")
        vf = torch.fft.fft(v, dim=2, norm="ortho")
        sim = torch.einsum("bhid,bhjd->bhij", qf, kf.conj())
        attn = F.softmax(sim.abs(), dim=-1)
        attn = self.drop(attn)
        attn = torch.complex(attn, torch.zeros_like(attn))
        out_f = torch.einsum("bhij,bhjd->bhid", attn, vf)
        out_t = torch.fft.ifft(out_f, dim=2, norm="ortho")
        out = out_t.transpose(1, 2).contiguous().reshape(B, N, D)
        return self.to_out(out)

# ------------------------------------------------------------------
# 4.  hybrid router - FIX: Use real tensors for router
# ------------------------------------------------------------------
class HybridAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.q_attn = QuantumAttention(dim, heads, dropout)
        self.f_attn = FFTAttention(dim, heads, dropout)
        # FIX: Router input is dim*4 (2*dim for each complex tensor when converted to real)
        self.router = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, psi):
        x, psi = ensure_complex(x), ensure_complex(psi)
        B, N, D = x.shape
        seq_mean = x.mean(dim=1)                         # (B,D)
        # FIX: Convert complex to real for router - each complex D becomes 2*D real
        seq_mean_real = torch.view_as_real(seq_mean).reshape(B, -1)  # (B, D*2)
        psi_real = torch.view_as_real(psi).reshape(B, -1)             # (B, D*2)
        router_in = torch.cat([seq_mean_real, psi_real], dim=-1)      # (B, D*4)
        r = self.router(router_in).view(B, 1, 1)         # (B,1,1)
        aq = self.q_attn(x)
        af = self.f_attn(x)
        # FIX: Convert r to complex for mixing
        r_complex = torch.complex(r, torch.zeros_like(r))
        out = r_complex * aq + (1 - r_complex) * af
        return out, r

# ------------------------------------------------------------------
# 5.  Schrödinger evolution - FIX: Handle both 2D and 3D inputs
# ------------------------------------------------------------------
class SchrodingerEvolution(nn.Module):
    def __init__(self, dim, rank=16, dt=0.05):
        super().__init__()
        self.dim, self.rank = dim, rank
        # FIX: Make ε and V complex parameters
        self.ε_real = nn.Parameter(torch.randn(dim) * 0.02)
        self.ε_imag = nn.Parameter(torch.zeros(dim))
        self.V_real = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.V_imag = nn.Parameter(torch.zeros(dim, rank))
        self.dt = nn.Parameter(torch.tensor(dt))

    def forward(self, x):
        x = ensure_complex(x)
        
        # FIX: Handle both 2D (B,D) and 3D (B,N,D) inputs
        original_shape = x.shape
        if len(original_shape) == 2:
            # Add sequence dimension: (B,D) -> (B,1,D)
            x = x.unsqueeze(1)
        
        B, N, D = x.shape
        flat = x.reshape(-1, D)  # (B*N, D)
        
        # Build complex tensors properly
        ε = torch.complex(self.ε_real, self.ε_imag)
        V = torch.complex(self.V_real, self.V_imag)
        
        H = torch.diag(ε) + V @ V.conj().T
        U = torch.matrix_exp(-1j * H * self.dt)
        out = flat @ U.T
        out = out.reshape(B, N, D)
        
        # Restore original shape
        if len(original_shape) == 2:
            out = out.squeeze(1)  # (B,1,D) -> (B,D)
        
        return out

# ------------------------------------------------------------------
# 6.  global superposition memory
# ------------------------------------------------------------------
class PsiMemory(nn.Module):
    psi: torch.Tensor  # Type annotation for the buffer
    
    def __init__(self, dim, decay=0.97):
        super().__init__()
        self.decay = decay
        self.register_buffer("psi", torch.zeros(dim, dtype=torch.complex64))
        self.evolver = SchrodingerEvolution(dim)

    def forward(self, x):
        x = ensure_complex(x)
        seq_mean = x.mean(dim=1)               # (B,D)
        batch_mean = seq_mean.mean(dim=0)      # (D,)
        
        # FIX: Detach batch_mean before in-place operations to avoid autograd issues
        batch_mean_detached = batch_mean.detach()
        
        # Update psi with exponential moving average (in-place, no gradients)
        weight = 1 - self.decay
        self.psi.mul_(self.decay)
        self.psi.add_(batch_mean_detached * weight)
        
        # Evolver expects (B,D) and returns (B,D) now
        # Use the updated psi (detached) for evolution
        psi_evo = self.evolver(self.psi.unsqueeze(0)).squeeze(0)
        self.psi.copy_(psi_evo.detach())
        
        # Return a version with gradients for the forward pass
        return psi_evo.unsqueeze(0).expand(x.size(0), -1).detach()
# ------------------------------------------------------------------
# 6b. Quantum State Tracker (NEW)
# ------------------------------------------------------------------
class QuantumStateTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = []  # stores dicts of movement

    def measure(self, label, psi_before, psi_after):
        """
        psi_before, psi_after are (B, N, D) complex tensors
        """
        # Probability (magnitude squared)
        prob_before = psi_before.abs()**2
        prob_after = psi_after.abs()**2
        prob_flow = (prob_after - prob_before).mean().item()

        # Phase (angle)
        phase_before = torch.angle(psi_before)
        phase_after = torch.angle(psi_after)
        phase_shift = (phase_after - phase_before).mean().item()

        self.history.append({
            "stage": label,
            "prob_flow": prob_flow,
            "phase_shift": phase_shift
        })

    def reset(self):
        self.history = []

    def get(self):
        return self.history

# ------------------------------------------------------------------
# 7.  feed-forward (complex)
# ------------------------------------------------------------------
class QuantumFFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        h = int(dim * mult)
        self.w1 = ComplexLinear(dim, h)
        self.w2 = ComplexLinear(h, dim)
        self.drop = ComplexDropout(dropout)

    def forward(self, x):
        x = ensure_complex(x)
        return self.drop(self.w2(complex_gelu(self.w1(x))))

# ------------------------------------------------------------------
# 8.  WaveBlock - FIX: LayerNorm on real view
# ------------------------------------------------------------------
class WaveBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, tracker=None):
        super().__init__()
        self.hybrid = HybridAttention(dim, heads, dropout)
        self.ff = QuantumFFN(dim, mult=4, dropout=dropout)
        self.evolver = SchrodingerEvolution(dim)
        self.norm1 = nn.LayerNorm(dim * 2)
        self.norm2 = nn.LayerNorm(dim * 2)

        # NEW: tracker hook
        self.tracker = tracker

    def forward(self, x, psi):
        x = ensure_complex(x)

        # track movement before block
        if self.tracker is not None:
            x_before = x.clone()

        # ---- Attention ----
        h, r = self.hybrid(x, psi)
        x = x + h

        # track attention movement
        if self.tracker is not None:
            self.tracker.measure("attention", x_before, x)
            x_before = x.clone()

        # ---- LayerNorm ----
        x_real = torch.view_as_real(x).reshape(*x.shape[:-1], -1)
        x_real = self.norm1(x_real)
        x = torch.view_as_complex(x_real.reshape(*x.shape, 2).contiguous())

        # ---- FFN ----
        x_ff = self.ff(x)
        x = x + x_ff

        # track FFN movement
        if self.tracker is not None:
            self.tracker.measure("ffn", x_before, x)
            x_before = x.clone()

        # ---- Norm ----
        x_real = torch.view_as_real(x).reshape(*x.shape[:-1], -1)
        x_real = self.norm2(x_real)
        x = torch.view_as_complex(x_real.reshape(*x.shape, 2).contiguous())

        # ---- Schrödinger evolver ----
        x = self.evolver(x)

        # track quantum evolution movement
        if self.tracker is not None:
            self.tracker.measure("schrodinger", x_before, x)

        return x, r

# ------------------------------------------------------------------
# 9.  full model
# ------------------------------------------------------------------
class QuantumWaveTransformer(nn.Module):
    def __init__(self, dim=256, depth=8, heads=8, dropout=0.1):
        super().__init__()

        self.input_proj = ComplexLinear(dim, dim)
        self.psi_mem = PsiMemory(dim)

        # FIX: tracker must be defined BEFORE using it
        self.tracker = QuantumStateTracker()

        # now the tracker exists → safe to pass it
        self.layers = nn.ModuleList([
            WaveBlock(dim, heads, dropout, tracker=self.tracker)
            for _ in range(depth)
        ])

        self.out_proj = nn.Linear(dim * 2, dim)


    def forward(self, x, return_router=False):
        # x: (B,N,D) real
        x = ensure_complex(self.input_proj(ensure_complex(x)))
        psi = self.psi_mem(x)
        router_vals = []
        for layer in self.layers:
            x, r = layer(x, psi)
            router_vals.append(r.detach().cpu())
            psi = self.psi_mem(x)
        out = self.out_proj(complex_to_real(x))
        return (out, router_vals) if return_router else out

# ------------------------------------------------------------------
# 10.  data generators
# ------------------------------------------------------------------
def generate_RLC(batch=16, length=64):
    t = torch.linspace(0, 8, length)
    alpha, omega = 0.3, 1.5
    I = torch.exp(-alpha * t) * torch.cos(omega * t)
    Q = torch.exp(-alpha * t) * torch.sin(omega * t)
    sig = torch.stack([I, Q], dim=-1)                 # (L,2)
    sig = sig / (sig.norm(dim=-1, keepdim=True) + 1e-6)
    batch_data = sig.unsqueeze(0).repeat(batch, 1, 1)  # (B,L,2)
    pad = torch.zeros(batch, length, 254)
    return torch.cat([batch_data, pad], dim=-1).float()

_lang_emb = nn.Embedding(1000, 256)
def generate_language(batch=16, length=64, vocab=1000):
    tok = torch.randint(0, vocab, (batch, length))
    return _lang_emb(tok).detach()

def generate_hybrid(batch=16, length=64):
    lang = generate_language(batch // 2, length)
    phys = generate_RLC(batch // 2, length)
    return torch.cat([lang, phys], dim=0)

# ------------------------------------------------------------------
# 11.  training & visualisation
# ------------------------------------------------------------------
def train_model(model, steps=800, batch=16, mode="hybrid"):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=steps)
    router_log = []
    print(f"\n=== TRAINING MODE: {mode.upper()} ===")
    for step in range(steps):
        if mode == "physics":
            x = generate_RLC(batch)
        elif mode == "language":
            x = generate_language(batch)
        else:
            x = generate_hybrid(batch)

        pred, routers = model(x, return_router=True)
        loss = F.mse_loss(pred[..., :2], x[..., :2])   # non-padded slice

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            r_mean = torch.cat([r.flatten() for r in routers]).mean()
            router_log.append(r_mean.item())
            print(f"[{step}] loss={loss.item():.6f}  router={r_mean:.3f}  lr={sched.get_last_lr()[0]:.2e}")
    return router_log

def show_physics_prediction(model):
    model.eval()
    with torch.no_grad():
        x = generate_RLC(1)
        pred = model(x).detach()[0, :, 0].numpy()
        gt   = x[0, :, 0].numpy()
    model.train()
    
    plt.figure(figsize=(10, 5))
    plt.plot(gt, label="Ground Truth", linewidth=2)
    plt.plot(pred, label="Prediction", linestyle="--")
    plt.title("RLC Circuit – Quantum Wave Transformer")
    plt.legend(); plt.grid(); plt.show()

def show_router(router_log):
    plt.figure(figsize=(8, 4))
    plt.plot(router_log)
    plt.title("Router Strength (Quantum vs FFT)")
    plt.ylabel("r"); plt.xlabel("Checkpoint (×100 steps)")
    plt.grid(); plt.show()

# ------------------------------------------------------------------
# 12.  main demo
# ------------------------------------------------------------------
def main_demo():
    print("\n=== QUANTUM WAVE TRANSFORMER DEMO ===")
    model = QuantumWaveTransformer(dim=256, depth=8, heads=8)

    router_phys = train_model(model, steps=800, mode="physics")
    show_router(router_phys)
    show_physics_prediction(model)

    router_lang = train_model(model, steps=800, mode="language")
    show_router(router_lang)

    router_mix = train_model(model, steps=800, mode="hybrid")
    show_router(router_mix)

    print("\n=== TRAINING COMPLETE ===")

def show_quanta_movement(model):
    print("\n=== QUANTA MOVEMENT TRACE ===")
    for step in model.tracker.get():
        stage = step["stage"]
        pf = step["prob_flow"]
        ph = step["phase_shift"]
        print(f"{stage:15s} | prob_flow={pf:+.6f} | phase_shift={ph:+.6f}")


if __name__ == "__main__":
    main_demo()