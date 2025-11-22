import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================================
# GLOBAL SETTINGS (CPU optimized)
# ==============================================================

DEVICE = "cpu"
DIM = 128
LAYERS = 8
SEQ_LEN = 64        # Physics sequence T=64
FFT_MODES = 32      # half of DIM/2


# ==============================================================
# 1. COMPLEX LINEAR LAYER
# ==============================================================

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.Wi = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        self.br = nn.Parameter(torch.zeros(out_features))
        self.bi = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        xr, xi = x.real, x.imag

        real = F.linear(xr, self.Wr, self.br) - F.linear(xi, self.Wi)
        imag = F.linear(xr, self.Wi, self.bi) + F.linear(xi, self.Wr)

        return torch.complex(real, imag)


# ==============================================================
# 2. UNITARY APPROXIMATION (for fast CPU K and V evolution)
# ==============================================================

def make_unitary(W):
    # QR decomposition → Q is orthogonal/unitary
    Q, _ = torch.linalg.qr(W)
    return Q


# ==============================================================
# 3. FULL SCHRÖDINGER EVOLUTION (for Q)
# ==============================================================

class SchrodingerEvolver(nn.Module):
    def __init__(self, dim, dt=0.06):
        super().__init__()
        H = torch.randn(dim, dim) * 0.02
        H = (H + H.T) / 2  # Hermitian
        self.H = nn.Parameter(H)
        self.dt = dt

    def forward(self, x):
        B, N, D = x.shape
        U = torch.matrix_exp(-1j * self.H * self.dt)
        x = x.reshape(B*N, D)
        out = x @ U
        return out.reshape(B, N, D)


# ==============================================================
# 4. GAUSSIAN WAVE TOKENIZER (for physics)
# ==============================================================

def gaussian_wave_tokenizer(x, dim):
    """
    x : (B, T, 1) → physics input (scalar signal)
    returns: complex wave embedding (B, T, dim)
    """
    B, T, _ = x.shape

    grid = torch.linspace(-3, 3, dim)
    grid = grid.unsqueeze(0).unsqueeze(0)  # (1,1,dim)

    mu = x  # center of wave packet
    sigma = 0.7

    psi = torch.exp(- (grid - mu)**2 / (2*sigma*sigma))
    psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-8)

    phase = torch.exp(1j * (grid * mu * 3.2))
    psi = psi * phase

    return psi.to(torch.complex64)


# ==============================================================
# 5. FFT TOKENIZER (for language)
# ==============================================================

class FFTTokenizer(nn.Module):
    def __init__(self, vocab_size=30000, dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, toks):
        # toks : (B, T)
        x = self.emb(toks).float()  # real
        xf = torch.fft.fft(x, dim=-1)
        return xf


# ==============================================================
# 6. HYBRID TOKENIZER (Your design)
# ==============================================================

class HybridTokenizer(nn.Module):
    def __init__(self, dim=DIM, vocab=30000):
        super().__init__()
        self.lang_tok = FFTTokenizer(vocab, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # mix weight

    def forward(self, physics_input=None, language_input=None):
        """
        physics_input: (B,T,1)
        language_input: (B,T)
        one of them is provided
        """

        if physics_input is not None:
            psi_phys = gaussian_wave_tokenizer(physics_input, DIM)
            return psi_phys  # pure wave, no mixing

        if language_input is not None:
            psi_lang = self.lang_tok(language_input)
            psi_lang = psi_lang / (psi_lang.norm(dim=-1, keepdim=True) + 1e-8)
            return psi_lang  # pure FFT embed

        raise ValueError("No tokenizer input provided!")


# ==============================================================
# 7. HYBRID Q-K-V EVOLUTION MODULE
# ==============================================================

class QuantumQKV(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Full Schrödinger for Q
        self.sch_Q = SchrodingerEvolver(dim)

        # Approximate unitary for K
        Wk = make_unitary(torch.randn(dim, dim))
        self.Wk = nn.Parameter(Wk)

        # Hybrid V
        self.V_linear = ComplexLinear(dim, dim)
        Wv = make_unitary(torch.randn(dim, dim))
        self.Wv = nn.Parameter(Wv)
        self.sch_V = SchrodingerEvolver(dim)

        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x is complex wave: (B, T, dim)

        Q = self.sch_Q(x)

        xr = x.real
        xi = x.imag
        K = torch.complex(xr @ self.Wk, xi @ self.Wk)

        V1 = self.V_linear(x)
        V2 = self.sch_V(x)
        V3 = torch.complex(xr @ self.Wv, xi @ self.Wv)

        V = self.gamma * V1 + (1 - self.gamma) * (V2 + V3) / 2
        return Q, K, V


# ==============================================================
# 8. QUANTUM INTERFERENCE ATTENTION
# ==============================================================

class QuantumAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, Q, K, V):
        B, T, D = Q.shape

        Qf = torch.fft.fft(Q, dim=1)
        Kf = torch.fft.fft(K, dim=1)
        Vf = torch.fft.fft(V, dim=1)

        scores = Qf * Kf.conj()  # complex interference
        attn = F.softmax(scores.real, dim=1)

        attn = attn.to(torch.complex64)

        out_f = attn * Vf
        out = torch.fft.ifft(out_f, dim=1)

        return out


# ==============================================================
# 9. COMPLEX FEEDFORWARD
# ==============================================================

class ComplexFF(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.lin1 = ComplexLinear(dim, dim * mult)
        self.lin2 = ComplexLinear(dim * mult, dim)

    def forward(self, x):
        x = self.lin1(x)
        xr = F.gelu(x.real)
        xi = F.gelu(x.imag)
        x = torch.complex(xr, xi)
        x = self.lin2(x)
        return x


# ==============================================================
# 10. SINGLE QUANTUMWAVE BLOCK
# ==============================================================

class QuantumWaveBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = QuantumQKV(dim)
        self.attn = QuantumAttention(dim)
        self.ff = ComplexFF(dim)

    def forward(self, x):
        Q, K, V = self.qkv(x)
        x = x + self.attn(Q, K, V)
        x = x + self.ff(x)
        return x


# ==============================================================
# 11. FULL QUANTUMWAVE TRANSFORMER
# ==============================================================

class QuantumWaveTransformer(nn.Module):
    def __init__(self, dim=DIM, layers=LAYERS):
        super().__init__()
        self.tokenizer = HybridTokenizer(dim)
        self.blocks = nn.ModuleList([QuantumWaveBlock(dim) for _ in range(layers)])
        self.out_proj = nn.Linear(dim*2, 1)  # simple projection

    def forward(self, wave):
        x = wave
        for blk in self.blocks:
            x = blk(x)

        xr = x.real
        xi = x.imag
        out = torch.cat([xr, xi], dim=-1)
        out = self.out_proj(out)
        return out


# ==============================================================
# 12. SCHRÖDINGER DATASET (Split-Step Fourier)
# ==============================================================

def generate_schrodinger_dataset(batch=4, T=64, N=96):
    x = np.linspace(-4, 4, N)
    dx = x[1] - x[0]
    k = 2*np.pi*np.fft.fftfreq(N, d=dx)
    V = 0.5 * x**2

    all_seq = []

    for _ in range(batch):
        psi = np.exp(-x**2) * np.exp(1j*np.random.uniform(1,4)*x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2))

        seq = []
        for _ in range(T):
            seq.append(psi.copy())

            psi *= np.exp(-1j*V*0.05/2)
            psi_k = np.fft.fft(psi)
            psi_k *= np.exp(-1j*(k**2)/2 * 0.05)
            psi = np.fft.ifft(psi_k)
            psi *= np.exp(-1j*V*0.05/2)
            psi /= np.sqrt(np.sum(np.abs(psi)**2))

        seq = np.stack(seq)
        all_seq.append(seq)

    return torch.from_numpy(np.stack(all_seq)).to(torch.complex64)


def compress_wave(psi, modes=64):
    # psi (B,T,N)
    fft = torch.fft.fft(psi, dim=-1)
    low = fft[..., :modes]
    x = torch.cat([low.real, low.imag], dim=-1)
    return x.float()


# ==============================================================
# 13. TRAINING LOOP (Physics-first)
# ==============================================================

def train_physics(model, steps=600):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(steps):
        psi = generate_schrodinger_dataset(batch=2, T=SEQ_LEN, N=96)
        inp = compress_wave(psi)  # (B,T,128)
        inp = gaussian_wave_tokenizer(inp.mean(dim=-1, keepdim=True), DIM)

        pred = model(inp)
        target = inp.real.mean(dim=-1, keepdim=True)

        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[Physics {step}] Loss = {loss.item():.6f}")

    print("Physics training complete.")


# ==============================================================
# 14. VISUALIZATION
# ==============================================================

def visualize_wave(model):
    psi = generate_schrodinger_dataset(batch=1, T=SEQ_LEN, N=96)
    inp = compress_wave(psi)
    wave = gaussian_wave_tokenizer(inp.mean(dim=-1, keepdim=True), DIM)

    out = model(wave).detach().numpy()[0,:,0]

    plt.plot(out, label="Predicted Signal")
    plt.legend()
    plt.title("QuantumWave Output (Physics Mode)")
    plt.show()


# ==============================================================
# 15. MAIN
# ==============================================================

def main():
    print("=== QuantumWave Transformer v0.2 (CPU Edition) ===")

    model = QuantumWaveTransformer().to(DEVICE)

    print("\nTraining physics core...")
    train_physics(model)

    print("\nVisualizing...")
    visualize_wave(model)


if __name__ == "__main__":
    main()
