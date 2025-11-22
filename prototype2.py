import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# =====================================================
# 1. COMPLEX LINEAR LAYER
# =====================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(out_dim, in_dim)*0.02)
        self.Wi = nn.Parameter(torch.randn(out_dim, in_dim)*0.02)
        self.br = nn.Parameter(torch.zeros(out_dim))
        self.bi = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        xr, xi = x.real, x.imag
        real = F.linear(xr, self.Wr, self.br) - F.linear(xi, self.Wi)
        imag = F.linear(xr, self.Wi, self.bi) + F.linear(xi, self.Wr)
        return torch.complex(real, imag)

# =====================================================
# 2. QUANTUM ATTENTION
# =====================================================
class QuantumAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads

        self.to_q = ComplexLinear(dim, dim)
        self.to_k = ComplexLinear(dim, dim)
        self.to_v = ComplexLinear(dim, dim)
        self.out = ComplexLinear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        Q = self.to_q(x).view(B,N,self.heads,self.dh).permute(0,2,1,3)
        K = self.to_k(x).view(B,N,self.heads,self.dh).permute(0,2,1,3)
        V = self.to_v(x).view(B,N,self.heads,self.dh).permute(0,2,1,3)

        # Quantum overlap
        sim = torch.einsum("bhid,bhjd->bhij", Q, torch.conj(K))

        # Real scalar score for softmax
        attn = F.softmax(sim.abs() / math.sqrt(self.dh), dim=-1)

        # Convert attention -> complex (same dtype as V)
        attn = torch.complex(attn, torch.zeros_like(attn))

        # Complex weighted sum
        out = torch.einsum("bhij,bhjd->bhid", attn, V)

        out = out.permute(0,2,1,3).contiguous().view(B,N,D)
        return self.out(out)

# =====================================================
# 3. QUANTUM FEED-FORWARD
# =====================================================
class QuantumFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc1 = ComplexLinear(dim, dim * mult)
        self.fc2 = ComplexLinear(dim * mult, dim)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.complex(F.gelu(h.real), F.gelu(h.imag))
        return self.fc2(h)

# =====================================================
# 4. SCHRÖDINGER EVOLUTION LAYER
# =====================================================
class SchrodingerEvolution(nn.Module):
    def __init__(self, dim, dt=0.05):
        super().__init__()
        H = torch.randn(dim, dim)*0.02
        H = (H + H.T) / 2
        self.H = nn.Parameter(H)
        self.dt = nn.Parameter(torch.tensor(dt))

    def forward(self, x):
        B, N, D = x.shape
        U = torch.matrix_exp(-1j * self.H * self.dt)
        x_flat = x.reshape(B*N, D)
        out = x_flat @ U
        return out.reshape(B,N,D)

# =====================================================
# 5. COMPLETE TRANSFORMER BLOCK
# =====================================================
class QuantumBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = QuantumAttention(dim, heads)
        self.ff = QuantumFFN(dim)
        self.sch = SchrodingerEvolution(dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        x = self.sch(x)
        return x

# =====================================================
# 6. FULL QUANTUM TRANSFORMER
# =====================================================
class QuantumTransformer(nn.Module):
    def __init__(self, dim=128, depth=8, heads=8):
        super().__init__()
        self.input_proj = ComplexLinear(dim, dim)
        self.layers = nn.ModuleList([QuantumBlock(dim, heads) for _ in range(depth)])
        self.output_proj = nn.Linear(dim*2, dim)

    def forward(self, x):
        x = torch.complex(x, torch.zeros_like(x))
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x_final = torch.cat([x.real, x.imag], dim=-1)
        return self.output_proj(x_final)

# =====================================================
# 7. PHYSICS DATA (RLC)
# =====================================================
def generate_RLC(batch=8, length=64):
    t = torch.linspace(0, 8, length)
    alpha = 0.3
    omega = 1.5

    I = torch.exp(-alpha*t) * torch.cos(omega*t)
    I = I / (I.abs().max() + 1e-6)

    dI = torch.gradient(I)[0]

    base = torch.stack([I, dI], dim=-1)
    batch_data = base.unsqueeze(0).repeat(batch,1,1)

    pad = torch.zeros(batch, length, 126)
    return torch.cat([batch_data, pad], dim=-1).float()

# =====================================================
# 8. TRAINING LOOP
# =====================================================
def train_physics(model, steps=2000):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Training on RLC physics...")

    for step in range(steps):
        x = generate_RLC(8)
        pred = model(x)
        loss = ((pred - x)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"[{step}] Loss = {loss.item():.6f}")

    print("Training Done!")

# =====================================================
# 9. VISUALIZATION
# =====================================================
def visualize_physics(model):
    x = generate_RLC(1)
    pred = model(x).detach()

    gt = x[0,:,0].numpy()
    pr = pred[0,:,0].numpy()

    plt.plot(gt, label="Ground Truth", linewidth=2)
    plt.plot(pr, "--", label="Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()

# =====================================================
# 10. MAIN
# =====================================================
if __name__ == "__main__":
    model = QuantumTransformer(dim=128, depth=8, heads=8)
    train_physics(model, steps=1000)
    visualize_physics(model)
