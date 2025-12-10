# MIRAS Implementation Spec

> **Goal**: Implement a minimal, educational reproduction of the MIRAS framework from "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization" (Behrouz et al. 2025), building on concepts from the TITANS paper.

## Overview

MIRAS (meaning "Legacy" in Persian/Arabic/Turkish) is a framework for designing sequence models based on four fundamental design choices:

1. **Memory Architecture** — Vector, matrix, or deep MLP
2. **Attentional Bias** — The internal objective function that defines "similarity"
3. **Retention Gate** — How to balance learning new concepts vs. retaining old ones
4. **Memory Learning Algorithm** — The optimizer (GD, GD+momentum, etc.)

This spec outlines an incremental, tutorial-style implementation similar to `007_transformer_together.ipynb`.

---

## Part 1: Theoretical Foundation

### 1.1 Core Concept: Associative Memory with Attentional Bias

The fundamental insight of MIRAS is that most sequence models can be viewed as **associative memory modules** that learn a mapping from keys to values:

$$
M^* = \arg\min_M \mathcal{L}(M(K); V)
$$

Where:
- $K \subseteq \mathbb{R}^{d_k}$ are keys (projections of input)
- $V \subseteq \mathbb{R}^{d_v}$ are values (projections of input)
- $\mathcal{L}$ is the **attentional bias** (internal objective function)
- $M$ is the memory module (parameterized or non-parametric)

### 1.2 Two Optimization Viewpoints

**Viewpoint 1: Follow-The-Regularized-Leader (FTRL)**
$$
W_t = \arg\min_W \sum_{i=1}^{t} \hat{\ell}_i(W; k_i, v_i) + \frac{1}{\eta_t} R_t(W)
$$

**Viewpoint 2: Learning-Retaining**
$$
W_t = \arg\min_W \tilde{\ell}_t(W; k_t, v_t) + \text{Ret}_t(W, W_{t-1})
$$

The Learning-Retaining viewpoint is more general and intuitive:
- First term: Learn from new data
- Second term: Retain previously learned knowledge

### 1.3 Unifying Existing Architectures

| Model | Memory | Attentional Bias | Retention | Algorithm |
|-------|--------|-----------------|-----------|-----------|
| Linear Attention | Matrix | Dot-Product | - | GD |
| RetNet | Vector | Dot-Product | ℓ₂ | GD |
| Mamba | Vector | Dot-Product | ℓ₂ | GD |
| Mamba-2 | Matrix | Dot-Product | ℓ₂ | GD |
| DeltaNet | Matrix | ℓ₂ | - | GD |
| Transformer | Non-parametric | Dot-Product | - | Implicit |
| TTT-Linear | Matrix | ℓ₂ | - | GD |
| TTT-MLP | 2-layer MLP | ℓ₂ | - | GD |
| Titans-LMM | k-layer MLP | ℓ₂ | ℓ₂ | GD+Momentum |
| Longhorn | Matrix | ℓ₂ | - | Implicit GD |
| HGRN2 | Matrix | ℓ₁ | - | GD |
| Gated DeltaNet | Matrix | ℓ₂ | ℓ₂ | GD |
| RWKV-7 | Matrix | ℓ₂ | ℓ₂ | GD |
| DeltaProduct | Matrix | ℓ₂ | ℓ₂ | MGD* |
| GLA | Matrix | Dot-Product | ℓ₂ | GD |
| Lightning Attention | Matrix | Dot-Product | ℓ₂ | GD |
| DFW | Matrix | ℓ₂ | - | Implicit GD |

> **Note**: "-" means no retention (memory grows indefinitely). "MGD*" = multiple rounds of GD per token. GLA = Gated Linear Attention. DFW = Data-Dependent Fast Weight.

---

## Part 2: Implementation Roadmap

### Phase 1: Simplest Possible Memory (Linear + ℓ₂)

Start with the simplest case to build intuition.

#### 2.1 Linear Memory with Hebbian Update (Linear Attention)

**Memory**: $M \in \mathbb{R}^{d \times d}$ (matrix)

**Attentional Bias**: Dot-product similarity $\tilde{\ell}_t = -2\langle Mk_t, v_t \rangle$

**Retention**: $\text{Ret}_t(M, M_{t-1}) = \|M - \alpha M_{t-1}\|_F^2$

**Update Rule** (via gradient descent):
```
M_t = α * M_{t-1} + v_t @ k_t.T
```

This is equivalent to **Linear Attention** / **RetNet**.

#### 2.2 Linear Memory with Delta Rule (DeltaNet)

**Attentional Bias**: ℓ₂ regression $\ell(M; k_t, v_t) = \|Mk_t - v_t\|_2^2$

**Update Rule**:
```
M_t = α * (I - η * k_t @ k_t.T) @ M_{t-1} + v_t @ k_t.T
```

This is the **Delta Rule** — it removes old value before writing new one.

> **Note**: The learning rate η only appears in the "erase" term `(I - η * k_t @ k_t.T)`, NOT in the "write" term `v_t @ k_t.T`.

---

### Phase 2: Adding Depth and Momentum (Titans-style)

#### 2.3 Deep Memory Module

Replace linear memory with a 2-layer MLP (MIRAS Equation 5, **post-norm** architecture):
$$
M(x) = x + \mathrm{LayerNorm}(W_1 \sigma(W_2 x))
$$

where $W_2 \in \mathbb{R}^{h \times d}$ projects up, $W_1 \in \mathbb{R}^{d \times h}$ projects down, and $\sigma$ is GELU activation.

**Why deep?** Linear memory assumes linear dependencies. Deep memory can learn non-linear patterns in the key-value mapping.

**Gradient computation**: For deep memory, use PyTorch autograd to compute gradients:
```python
def compute_memory_grad(memory, k, v, loss_fn):
    """Compute gradient of loss w.r.t. memory parameters using autograd."""
    pred = memory(k)
    loss = loss_fn(pred, v)
    return torch.autograd.grad(loss, memory.parameters())
```

#### 2.4 Adding Momentum (Surprise Metric)

TITANS insight: An event that violates expectations is more memorable.

**Surprise = gradient** of the loss w.r.t. input.

**Problem**: Momentary surprise can miss important info after a big surprise.

**Solution**: Track "past surprise" with momentum:
```
S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; k_t, v_t)
M_t = (1 - α_t) * M_{t-1} + S_t
```

This is **gradient descent with momentum and weight decay**.

#### 2.5 Optional Architectural Enhancements

The following components appear in production implementations (Titans, MIRAS) but are not essential for understanding the core concepts:

**Convolutional pre-processing**: A 1D convolution before memory access helps capture local patterns:
```python
self.conv = nn.Conv1d(d, d, kernel_size=4, padding=3, groups=d)
x = self.conv(x.transpose(1, 2)).transpose(1, 2)[:, :T]
```

**L2 normalization on Q/K**: Stabilizes attention-like computations:
```python
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
```

**RoPE (Rotary Position Embeddings)**: For position-aware memory, though pure associative memory is position-agnostic.

**Gating mechanism**: Output gating with sigmoid:
```python
gate = torch.sigmoid(self.gate_proj(x))
output = gate * memory_output + (1 - gate) * x
```

**Channel-wise parameters**: α, η as per-dimension vectors for finer control:
```python
self.alpha = nn.Parameter(torch.ones(d) * 0.9)  # [d] not scalar
self.eta = nn.Parameter(torch.ones(d) * 0.1)
```

**RMSNorm + SiLU**: Common in modern architectures:
```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale
```

> **Note**: This spec focuses on the minimal core. Add these enhancements incrementally when building production systems.

---

### Phase 3: Novel Attentional Biases (MIRAS)

#### 3.1 ℓₚ Attentional Bias

**Motivation**: ℓ₂ is sensitive to noise. Other p-norms may be more robust.

$$
\mathcal{L}(M(k_t); v_t) = \|M(k_t) - v_t\|_p^p
$$

**Gradient**:
```python
grad = p * sign(M @ k - v) * |M @ k - v|^(p-1) @ k.T
```

**Special case p=1**: "Value-less" memory — stores only ±1, like a coping mechanism for extreme events.

**Smooth approximations** (for backprop stability):
```python
sign(x) ≈ tanh(α * x)
|x| = sqrt(x² + ε)
```

#### 3.2 Huber Attentional Bias (Yaad)

**Motivation**: Robust to outliers, like human memory's coping mechanism.

**Definition**:
$$
H(a) = \begin{cases}
\frac{1}{2}a^2 & \text{if } |a| \leq \delta \\
\delta(|a| - \frac{1}{2}\delta) & \text{if } |a| > \delta
\end{cases}
$$

**Practical formulation** (smooth mixture):
```python
if ||M(k_t) - v_t|| <= δ_t:
    use ℓ₂ gradient
else:
    use δ_t * ℓ₁ gradient
```

The threshold $\delta_t$ is **data-dependent** (learned).

#### 3.3 Robust to Value Shifts

**Motivation**: Memory that performs well even under perturbations.

$$
\mathcal{L} = \max_{\|\delta_{v_t}\|_2 \leq \Delta} \frac{1}{2}\|M(k_t) - (v_t + \delta_{v_t})\|_2^2
$$

Solving the inner max:
$$
\mathcal{L} = \frac{1}{2}\|M(k_t) - v_t\|_2^2 + \Delta\|M(k_t) - v_t\|_2 + \frac{1}{2}\Delta^2
$$

---

### Phase 4: Novel Retention Gates (MIRAS)

#### 4.1 f-Divergence / KL Retention (Memora)

**Motivation**: Constrain memory to a scaled probability simplex for numerical stability.

**KL Divergence Retention**:
$$
\text{Ret}_t(W, W_{t-1}) = \frac{1}{\eta_t}\sum_{jl} W_{jl} \log\frac{W_{jl}}{(W_{t-1})_{jl}} + \frac{1}{\alpha_t}\sum_{jl} W_{jl}\log(W_{jl})
$$

**Update Rule**:
```python
W_t = c * Softmax((1 - λ_t) * log(W_{t-1}) - η'_t * ∇ℓ(W_{t-1}; k_t, v_t))
```

The Softmax ensures memory stays in valid range.

#### 4.2 ℓq Memory Stability (Moneta)

**Motivation**: Generalize ℓ₂ regularization.

**Update Rule**:
```python
A_t = α_t * A_{t-1} - η * ∇ℓ(W_{t-1}; k_t, v_t)
W_t = A_t / ||A_t||^((q-2)/q)
```

Where $p = \frac{q}{q-1}$.

#### 4.3 Elastic Net (Hard + Soft Forgetting)

**Motivation**: Combine feature selection (ℓ₁) with bias reduction (ℓ₂).

**Update Rule**:
```python
W_t = soft_threshold(λ * W_{t-1} - ζ * ∇ℓ(...), γ)
```

Where `soft_threshold(z, γ) = sign(z) * max(0, |z| - γ)`.

**Implementation**:
```python
def soft_threshold(z, gamma):
    """Proximal operator for ℓ₁ regularization (soft thresholding)."""
    return torch.sign(z) * F.relu(torch.abs(z) - gamma)

def elastic_net_update(W, grad, lambda_decay, zeta_lr, gamma_l1):
    """Elastic net update combining ℓ₂ decay with ℓ₁ sparsity.

    Args:
        W: Current weights
        grad: Gradient of loss
        lambda_decay: Weight decay factor (ℓ₂ retention)
        zeta_lr: Learning rate
        gamma_l1: ℓ₁ threshold (sparsity)
    """
    return soft_threshold(lambda_decay * W - zeta_lr * grad, gamma_l1)
```

---

## Part 3: Three Novel Models

### Model 1: Moneta (p,q-Moneta)

**Configuration**:
- Memory: 2-layer MLP with GELU, residual + LayerNorm
- Attentional Bias: ℓₚ (p=3)
- Retention: ℓq (q=4) + ℓ₂
- Algorithm: Gradient Descent

**Update Rules**:
```python
A_t = α_t * A_{t-1} - η_t * ∇ℓ_p(W_{t-1}; k_t, v_t)
W_t = A_t / ||A_t||^((q-2)/q)
```

**Gradient** (for MLP, per layer):
```python
∇ℓ_p = p * (sign(M @ k - v) ⊙ |M @ k - v|^(p-1)) @ k.T
```

### Model 2: Yaad (Robust Memory with Coping)

**Configuration**:
- Memory: 2-layer MLP with GELU, residual + LayerNorm
- Attentional Bias: Huber loss
- Retention: ℓ₂ local + global (Titans-style)
- Algorithm: Gradient Descent

**Update Rule**:
```python
if ||M(k_t) - v_t|| <= δ_t:
    W_t = α_t * W_{t-1} - η_t * ∇ℓ_2(W_{t-1}; k_t, v_t)
else:
    W_t = α_t * W_{t-1} - η_t * δ_t * ∇ℓ_1(W_{t-1}; k_t, v_t)
```

### Model 3: Memora (Entropy-Regularized Memory)

**Configuration**:
- Memory: 2-layer MLP with GELU, residual + LayerNorm
- Attentional Bias: ℓ₂
- Retention: KL divergence with scaling constant $c$
- Algorithm: Gradient Descent

**Update Rule** (MIRAS Equation 27):
```python
W_t = c * Softmax(α_t * log(W_{t-1}) - η_t * ∇ℓ_2(W_{t-1}; k_t, v_t))
```

> **Note**: The scaling constant $c$ is a learnable or fixed hyperparameter that controls the magnitude of the memory weights. This ensures the softmax-normalized weights have appropriate scale for downstream computation.

---

## Part 4: Parallelizable Training

### 4.1 Chunked Gradient Descent

The recurrent update can be parallelized within chunks:

1. Split sequence into chunks of size $b$ (e.g., 16 or 64)
2. Within each chunk, use the **same** memory state for all gradients
3. Expand the recurrence:

$$
M_t = \beta_t M_0 - \sum_{i=1}^{t} \theta_i \frac{\beta_t}{\beta_i} \nabla\ell(M_{t'}; x_i)
$$

Where $t' = t - \mod(t, b)$ and $\beta_i = \prod_{j=1}^{i}(1-\alpha_j)$.

### 4.2 Matrix Form (for GPU efficiency)

For linear memory:
```python
Θ_b = diag([θ_1, θ_2, ..., θ_b])
B_b = diag([β_b/β_1, β_b/β_2, ..., β_b/β_b])

# Batched gradient computation
grads = (W_0 @ K - V) @ K.T  # All gradients at once
weighted_grads = Θ_b @ B_b @ grads

M_b = β_b * M_0 - weighted_grads
```

### 4.3 Handling Momentum

Within each chunk, momentum is a linear recurrence:
```python
S_t = η_t * S_{t-1} - θ_t * u_t  # where u_t = ∇ℓ(...)
```

Use **parallel associative scan** to compute all $S_t$ in O(log b) parallel steps.

### 4.4 Non-linear Components

For deep memory or non-linear retention (like Memora's Softmax):
- Use linear approximation **within** each chunk
- Apply non-linearity **between** chunks

**Lag Token Strategy for Memora**: Memora's KL retention uses softmax which breaks the linear recurrence structure needed for parallelization. The solution:
1. Within each chunk, accumulate updates in log-domain: $\log W_t = \alpha_t \log W_{t-1} - \eta_t \nabla\ell$
2. At chunk boundaries, insert a "lag token" — a dedicated position that:
   - Applies `exp()` to materialize the actual weights
   - Normalizes via softmax to keep weights on probability simplex
   - Converts back to log-domain for the next chunk
3. This allows linear parallel computation within chunks while preserving Memora's entropy regularization across chunks

---

## Part 5: Implementation Steps (Notebook Cells)

### Cell 1: Setup & Data
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load tiny shakespeare or similar
# Tokenize with character-level encoding
# Create batches
```

### Cell 2: Key-Value Projections
```python
class KeyValueProjection(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_K = nn.Linear(d_in, d_out)
        self.W_V = nn.Linear(d_in, d_out)
        self.W_Q = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.W_K(x), self.W_V(x), self.W_Q(x)
```

### Cell 3: Linear Memory (Hebbian)
```python
class LinearMemoryHebbian(nn.Module):
    """Simplest associative memory - equivalent to Linear Attention."""
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, keys, values, queries, alpha=0.9):
        B, T, D = keys.shape
        M = torch.zeros(B, D, D, device=keys.device)
        outputs = []

        for t in range(T):
            k_t, v_t, q_t = keys[:, t], values[:, t], queries[:, t]
            M = alpha * M + torch.einsum('bd,be->bde', v_t, k_t)
            y_t = torch.einsum('bde,be->bd', M, q_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
```

### Cell 4: Linear Memory (Delta Rule)
```python
class LinearMemoryDelta(nn.Module):
    """Delta rule - removes old value before writing new.

    M_t = α * (I - η * k_t @ k_t.T) @ M_{t-1} + v_t @ k_t.T

    The (I - η * k_t @ k_t.T) term "erases" the old value associated with k_t
    before writing the new association v_t @ k_t.T.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, keys, values, queries, alpha=0.9, eta=0.1):
        B, T, D = keys.shape
        M = torch.zeros(B, D, D, device=keys.device)
        outputs = []

        for t in range(T):
            k_t, v_t, q_t = keys[:, t], values[:, t], queries[:, t]

            # Delta rule: erase old value, then write new
            kk = torch.einsum('bd,be->bde', k_t, k_t)  # [B, D, D]
            # Identity needs batch dimension for broadcasting
            I = torch.eye(D, device=M.device).unsqueeze(0)  # [1, D, D]
            M = alpha * torch.bmm(I - eta * kk, M)  # Batched matmul
            M = M + torch.einsum('bd,be->bde', v_t, k_t)  # Note: No eta here!

            y_t = torch.einsum('bde,be->bd', M, q_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
```

### Cell 5: Deep Memory Module
```python
class DeepMemory(nn.Module):
    """2-layer MLP memory as in Titans/MIRAS (post-norm architecture per MIRAS Eq. 5).

    M(x) = x + LayerNorm(W1 @ σ(W2 @ x))
    """
    def __init__(self, d, expansion=4):
        super().__init__()
        self.W2 = nn.Linear(d, d * expansion, bias=False)      # Up projection
        self.W1 = nn.Linear(d * expansion, d, bias=False)      # Down projection
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        # Post-norm: MLP first, then LayerNorm, then residual
        h = F.gelu(self.W2(x))      # h = σ(W2 @ x)
        return x + self.ln(self.W1(h))  # x + LN(W1 @ h)
```

### Cell 6: Attentional Bias Objectives
```python
def l2_loss(pred, target):
    """Standard ℓ₂ loss."""
    return 0.5 * ((pred - target) ** 2).sum(dim=-1)

def lp_loss(pred, target, p=3):
    """ℓₚ loss - more robust for p < 2."""
    return (torch.abs(pred - target) ** p).sum(dim=-1)

def huber_loss(pred, target, delta):
    """Huber loss - robust to outliers."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = 0.5 * diff ** 2
    linear = delta * (abs_diff - 0.5 * delta)
    return torch.where(abs_diff <= delta, quadratic, linear).sum(dim=-1)
```

### Cell 7: Gradient Computation for Memory Update
```python
def compute_memory_update(memory, k, v, loss_fn, eta):
    """Compute gradient-based update for memory parameters.

    Uses autograd to handle both linear and deep memory architectures.
    Returns a function that applies the update.
    """
    pred = memory(k)
    loss = loss_fn(pred, v)

    # Compute gradients w.r.t. memory parameters
    grads = torch.autograd.grad(loss.sum(), memory.parameters(), create_graph=False)
    return grads

def apply_memory_update(memory, grads, alpha, eta):
    """Apply gradient update with retention (weight decay)."""
    with torch.no_grad():
        for param, grad in zip(memory.parameters(), grads):
            param.mul_(alpha).sub_(eta * grad)

# For linear memory, the closed-form gradient is simpler:
def l2_gradient_linear(M, k, v):
    """Gradient of ℓ₂ loss for linear memory: (Mk - v) @ k.T"""
    pred = M @ k.unsqueeze(-1)  # [B, D, 1]
    error = pred.squeeze(-1) - v  # [B, D]
    return error.unsqueeze(-1) @ k.unsqueeze(-2)  # [B, D, D]

def lp_gradient_linear(M, k, v, p=3, eps=1e-6):
    """Gradient of ℓₚ loss for linear memory with smooth approximations."""
    pred = M @ k.unsqueeze(-1)
    diff = pred.squeeze(-1) - v
    sign_diff = torch.tanh(100 * diff)  # Smooth sign
    abs_diff = torch.sqrt(diff ** 2 + eps)
    return p * (sign_diff * abs_diff ** (p-1)).unsqueeze(-1) @ k.unsqueeze(-2)
```

### Cell 8: Retention Gates
```python
def l2_retention(W_prev, alpha):
    """Standard ℓ₂ retention (weight decay).

    Note: This returns the decayed weights, NOT a retention penalty.
    The name refers to the fact that ℓ₂ regularization induces weight decay.
    """
    return alpha * W_prev

def kl_retention(W_prev, grad, alpha, eta, c=1.0):
    """KL divergence retention (Memora) per MIRAS Equation 27.

    W_t = c * Softmax(α * log(W_{t-1}) - η * ∇ℓ)

    Args:
        W_prev: Previous weights (must be positive for log)
        grad: Gradient of loss
        alpha: Retention strength (1 - λ_t in paper)
        eta: Learning rate (η'_t in paper)
        c: Scaling constant for output magnitude
    """
    log_W = torch.log(W_prev.clamp(min=1e-10))
    return c * F.softmax(alpha * log_W - eta * grad, dim=-1)
```

### Cell 9: Moneta Implementation
```python
class Moneta(nn.Module):
    """ℓₚ attentional bias + ℓq retention.

    Uses a functional approach: maintains memory state explicitly
    rather than modifying nn.Module parameters in-place.
    """
    def __init__(self, d, expansion=4, p=3, q=4):
        super().__init__()
        self.d = d
        self.expansion = expansion
        self.p = p
        self.q = q
        self.kv_proj = KeyValueProjection(d, d)

        # Initial memory weights (will be copied and updated per sequence)
        self.W1_init = nn.Parameter(torch.randn(d, d * expansion) * 0.02)
        self.W2_init = nn.Parameter(torch.randn(d * expansion, d) * 0.02)
        self.ln = nn.LayerNorm(d)

    def memory_forward(self, x, W1, W2):
        """Functional memory forward pass."""
        h = F.gelu(x @ W2.T)
        return x + self.ln(h @ W1.T)

    def forward(self, x, alpha, eta):
        k, v, q = self.kv_proj(x)
        B, T, D = k.shape

        # Initialize accumulators for ℓq normalization
        A1 = self.W1_init.clone().unsqueeze(0).expand(B, -1, -1)
        A2 = self.W2_init.clone().unsqueeze(0).expand(B, -1, -1)
        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]

            # Current normalized weights
            norm1 = torch.norm(A1, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            norm2 = torch.norm(A2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            W1 = A1 / (norm1 ** ((self.q - 2) / self.q))
            W2 = A2 / (norm2 ** ((self.q - 2) / self.q))

            # Compute ℓₚ loss and gradient via autograd
            W1_leaf = W1.detach().requires_grad_(True)
            W2_leaf = W2.detach().requires_grad_(True)
            pred = self.memory_forward(k_t, W1_leaf, W2_leaf)
            loss = (torch.abs(pred - v_t) ** self.p).sum()
            grad1, grad2 = torch.autograd.grad(loss, [W1_leaf, W2_leaf])

            # Update accumulators
            A1 = alpha * A1 - eta * grad1
            A2 = alpha * A2 - eta * grad2

            # Output uses normalized weights
            y_t = self.memory_forward(q_t, W1, W2)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
```

### Cell 10: Yaad Implementation
```python
class Yaad(nn.Module):
    """Huber attentional bias - robust to outliers.

    Switches between ℓ₂ and ℓ₁ gradients based on error magnitude,
    providing robustness similar to human memory's coping mechanisms.
    """
    def __init__(self, d, expansion=4):
        super().__init__()
        self.d = d
        self.kv_proj = KeyValueProjection(d, d)
        self.delta_proj = nn.Linear(d, 1)  # Data-dependent threshold

        # Functional memory weights
        self.W1_init = nn.Parameter(torch.randn(d, d * expansion) * 0.02)
        self.W2_init = nn.Parameter(torch.randn(d * expansion, d) * 0.02)
        self.ln = nn.LayerNorm(d)

    def memory_forward(self, x, W1, W2):
        """Functional memory forward pass."""
        h = F.gelu(x @ W2.T)
        return x + self.ln(h @ W1.T)

    def forward(self, x, alpha, eta):
        k, v, q = self.kv_proj(x)
        B, T, D = k.shape

        # Clone weights for this sequence
        W1 = self.W1_init.clone().unsqueeze(0).expand(B, -1, -1)
        W2 = self.W2_init.clone().unsqueeze(0).expand(B, -1, -1)
        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]

            # Compute prediction and error
            W1_leaf = W1.detach().requires_grad_(True)
            W2_leaf = W2.detach().requires_grad_(True)
            pred = self.memory_forward(k_t, W1_leaf, W2_leaf)
            error = pred - v_t
            error_norm = torch.norm(error, dim=-1, keepdim=True)

            # Data-dependent threshold
            delta_t = F.softplus(self.delta_proj(x[:, t]))

            # Huber loss: ℓ₂ for small errors, δ * ℓ₁ for large errors
            l2_loss = 0.5 * (error ** 2).sum()
            l1_loss = delta_t * torch.abs(error).sum()
            mask = (error_norm <= delta_t).float()
            loss = (mask * 0.5 * (error ** 2) + (1 - mask) * delta_t * (torch.abs(error) - 0.5 * delta_t)).sum()

            # Gradient via autograd
            grad1, grad2 = torch.autograd.grad(loss, [W1_leaf, W2_leaf])

            # Update weights with retention
            W1 = alpha * W1 - eta * grad1
            W2 = alpha * W2 - eta * grad2

            y_t = self.memory_forward(q_t, W1.detach(), W2.detach())
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
```

### Cell 11: Memora Implementation
```python
class Memora(nn.Module):
    """KL divergence retention - entropy-regularized memory (MIRAS Eq. 27).

    W_t = c * Softmax(α_t * log(W_{t-1}) - η_t * ∇ℓ₂)

    The softmax ensures weights stay on a scaled probability simplex,
    providing numerical stability. The scaling constant 'c' controls output magnitude.
    """
    def __init__(self, d, expansion=4, c=1.0):
        super().__init__()
        self.d = d
        self.kv_proj = KeyValueProjection(d, d)
        self.c = nn.Parameter(torch.tensor(c))
        self.ln = nn.LayerNorm(d)

        # Initialize weights on probability simplex (positive, normalized)
        W1_raw = torch.rand(d, d * expansion)
        W2_raw = torch.rand(d * expansion, d)
        self.register_buffer('W1_init', F.softmax(W1_raw, dim=-1))
        self.register_buffer('W2_init', F.softmax(W2_raw, dim=-1))

    def memory_forward(self, x, W1, W2):
        """Functional memory forward pass with scaled weights."""
        h = F.gelu(x @ (self.c * W2).T)
        return x + self.ln(h @ (self.c * W1).T)

    def forward(self, x, alpha, eta):
        k, v, q = self.kv_proj(x)
        B, T, D = k.shape

        # Work in log domain for numerical stability
        log_W1 = torch.log(self.W1_init.clamp(min=1e-10)).unsqueeze(0).expand(B, -1, -1)
        log_W2 = torch.log(self.W2_init.clamp(min=1e-10)).unsqueeze(0).expand(B, -1, -1)
        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]

            # Current weights via softmax
            W1 = self.c * F.softmax(log_W1, dim=-1)
            W2 = self.c * F.softmax(log_W2, dim=-1)

            # Compute ℓ₂ gradient via autograd
            W1_leaf = W1.detach().requires_grad_(True)
            W2_leaf = W2.detach().requires_grad_(True)
            pred = self.memory_forward(k_t, W1_leaf / self.c, W2_leaf / self.c)
            loss = 0.5 * ((pred - v_t) ** 2).sum()
            grad1, grad2 = torch.autograd.grad(loss, [W1_leaf, W2_leaf])

            # KL retention update in log domain
            log_W1 = alpha * log_W1 - eta * grad1
            log_W2 = alpha * log_W2 - eta * grad2

            # Output (softmax applied implicitly in memory_forward)
            W1_out = self.c * F.softmax(log_W1.detach(), dim=-1)
            W2_out = self.c * F.softmax(log_W2.detach(), dim=-1)
            y_t = self.memory_forward(q_t, W1_out / self.c, W2_out / self.c)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
```

### Cell 12: Full MIRAS Layer
```python
class MIRASLayer(nn.Module):
    """Complete MIRAS layer with configurable components."""
    def __init__(self, d,
                 memory_type='deep',        # 'linear', 'deep'
                 attentional_bias='l2',     # 'l2', 'lp', 'huber'
                 retention='l2',            # 'l2', 'kl', 'elastic'
                 p=3, q=4):
        super().__init__()
        self.d = d
        self.attentional_bias = attentional_bias
        self.retention = retention
        self.p, self.q = p, q

        # Memory architecture
        if memory_type == 'linear':
            self.memory = nn.Linear(d, d, bias=False)
        else:
            self.memory = DeepMemory(d)

        # Projections
        self.kv_proj = KeyValueProjection(d, d)

        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(d) * 0.9)
        self.eta = nn.Parameter(torch.ones(d) * 0.1)

    def forward(self, x):
        k, v, q = self.kv_proj(x)
        # ... implementation based on config
```

### Cell 13: Training Loop
```python
def train_miras(model, data, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch in data:
            x, targets = batch

            # Forward pass
            outputs = model(x)

            # Language modeling loss (outer loop)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Part 6: Experiments & Validation

### 6.1 Sanity Checks
1. Linear memory should reproduce Linear Attention behavior
2. Delta rule should show better memory management than Hebbian
3. Deep memory should outperform linear on non-linear patterns

### 6.2 Ablation Studies
1. Effect of p in ℓₚ attentional bias
2. Effect of q in ℓq retention
3. Deep vs. linear memory
4. With/without momentum

### 6.3 Comparison with Baselines
- Compare against simple Transformer (from 007 notebook)
- Compare against Mamba-style linear RNN
- Evaluate on:
  - Language modeling perplexity
  - Needle-in-haystack retrieval
  - Memory capacity tests

---

## Part 7: Key Insights & Takeaways

### Why MIRAS Matters

1. **Unification**: Shows that Transformers, RNNs, and SSMs are all associative memory variants
2. **Generalization**: The framework enables designing new architectures by mixing components
3. **Robustness**: Novel attentional biases (Huber, ℓₚ) handle outliers better
4. **Memory Management**: Novel retention gates provide better forgetting mechanisms

### Key Equations to Remember

**Hebbian (Linear Attention)**:
$$M_t = \alpha M_{t-1} + v_t k_t^\top$$

**Delta Rule**:
$$M_t = \alpha(I - \eta k_t k_t^\top) M_{t-1} + v_t k_t^\top$$

**Titans/MIRAS with Momentum**:
$$S_t = \eta_t S_{t-1} - \theta_t \nabla\ell(M_{t-1}; x_t)$$
$$M_t = (1 - \alpha_t) M_{t-1} + S_t$$

**Memora (KL Retention)**:
$$W_t = \text{Softmax}(\alpha_t \log(W_{t-1}) - \eta_t \nabla\ell_2)$$

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| Attentional Bias | Internal objective function defining "similarity" in memory |
| Retention Gate | Mechanism balancing new learning vs. old retention |
| Momentary Surprise | Gradient at current timestep |
| Past Surprise | Momentum carrying surprise across tokens |
| Hebbian Rule | Additive memory update: M += vk^T |
| Delta Rule | Replacement memory update: removes old before adding new |

---

## References

1. Behrouz et al. (2025). "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization" (MIRAS paper)
2. Behrouz et al. (2024). "Titans: Learning to Memorize at Test Time"
3. Yang et al. (2024). "Gated Delta Networks"
4. Sun et al. (2024). "TTT: Learning to (learn at test time)"
5. Katharopoulos et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
