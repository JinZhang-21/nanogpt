import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
dropout = 0.2
n_layers = 6
n_heads = 6
n_embd = 384
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------

torch.manual_seed(43)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# --------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        theta = 10000.0
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('frequencies', frequencies)
    
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.frequencies.device).float()
        freqs = torch.einsum('i,j->ij', t, self.frequencies)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

    def apply_rotary_pos_emb(self, q, k):
        seq_len, dim = q.shape[1], q.shape[2]
        cos, sin = self(seq_len)[:, 0::2], self(seq_len)[:, 1::2]  # Separate sine and cosine
        cos, sin = cos.to(q.device), sin.to(q.device)

        # Expand cos and sin for each element in the batch and head dimension
        cos = cos.unsqueeze(0).repeat(q.shape[0], 1, 1)  # (B, T, D/2)
        sin = sin.unsqueeze(0).repeat(q.shape[0], 1, 1)  # (B, T, D/2)

        # Apply RoPE to q and k
        q_r, q_i = q[..., 0::2], q[..., 1::2]
        k_r, k_i = k[..., 0::2], k[..., 1::2]

        q_rot = q_r * cos + q_i * sin
        k_rot = k_r * cos - k_i * sin

        # Re-assemble the transformed q and k
        q_transformed = torch.stack([q_rot, q_i - q_rot * sin], dim=-1).reshape_as(q)
        k_transformed = torch.stack([k_rot, k_i - k_rot * sin], dim=-1).reshape_as(k)

        return q_transformed, k_transformed
    
class Head(nn.Module):
    """One head self-attention with KV-Cache."""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(head_size)
        
        # Cache for keys and values
        self.register_buffer("cache_k", torch.zeros(0, 0, head_size))
        self.register_buffer("cache_v", torch.zeros(0, 0, head_size))

    def forward(self, x, use_cache=True):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Apply RoPE to queries and keys
        q, k = self.rope.apply_rotary_pos_emb(q, k)

        # Concatenate with cache if use_cache is True
        if use_cache and self.cache_k.size(1) > 0:
            k = torch.cat([self.cache_k, k], dim=1)
            v = torch.cat([self.cache_v, v], dim=1)

        # Create dynamic triangular mask
        combined_len = k.shape[1]
        assert combined_len == v.shape[1] & combined_len == k.shape[1] # Check if the combined length is the same
        tril = torch.tril(torch.ones(combined_len, combined_len, device=x.device))

        # Debug print to check sizes
        # print(f"Q size: {q.shape}, K size: {k.shape}, Combined length: {combined_len}")
        
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(tril[:T, :combined_len] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        
        # Update cache
        self.cache_k = k
        self.cache_v = v
        
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel with KV-Cache."""
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache=True):
        out = torch.cat([head(x, use_cache) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # parameters trained with backprop
        self.gamma = nn.Parameter(torch.ones(dim).to(device))
        self.beta = nn.Parameter(torch.zeros(dim).to(device))
    
    def forward(self, x) :
        xmean = x.mean(1, keepdim=True) # mean along the batch dimension
        xvar = x.var(1, keepdim=True) # variance along the batch dimension
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    """transformer layer"""

    def __init__(self, n_embd, n_heads) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.ffn = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        out = self.mha(self.ln1(x)) + x
        out = self.ffn(self.ln1(x)) + x
        return out


class GPTLanguageModel(nn.Module):
    """gpt"""

    def __init__(self) -> None:
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T= idx.shape
        out = self.tok_embedding(idx)

        out = self.blocks(out)
        out = self.ln_f(out)
        logits = self.lm_head(out)

        if targets is None:
            loss = None
        else:
            # ! attention the shape here
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # idx_cond = idx[:, -block_size:, :]
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # idx = torch.cat((idx_cond, idx_next), dim=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = GPTLanguageModel().to(device)
print(f"model parameters:{sum(p.numel() for p in model.parameters())/1e6} M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
