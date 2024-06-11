import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------Config for Debugging-------------------
batch_size = 4
block_size = 8
learning_rate = 3e-4
max_iters = 50
eval_interval = 5
eval_iters = 2
dropout = 0.2
n_layers = 2
n_heads = 4
n_embd = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------

# # -------------------------------
# batch_size = 64
# block_size = 256
# learning_rate = 3e-4
# max_iters = 5000
# eval_interval = 500
# eval_iters = 200
# dropout = 0.2
# n_layers = 6
# n_heads = 6
# n_embd = 384
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # -------------------------

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
    def __init__(self, dim, seq_len, theta: float=10000.0) -> None:
        super().__init__()
        # 10000 is the cycle length, expected to be 10k > seq_len
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim)) # (D/2,)
        t = torch.arange(seq_len)
        freqs = torch.einsum('i,j->ij', t, freqs) # (T, D/2)
        # torch.polar(abs, angle) -> (abs * cos(angle)+iabs * sin(angle))
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # (T, D/2)
        self.register_buffer('freqs_cis', freqs_cis)

    def apply_rotary_pos_emb(self, q, k):
        freqs_cis = self.freqs_cis[:q.shape[1]] # (T, D/2)
        
        q_ = q.float().reshape(*q.shape[:-1], -1, 2) # (B, T, D) -> (B, T, D/2, 2)
        k_ = k.float().reshape(*k.shape[:-1], -1, 2) # (B, T, D) -> (B, T, D/2, 2)
        
        # torch.view_as_complex((x, y)) -> x + iy
        q_ = torch.view_as_complex(q_)
        k_ = torch.view_as_complex(k_)
        
        # torch.view_as_real(x+iy) -> (x, y)
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(2) # (B, T, D/2, 2) -> (B, T, D)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(-2) # (B, T, D/2, 2) -> (B, T, D)
        return q_out.type_as(q), k_out.type_as(k)

class Head(nn.Module):
    """one head self-attention"""

    def __init__(self, head_size) -> None:
        super().__init__()
        # nn.Linear: 接受连续值的特征向量作为输入。
        # nn.Embedding: 接受离散的索引值作为输入。
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(head_size, block_size)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, C)
        k = self.key(x)
        v = self.value(x)

        # Apply RoPE to queries and keys
        q, k = self.rope.apply_rotary_pos_emb(q, k)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
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
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def greedy_sampling(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def beam_sampling(self, idx, max_new_tokens, beam_width=3):
        sequences = [(idx, 0.0)]
        for _ in range(max_new_tokens):
            all_candidates = []
            for seq, score in sequences:
                idx_cond = seq[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    candidate = torch.cat((seq, topk_indices[:, i].unsqueeze(-1)), dim=1)
                    candidate_score = score + topk_probs[:, i].item()
                    all_candidates.append((candidate, candidate_score))
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]
        return sequences[0][0]

    def nucleus_sampling(self, idx, max_new_tokens, p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
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

# generate from model using greedy sampling
context = torch.tensor(encode("The meaning of life is"), dtype=torch.long, device=device).unsqueeze(0)
print(decode(model.greedy_sampling(context, max_new_tokens=500)[0].tolist()))

# generate from model using beam sampling
context = torch.tensor(encode("The meaning of life is"), dtype=torch.long, device=device).unsqueeze(0)
print(decode(model.beam_sampling(context, max_new_tokens=500, beam_width=3)[0].tolist()))

# generate from model using nucleus sampling
context = torch.tensor(encode("The meaning of life is"), dtype=torch.long, device=device).unsqueeze(0)
print(decode(model.nucleus_sampling(context, max_new_tokens=500, p=0.9)[0].tolist()))