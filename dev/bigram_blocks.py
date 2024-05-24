import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000 # for lowing the learning rate
eval_interval = 500
learning_rate = 1e-3 # self-attention model is very sensitive to the learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ---------------

torch.manual_seed(43)

with open('input.txt', encoding='utf-8') as f:
    text = f.read()

# encode decode moudule
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mapping func
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda x: ''.join([itos[i] for i in x])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # *因为tril不是一个参数，所以不会被优化，所以不需要注册为参数, 但是需要注册为buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        # attention scores
        wei = q @ k.transpose(-2, -1) * (1.0 / C**0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        # ! attention here is just a decoder block
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the upper triangular part
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        v = self.value(x) # (B, T, H)
        out = wei @ v # (B, T, T) @ (B, T, H) = (B, T, H)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, C)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = self.sa(x) + x
        x = self.ffwd(x) + x
        return x

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both of shape (B, T)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C=n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embed)
        x = tok_emb + pos_emb # (B, T, C=n_embed)
        x = self.blocks(x) # apply the single-head self-attention
        logits = self.lm_head(x) # (B, T, V=Vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #! Channel as second dimension for cross-entropy of pytorch
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        """generat from (B, T) to (B, T+max_new_tokens)"""
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # logits, loss = self(idx) # calling forward to get the logits
            # todo here we crop the idx in case the idx is longer than block_size, think about it[because of the position embedding is no more than block_size]
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # calling forward to get the logits
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel().to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter:5d}, Train loss: {losses["train"]:.4f}, Test loss: {losses["test"]:.4f}')
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # set_to_none: 当为True时，梯度张量将被置为None，而不是0
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, 1000)[0].tolist()))

