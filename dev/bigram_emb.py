import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
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

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both of shape (B, T)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C=n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embed)
        x = tok_emb + pos_emb # (B, T, C=n_embed)
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
            logits, loss = self(idx) # calling forward to get the logits
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

