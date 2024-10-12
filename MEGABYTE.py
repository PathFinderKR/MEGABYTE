import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CausalSelfAttention(nn.Module):
    def __init__(self, d_embed: int, n_heads: int, d_head: int):
        super().__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_head

        self.query = nn.Linear(self.d_embed, self.n_heads * self.d_head, bias=False)
        self.key = nn.Linear(self.d_embed, self.n_heads * self.d_head, bias=False)
        self.value = nn.Linear(self.d_embed, self.n_heads * self.d_head, bias=False)
        self.out = nn.Linear(self.n_heads * self.d_head, self.d_embed, bias=False)

    def forward(self, x):  # [batch_size, context_size, d_embed]
        batch_size, context_size, _ = x.size()
        q = self.query(x).view(batch_size, context_size, self.n_heads,
                               self.d_head)  # [batch_size, context_size, n_heads, d_head]
        k = self.key(x).view(batch_size, context_size, self.n_heads,
                             self.d_head)  # [batch_size, context_size, n_heads, d_head]
        v = self.value(x).view(batch_size, context_size, self.n_heads,
                               self.d_head)  # [batch_size, context_size, n_heads, d_head]

        q = q.transpose(1, 2)  # [batch_size, n_heads, context_size, d_head]
        k = k.transpose(1, 2)  # [batch_size, n_heads, context_size, d_head]
        v = v.transpose(1, 2)  # [batch_size, n_heads, context_size, d_head]

        # Masked Self Attention
        mask = torch.triu(torch.ones(context_size, context_size, device=x.device),
                          diagonal=1).bool()  # [context_size, context_size]
        mask = mask.view(1, 1, context_size, context_size)  # [1, 1, context_size, context_size]
        mask = mask.repeat(batch_size, self.n_heads, 1, 1)  # [batch_size, n_heads, context_size, context_size]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.d_head ** 0.5)  # [batch_size, n_heads, context_size, context_size]
        scores = scores.masked_fill(mask, float('-inf'))  # [batch_size, n_heads, context_size, context_size]
        scores = F.softmax(scores, dim=-1)  # [batch_size, n_heads, context_size, context_size]

        x = torch.matmul(scores, v)  # [batch_size, n_heads, context_size, d_head]
        x = x.transpose(1, 2).contiguous().view(batch_size, context_size,
                                                self.n_heads * self.d_head)  # [batch_size, context_size, n_heads * d_head]
        x = self.out(x)  # [batch_size, context_size, d_embed]
        return x


# %%
class MLP(nn.Module):
    def __init__(self, d_embed: int, d_ff: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(d_embed, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_embed, bias=False)

    def forward(self, x):  # [batch_size, context_size, d_embed]
        x = F.gelu(self.fc1(x))  # [batch_size, context_size, d_ff]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)  # [batch_size, context_size, d_embed]
        return x


# %%
class Decoder(nn.Module):
    def __init__(self, n_heads: int, d_head: int, d_embed: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = CausalSelfAttention(d_embed=d_embed, n_heads=n_heads, d_head=d_head)
        self.norm1 = nn.LayerNorm(d_embed)

        self.mlp = MLP(d_embed=d_embed, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_embed)

    def forward(self,
                x):  # [batch_size, context_size, d_embed], [batch_size, num_patches, patch_size * d_embed], [batch_size * num_patches, patch_size, local_d_embed]
        x = x + self.attention(self.norm1(
            x))  # [batch_size, context_size, d_embed], [batch_size, num_patches, patch_size * d_embed], [batch_size * num_patches, patch_size, local_d_embed]
        x = x + self.mlp(self.norm2(
            x))  # [batch_size, context_size, d_embed], [batch_size, num_patches, patch_size * d_embed], [batch_size * num_patches, patch_size, local_d_embed]
        return x


# %% md
## MEGABYTE
# %%
class PatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.V, config.d_G)
        self.positional_embedding = nn.Embedding(config.T, config.d_G)

    def forward(self, bytes):  # [batch_size, context_size]
        assert self.config.T % self.config.P == 0, "context size must be divisible by patch size"

        bytes = self.embedding(bytes) + self.positional_embedding(
            torch.arange(self.config.T, device=bytes.device))  # [batch_size, context_size, d_embed]
        bytes = rearrange(bytes, "b (k p) d -> b k (p d)", b=bytes.shape[0], k=self.config.K, p=self.config.P,
                          d=self.config.d_G)  # [batch_size, num_patches, patch_size * d_embed]
        return bytes


# %%
class GlobalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedder = PatchEmbedder(config)
        self.decoder = Decoder(n_heads=config.n_heads_G, d_head=config.d_head_G, d_embed=config.P * config.d_G,
                               d_ff=config.d_ff_G, dropout=config.dropout_G)
        self.linear = nn.Linear(config.d_G, config.d_L, bias=False)

    def forward(self, bytes):  # [batch_size, context_size]
        x = self.patch_embedder(bytes)  # [batch_size, num_patches, patch_size * d_embed]
        for _ in range(self.config.n_layers_G):
            x = self.decoder(x)  # [batch_size, num_patches, patch_size * d_embed]
        x = rearrange(x, "b k (p d) -> (b k) p d", b=bytes.shape[0], k=self.config.K, p=self.config.P,
                      d=self.config.d_G)  # [batch_size * num_patches, patch_size, d_embed]
        x = self.linear(x)  # [batch_size * num_patches, patch_size, local_d_embed]
        return x


# %%
class LocalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.V, config.d_L)
        self.local_transformer = Decoder(n_heads=config.n_heads_L, d_head=config.d_head_L, d_embed=config.d_L,
                                         d_ff=config.d_ff_L, dropout=config.dropout_L)
        self.linear = nn.Linear(config.d_L, config.V, bias=False)

    def forward(self, local_input,
                global_output):  # [batch_size * num_patches, patch_size], [batch_size * num_patches, patch_size, local_d_embed]
        x = self.embedding(local_input) + global_output  # [batch_size * num_patches, patch_size, local_d_embed]
        for _ in range(self.config.n_layers_L):
            x = self.local_transformer(x)  # [batch_size * num_patches, patch_size, local_d_embed]
        x = self.linear(x)  # [batch_size * num_patches, patch_size, vocab_size]
        x = rearrange(x, "(b k) p v -> b (k p) v", k=self.config.K, p=self.config.P,
                      v=self.config.V)  # [batch_size, context_size, vocab_size]
        return x


# %%
class MEGABYTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.global_model = GlobalModel(config)
        self.local_model = LocalModel(config)
        self.max_len = config.max_len
        self.context_size = config.T
        self.temperature = config.temperature

    def forward(self, bytes):  # [batch_size, context_size]
        global_input, local_input = self.prepare_input(
            bytes)  # [batch_size, context_size], [batch_size * num_patches, patch_size]
        global_output = self.global_model(global_input)  # [batch_size * num_patches, patch_size, local_d_embed]
        local_output = self.local_model(local_input, global_output)  # [batch_size, context_size, vocab_size]
        return local_output

    def prepare_input(self, bytes):  # [batch_size, context_size]
        global_padding = bytes.new(bytes.shape[0], self.config.P).fill_(self.config.PAD_ID)  # [batch_size, patch_size]
        global_input = torch.cat((global_padding, bytes[:, :-self.config.P]), dim=-1)  # [batch_size, context_size]

        bytes_input = rearrange(bytes, "b (k p) -> (b k) p", p=self.config.P)  # [batch_size * num_patches, patch_size]
        local_padding = bytes_input.new(bytes_input.shape[0], 1).fill_(self.config.PAD_ID)  # [patch_size]
        local_input = torch.cat((local_padding, bytes_input[:, :-1]), dim=-1)  # [batch_size * num_patches, patch_size]
        return global_input, local_input

    def loss(self, bytes, y):  # y: [batch_size, context_size]
        y = rearrange(y, "b t -> (b t)")  # [batch_size * context_size]
        logits = self.forward(bytes)  # [batch_size, context_size, vocab_size]
        logits = rearrange(logits, "b t v -> (b t) v", v=self.config.V)  # [batch_size * context_size, vocab_size]
        return F.cross_entropy(logits, y, ignore_index=self.config.PAD_ID)

    @torch.no_grad()
    def generate(self, bytes, max_len=None, decode_fn=None):
        self.eval()
        if max_len is None:
            max_len = self.max_len

        for _ in range(max_len - bytes.size(1)):  # x: [batch_size, context]
            context = bytes[:, -self.context_size:]  # [batch_size, context_size]
            output = self.forward(context)  # [batch_size, context_size, vocab_size]
            logits = output[:, -1, :] / self.temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)  # [batch_size]
            bytes = torch.cat((bytes, next_token.unsqueeze(-1)), dim=-1)  # [batch_size, context]

            # Decode token
            if decode_fn is not None:
                decoded_token = decode_fn([next_token[0].item()])
                print(decoded_token, end='', flush=True)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
