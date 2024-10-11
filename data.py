from config import CONFIG
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset
# %% md
## Shakespeare
# %%
def load_shakespeare():
    with open(CONFIG.shakespeare_id, 'r') as f:
        shakespeare_text = f.read()
    return shakespeare_text

# %% md
## Wikipedia
# %%
# wiki_dataset = load_dataset(CONFIG.wiki_id, "20231101.en")
# %% md
# Tokenization
# %%
char2int = {chr(i): i for i in range(CONFIG.V)}
int2char = {i: chr(i) for i in range(CONFIG.V)}

# Add special tokens
char2int['<PAD>'] = CONFIG.PAD_ID
int2char[CONFIG.PAD_ID] = '<PAD>'
char2int['<EOS>'] = CONFIG.EOS_ID
int2char[CONFIG.EOS_ID] = '<EOS>'

# Encoding and Decoding
encode = lambda text: [char2int[c] for c in text]
decode = lambda tokens: ''.join([int2char[t] for t in tokens])

# %% md
# Preprocessing
# %%
def preprocess(dataset_id):
    if dataset_id == CONFIG.shakespeare_id:
        text = load_shakespeare()
    elif dataset_id == CONFIG.wiki_id:
        text = wiki_dataset['train']['text']
    else:
        raise ValueError("Invalid dataset id")
    tokens = torch.tensor(encode(text), dtype=torch.long)
    return tokens

# %%
def train_validation_split(tokens, validation_size):
    train_size = int(len(tokens) * (1 - validation_size))
    return tokens[:train_size], tokens[train_size:]

# %%
class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        return self.tokens[idx:idx + self.context_length], self.tokens[idx + 1:idx + self.context_length + 1]

# %%
def create_dataloader(train_tokens, validation_tokens, context_length, batch_size):
    train_dataset = TextDataset(train_tokens, context_length)
    validation_dataset = TextDataset(validation_tokens, context_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader
\