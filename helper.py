import csv
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import grain.python as pygrain
import optax
import tiktoken
from pathlib import Path

tokenizer = tiktoken.get_encoding("gpt2")

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 6
maxlen = 128
embed_dim = 192
num_heads = 6
feed_forward_dim = int(2/3 * 4 * embed_dim)
batch_size = 24
num_epochs = 3

def load_stories_from_file(file_path, *, text_column="text", limit=None, max_stories=None):
    """Load TinyStories from a .txt file or a Kaggle-style CSV file."""
    path = Path(file_path)
    if limit is None:
        limit = max_stories

    if path.suffix == ".csv":
        stories = []
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(
                    f"Expected a CSV column named {text_column!r}; "
                    f"found {reader.fieldnames!r}."
                )

            for row in reader:
                story = row[text_column].strip()
                if story:
                    stories.append(story)
                    if limit is not None and len(stories) >= limit:
                        break
        return stories

    data = path.read_text(encoding="utf-8", errors="replace")
    stories = [story.strip() for story in data.split("<|endoftext|>") if story.strip()]
    return stories if limit is None else stories[:limit]


class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs
        )
        
    def __call__(self, x, mask=None):
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        return x

class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)

class MiniGPT(nnx.Module):

    def __init__(self, maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                 feed_forward_dim=feed_forward_dim, num_transformer_blocks=num_transformer_blocks, *, rngs=nnx.Rngs(0)):

        self.maxlen = maxlen

        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)

        self.transformer_blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ])

        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)
        
    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)

        return logits

def generate_text(
    model,
    start_tokens,
    max_new_tokens=50,
    temperature=0.8,
    seed=0,
    top_k=40,
    repetition_penalty=1.2,
):
    tokens = list(start_tokens)
    rng = jax.random.key(seed)
    end_token = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # RIGHT-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :]

        if tokens and repetition_penalty > 1.0:
            recent_tokens = jnp.array(tokens[-32:], dtype=jnp.int32)
            next_token_logits = next_token_logits.at[recent_tokens].add(
                -jnp.log(jnp.array(repetition_penalty, dtype=next_token_logits.dtype))
            )

        next_token_logits = next_token_logits / max(float(temperature), 1e-6)

        if top_k is not None and top_k > 0:
            values, indices = jax.lax.top_k(next_token_logits, min(top_k, next_token_logits.shape[-1]))
            rng, step_rng = jax.random.split(rng)
            sampled_index = int(jax.random.categorical(step_rng, values))
            next_token = int(indices[sampled_index])
        else:
            rng, step_rng = jax.random.split(rng)
            next_token = int(jax.random.categorical(step_rng, next_token_logits))

        if next_token == end_token:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)



def generate_story(model, story_prompt, temperature=0.8, max_new_tokens=80, seed=0):
    start_tokens = tokenizer.encode(story_prompt)[:maxlen]
    generated = generate_text(
        model,
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    return generated


class StoryDataset:
    def __init__(self, stories, maxlen, tokenizer, add_endoftext=True):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.add_endoftext = add_endoftext
        self.end_token = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        if self.add_endoftext and not story.endswith('<|endoftext|>'):
            story = story + '<|endoftext|>'
        tokens = self.tokenizer.encode(story, allowed_special={'<|endoftext|>'})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return jnp.array(tokens, dtype=jnp.int32)



def load_and_preprocess_data(
    file_path,
    batch_size,
    maxlen,
    max_stories = 100_000,
    num_epochs = 1,
    shuffle = False,
    seed = 42
):
    """
    Load and preprocess TinyStories data from .txt or Kaggle-style .csv files.

    Args:
        file_path: Path to the text or CSV file
        batch_size: Batch size for training
        maxlen: Maximum sequence length
        max_stories: Maximum number of stories to load (for memory efficiency)
        num_epochs: Number of training epochs
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Grain DataLoader, estimated_batches_per_epoch)
    """

    print(f"Loading data from {file_path} (max {max_stories:,} stories)")

    stories = load_stories_from_file(file_path, max_stories=max_stories)

    print(f"Loaded {len(stories):,} stories")
    if len(stories) == 0:
        raise ValueError("No valid stories found in the dataset")

    # Calculate estimated batches per epoch
    estimated_batches_per_epoch = len(stories) // batch_size
    print(f"Estimated batches per epoch: {estimated_batches_per_epoch:,}")

    # Create efficient dataset
    dataset = StoryDataset(stories, maxlen, tokenizer)

    # Configure sampler with sharding support
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    # Create DataLoader with efficient batching
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            pygrain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")
    return dataloader, estimated_batches_per_epoch
