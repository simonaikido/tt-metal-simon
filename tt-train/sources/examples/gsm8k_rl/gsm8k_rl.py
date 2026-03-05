from transformers import AutoTokenizer, AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import initialize_device, set_seed
import ttnn
import ttml
import os
import numpy as np
import datasets
import time
from typing import List, TypeAlias, Any

CONFIG = "training_gsm8k_rl_llama.yaml"
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
LOAD_PRETRAINED = True

from ttml.common.config import (
    load_config,
    yaml_deep_update,
)

from ttml.common.utils import (
    create_optimizer,
    get_tt_metal_home,
    no_grad,
)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
vocab_size: int = tokenizer.vocab_size
temperature: float = 0
max_tokens_to_complete: int = 20
num_layers: int = 30
num_groups: int = 3
embedding_dim: int = 576
num_heads: int = 9
max_sequence_length: int = 128
seed = 42
tile_size: int = 32
group_size: int = 8
optimizer = None

Token: TypeAlias = int
Tokens: TypeAlias = List[int]
Completion: TypeAlias = List[int]
Completions: TypeAlias = List[List[int]]
Reward: TypeAlias = float

pad_token = tokenizer.pad_token_id
if pad_token is None:
    pad_token = tokenizer.eos_token_id


def get_device():
    ctx = ttml.autograd.AutoContext.get_instance()
    return ctx.get_device()


def _safe_deallocate(ttnn_tensor):
    if ttnn_tensor is None:
        return
    try:
        ttnn_tensor.deallocate(True)
    except Exception:
        pass


def _deallocate_list(ttml_tensors):
    for x in ttml_tensors:
        if x is None:
            continue

        _safe_deallocate(x.get_value())


def round_to_tile(x: int) -> int:
    return ((x + tile_size - 1) // tile_size) * tile_size


def tokens_to_model_tensor(tokens: List[int]):
    tokens_len = len(tokens)
    padded_len = round_to_tile(tokens_len)

    arr = np.full((padded_len,), pad_token, dtype=np.uint32)
    arr[:tokens_len] = np.asarray(tokens, dtype=np.uint32)

    t = ttml.autograd.Tensor.from_numpy(
        arr.reshape(1, 1, 1, padded_len),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    return t


# TODO: update the description of this function
# Generates [1, 1, len, len] lower-triangular tensor
# with everything below a diagonal set to 1.
# The diagonal and everything above it is 0.
# It is a ttml::Tensor
# grads are not tracked in the casual mask
def generate_casual_mask(query_len: int, processed_tokens: int):
    assert (query_len > 0 and processed_tokens == 0) or (
        query_len == 1 and processed_tokens > 0
    )

    if processed_tokens == 0:
        n = query_len
        padded_n = round_to_tile(n)

        m = np.zeros((padded_n, padded_n), dtype=np.uint32)
        m[:n, :n] = np.tril(np.ones((n, n), dtype=np.uint32))

        return ttml.autograd.Tensor.from_numpy(
            m.reshape(1, 1, padded_n, padded_n),
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.BFLOAT16,
        )
    else:
        n = processed_tokens + 1
        padded_n = round_to_tile(n)

        m = np.zeros((tile_size, padded_n), dtype=np.uint32)
        m[0, :n] = 1

        return ttml.autograd.Tensor.from_numpy(
            m.reshape(1, 1, tile_size, padded_n),
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.BFLOAT16,
        )


# Returns a token from raw logits
def sample_token(logits):
    n, m, k, V = logits.shape()
    assert n == 1 and m == 1 and k == 1

    if temperature < 0.01:
        argmax_result = ttnn.argmax(logits.get_value(), dim=3, keepdim=True)
        next_token = int(argmax_result.item())
        next_token = min(next_token, vocab_size - 1)
    else:
        sampled = ttml.ops.sample.sample_op(logits, temperature, seed, None)
        next_token = int(sampled.get_value().item())

    return next_token


class DecodeState:
    def __init__(self, tokens: List[int]):
        head_dim = embedding_dim // num_heads
        cfg = ttml.models.KvCacheConfig(
            num_layers, 1, num_groups, max_sequence_length, head_dim
        )
        self.kv_cache = ttml.models.KvCache(cfg)
        self.kv_cache.reset()
        self.tokens = tokens[:]
        self.prefilled = False


def complete_token(state: DecodeState) -> int:
    # step_tokens == tokens we are about to compute on in this call of complete_token.
    if not state.prefilled:
        step_tokens = state.tokens
        state.prefilled = True

        processed_tokens = 0
        step_tokens_len = len(step_tokens)
    else:
        step_tokens = [state.tokens[-1]]

        processed_tokens = len(state.tokens) - 1
        step_tokens_len = 1

    padded_step_len = round_to_tile(step_tokens_len)
    input_tensor = tokens_to_model_tensor(step_tokens)
    mask_tensor = generate_casual_mask(step_tokens_len, processed_tokens)

    logits = tt_model(
        input_tensor, mask_tensor, kv_cache=state.kv_cache, new_tokens=len(step_tokens)
    )

    n, m, k, V = logits.shape()
    assert n == 1 and m == 1 and k == padded_step_len

    sliced = ttnn.slice(
        logits.get_value(), [0, 0, step_tokens_len - 1, 0], [1, 1, step_tokens_len, V]
    )

    assert sliced.shape == [1, 1, 1, V]

    last_logits = ttml.autograd.Tensor(sliced, False)  # no grad

    next_token = sample_token(last_logits)
    state.tokens.append(next_token)
    return next_token


def complete_tokens(input_tokens: List[int]) -> Completion:
    state = DecodeState(input_tokens)

    tt_model.eval()
    with no_grad():
        for _ in range(max_tokens_to_complete):
            token = complete_token(state)
            if token == tokenizer.eos_token_id:
                break

        return state.tokens[len(input_tokens) :]


def tokenize_dataset(data, tokenizer: AutoTokenizer):
    """
    Tokenizes the questions and answers in the dataset using the provided tokenizer.

    data: dataset with "question" and "answer" fields
    tokenizer: HuggingFace tokenizer
    """
    X = [sample["question"] for sample in data]
    y = [sample["answer"] for sample in data]

    tok = lambda texts: tokenizer(texts, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ]
    return tok(X), tok(y)


def get_reward(c: Completion) -> Reward:
    # Example reward: shorter completion is better
    return -float(len(c))


def iter_pass(total: int, chunk: int):
    full, rem = divmod(total, chunk)
    for _ in range(full):
        yield chunk
    if rem:
        yield rem


# Takes np.arrays 'inputs_np', 'targets_np', returns a ttml tensor 'tokens_nlog', where
# for every i, j \in [0, B-1]x[0, T-1]
# tokens_nlog[i,j] = -log(prob(token[i,j])), where
# token[i,j] = vocab[targets_np[i,j]]
def compute_nlog_probs(inputs_np, targets_np, B, T) -> Any:
    x_np = inputs_np.astype(np.uint32).reshape(B, 1, 1, T)

    X_tt = ttml.autograd.Tensor.from_numpy(
        x_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    mask_tensor = generate_casual_mask(T, 0)  # [1, 1, T, T]
    logits = tt_model(X_tt, mask_tensor)  # [B, 1, T, V]

    targets_tt = ttml.autograd.Tensor.from_numpy(
        targets_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    tokens_nlog = ttml.ops.loss.cross_entropy_loss(
        logits, targets_tt, ttml.ops.ReduceType.NONE
    )

    tokens_nlog = ttml.ops.reshape.reshape(tokens_nlog, [B, T])

    return tokens_nlog


# Generates an array of size (B, T), where T is the longest sequence length
# Returns the sequence array, an array of lengths of shape (B).
def generate_sequences(prompt: Tokens, completions: Completions, start, B):
    batch_completions = completions[start : start + B]
    sequences = [prompt + c for c in batch_completions]
    T = max(len(s) for s in sequences)

    sequences_np = np.full((B, T), pad_token, dtype=np.int32)
    lengths_np = np.zeros((B,), dtype=np.int32)

    for i, seq in enumerate(sequences):
        sequences_np[i, : len(seq)] = np.asarray(seq, dtype=np.int32)
        lengths_np[i] = len(seq)

    return sequences_np, lengths_np


def generate_inputs_targets(sequences_np):
    inputs_np = sequences_np[:, :-1]
    targets_np = sequences_np[:, 1:]

    return inputs_np, targets_np


def ignore_probs(probs_tt, l_np, r_np, B, T):
    assert l_np.shape == (B,) and r_np.shape == (B,)
    assert probs_tt.shape == (B, T - 1)

    # Build keep-mask on host (fast/simple), then apply with TTML mul
    j = np.arange(T - 1, dtype=np.int32)[None, :]  # [1, T-1]
    l = l_np.astype(np.int32).reshape(B, 1)  # [B, 1]
    r = r_np.astype(np.int32).reshape(B, 1)  # [B, 1]

    keep_np = ((j >= l) & (j <= r)).astype(np.float32)  # [B, T-1], 1 inside, 0 outside

    keep_tt = ttml.autograd.Tensor.from_numpy(
        keep_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.BFLOAT16,
    )

    # zero-out outside [l, r] per row
    return ttml.ops.binary.mul(probs_tt, keep_tt)


def calculate_loss(
    nlog_probs_tt, advantages_tt, lengths_np, prompt_len: int, B: int, T: int
):
    assert nlog_probs_tt.shape() == (B, T - 1)
    assert advantages_tt.shape() == (B, T - 1)

    # completion token counts per row (avoid divide-by-zero)
    comp_lens_np = np.maximum(
        lengths_np.astype(np.float32) - float(prompt_len), 1.0
    )  # [B]

    row_scale_np = (float(T - 1) / comp_lens_np).reshape(B, 1)  # [B,1]
    row_scale_np = np.repeat(row_scale_np, T - 1, axis=1).astype(np.float32)  # [B, T-1]

    row_scale_tt = ttml.autograd.Tensor.from_numpy(
        row_scale_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.BFLOAT16,
    )  # [B, T-1]

    adv_scaled_tt = ttml.ops.binary.mul(advantages_tt, row_scale_tt)  # [B,T-1]

    weighted_tt = ttml.ops.binary.mul(nlog_probs_tt, adv_scaled_tt)  # [B,T-1]

    return ttml.ops.unary.mean(weighted_tt)


def train_gsm8k(max_steps: int = 100):
    print("Loading GSM8K dataset...")
    train_data = datasets.load_dataset("openai/gsm8k", "main", split="train")
    X, _ = tokenize_dataset(train_data, tokenizer)
    print("Loaded GSM8K dataset!")

    for step in range(min(max_steps, len(X))):
        print(f"{step=}")
        prompt: Tokens = X[step].tolist()

        # -------------------------
        # PHASE 1: sample + rewards
        # -------------------------
        completions = []
        rewards = []

        for _ in range(group_size):
            c: Completion = complete_tokens(prompt)
            r = get_reward(c)
            completions.append(c)
            rewards.append(r)

        rewards_np = np.asarray(rewards, dtype=np.float32)
        advantages_np = rewards_np - rewards_np.mean()

        # ------------------------------------
        # PHASE 2: differentiable policy update
        # ------------------------------------

        tt_model.train()
        optimizer.zero_grad()
        start = 0
        for pass_size in iter_pass(group_size, 4):
            print(f"Pass, {start=}, {pass_size=}")
            B = pass_size

            # sequences is of shape BxT, length is of shape (B)
            sequences_np, lengths_np = generate_sequences(prompt, completions, start, B)
            T = sequences_np.shape[1]

            # shape of inputs_np, and targets_np is (B, T-1)
            inputs_np, targets_np = generate_inputs_targets(sequences_np)

            # shape of nlog_probs is (B, T-1)
            nlog_probs = compute_nlog_probs(inputs_np, targets_np, B, T)

            l_np = np.full((B,), len(prompt) - 1, dtype=np.uint32)
            r_np = lengths_np - 2
            nlog_probs = ignore_probs(
                nlog_probs, l_np, r_np, B, T
            )  # shape of nlog_probs still (B, T-1)

            advantages_pass = advantages_np[start : start + B].reshape((B, 1))
            advantages_pass = np.repeat(advantages_pass, T - 1, axis=1).astype(
                np.float32
            )  # [B,T-1]
            assert advantages_pass.shape == (B, T - 1)

            advantages_pass_tt = ttml.autograd.Tensor.from_numpy(
                advantages_pass,
                layout=ttnn.Layout.ROW_MAJOR,
                new_type=ttnn.DataType.BFLOAT16,
            )

            loss = calculate_loss(
                nlog_probs, advantages_pass_tt, lengths_np, len(prompt), B, T
            )

            loss.backward()

            start += B

        optimizer.step()


def load_training_config():
    yaml_config = load_config(
        CONFIG, f"{get_tt_metal_home()}/tt-train/configs/training_configs"
    )

    print(f"YAML config: {yaml_config}")
    model_config = load_config(yaml_config["training_config"]["model_config"])

    override_config_path = (
        f"{os.environ['TT_METAL_HOME']}/tt-train/configs/training_overrides.yaml"
    )

    if os.path.isfile(override_config_path):
        print("Applying training overrides...")

        override_config = load_config(override_config_path)

        yaml_config = yaml_deep_update(yaml_config, override_config)
        model_config = yaml_deep_update(model_config, override_config)

        # pretty output of yaml config
        import yaml

        print("Loaded YAML config:")
        print(yaml.dump(yaml_config, sort_keys=False, default_flow_style=False))
        print("*********************************\n\n")

    return model_config


def create_model(model_config):
    print("Setting up model...")
    orig_vocab_size = tokenizer.vocab_size

    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    print("Created Model Factory")

    print("Creating model...")
    tt_model = tt_model_factory.create_model()

    if LOAD_PRETRAINED:
        model_repo_path = snapshot_download(
            repo_id=HF_MODEL_ID,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        print(f"Model snapshot path: {model_repo_path}")
        print("Loading from safetensors...")
        tt_model.load_from_safetensors(model_repo_path)

    return tt_model


if __name__ == "__main__":
    set_seed(42)
    training_config = load_training_config()
    print(training_config)

    initialize_device(training_config)

    tt_model = create_model(training_config)
    print(tt_model.__dir__())

    prompt = "The capital of France is"
    input_tokens = tokenizer.encode(prompt)

    completed_tokens = complete_tokens(input_tokens)
    print("Prompt + Generated = ")
    print(tokenizer.decode(input_tokens + completed_tokens))

    optimizer = create_optimizer(tt_model, training_config)
    train_gsm8k(max_steps=100)
