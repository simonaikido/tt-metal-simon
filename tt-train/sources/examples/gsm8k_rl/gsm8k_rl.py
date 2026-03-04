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
from typing import List

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
device = None
seed = 42
tile_size: int = 32


class InferenceOutput:
    prompt_ids: List[int]
    completion_ids: List[int]
    token_logprobs: List[float]

    def __init__(self, prompt_ids, completion_ids, token_logprobs):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids
        self.token_logprobs = token_logprobs


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


def tokens_to_model_tensor(tokens: List[int], device=None):
    tokens_len = len(tokens)
    padded_len = round_to_tile(tokens_len)

    arr = np.zeros((padded_len,), dtype=np.uint32)
    arr[:tokens_len] = np.asarray(tokens, dtype=np.uint32)

    t = ttml.autograd.Tensor.from_numpy(
        arr.reshape(1, 1, 1, padded_len),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    return t


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


def complete_tokens(input_tokens: List[int]):
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


def reward_fn_from_completion_ids(completion_ids):
    # Example reward: shorter completion is better
    return -float(len(completion_ids))


def train_gsm8k(tt_model, optimizer, max_steps=1000, group_size=2, max_new_tokens=8):
    print("Loading GSM8K dataset...")
    train_data = datasets.load_dataset("openai/gsm8k", "main", split="train")
    X, _ = tokenize_dataset(train_data, tokenizer)

    # You likely already have this from model config
    causal_mask_np = np.tril(
        np.ones((max_sequence_length, max_sequence_length), dtype=np.float32)
    )
    causal_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask_np.reshape(1, 1, max_sequence_length, max_sequence_length),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
    )

    for step in range(min(max_steps, len(X))):
        prompt_ids = X[step].tolist()

        # -------------------------
        # PHASE 1: sample + rewards
        # -------------------------
        sampled_completions = []
        rewards = []

        with no_grad():
            for _ in range(group_size):
                out = model_inference(
                    tt_model,
                    tokenizer,
                    prompt_ids,
                    mode="sample",
                    max_t=max_sequence_length,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                )
                sampled_completions.append(out.completion_ids)
                rewards.append(reward_fn_from_completion_ids(out.completion_ids))

            rewards_np = np.asarray(rewards, dtype=np.float32)
            advantages_np = (
                rewards_np - rewards_np.mean()
            )  # no std division (as requested)
            # advantages are now constants (detached scalars)

        # ------------------------------------
        # PHASE 2: differentiable policy update
        # ------------------------------------
        optimizer.zero_grad()

        # only for scaling, so gradient matches mean over valid samples
        valid_count = sum(1 for c in sampled_completions if len(c) > 0)
        if valid_count == 0:
            continue

        for completion_ids, adv in zip(sampled_completions, advantages_np):
            if len(completion_ids) == 0:
                continue

            # Build one training sequence: prompt + sampled completion
            seq = prompt_ids + completion_ids

            # Teacher forcing setup:
            # input is seq[:-1], target is seq[1:]
            inp = seq[:-1]
            tgt = seq[1:]

            # Truncate to model max len
            if len(inp) > max_sequence_length:
                inp = inp[-max_sequence_length:]
                tgt = tgt[-max_sequence_length:]

            # Pad to fixed length
            x_np = np.zeros((1, 1, 1, max_sequence_length), dtype=np.uint32)
            y_np = np.zeros((1, max_sequence_length), dtype=np.uint32)
            T = len(inp)
            x_np[0, 0, 0, :T] = np.asarray(inp, dtype=np.uint32)
            y_np[0, :T] = np.asarray(tgt, dtype=np.uint32)

            # Mask only completion-token positions in the target
            # completion starts after prompt, but target is shifted by 1
            prompt_len = len(prompt_ids)
            completion_start_in_target = max(0, prompt_len - 1)

            loss_scaler_np = np.zeros((1, 1, max_sequence_length, 1), dtype=np.float32)
            active_end = min(T, completion_start_in_target + len(completion_ids))
            if active_end > completion_start_in_target:
                loss_scaler_np[0, 0, completion_start_in_target:active_end, 0] = 1.0
                active = float(active_end - completion_start_in_target)
                # normalize so mean() over all tokens becomes mean over active tokens
                loss_scaler_np *= max_sequence_length / active

            X_tt = ttml.autograd.Tensor.from_numpy(
                x_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
            )
            y_tt = ttml.autograd.Tensor.from_numpy(
                y_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
            )
            scaler_tt = ttml.autograd.Tensor.from_numpy(
                loss_scaler_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
            )

            logits = tt_model(X_tt, causal_mask)
            per_tok_ce = ttml.ops.loss.cross_entropy_loss(
                logits, y_tt, ttml.ops.ReduceType.NONE
            )
            nll = ttml.ops.unary.mean(
                per_tok_ce * scaler_tt
            )  # mean NLL over completion tokens

            # GRPO policy loss: -A * logprob == A * NLL
            sample_loss = ttml.ops.binary.mul(nll, float(adv))
            sample_loss = ttml.ops.binary.mul(sample_loss, 1.0 / float(valid_count))

            # Backward per sample
            sample_loss.backward(False)
            ttml.autograd.AutoContext.get_instance().reset_graph()

            _deallocate_list(
                [
                    sample_loss,
                    nll,
                    per_tok_ce,
                    logits,
                    scaler_tt,
                    y_tt,
                    X_tt,
                ]
            )

            sample_loss = nll = per_tok_ce = logits = None
            scaler_tt = y_tt = X_tt = None

        optimizer.step()

        print(f"step={step} reward_mean={rewards_np.mean():.4f}")

    _safe_deallocate(causal_mask.get_value())


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

    # inference_output = model_inference(
    #     tt_model, tokenizer=tokenizer, prompt_ids=input_ids, mode="sample"
    # )

    # generated_text = tokenizer.decode(
    #     inference_output.completion_ids, skip_special_tokens=False
    # )
    # print(f"\nPrompt: {prompt}")
    # print(f"Generated: {generated_text}")

    # train_gsm8k(tt_model, optimizer=optim)
