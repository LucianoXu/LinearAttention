from .model import Kath20LA
from .config import TrainConfig
import torch
import numpy as np
from pathlib import Path


def sample_next_token(
    model: Kath20LA,
    input: torch.Tensor,
    T: float = 0.6,
    top_p: float = 0.9,
):
    '''
    only consider batch = 1. input size: (1, L)
    '''
    L = input.shape[-1]
    # causal mask: without it, intermediate layers let earlier positions
    # attend to future ones, which corrupts the final logits and creates a
    # train/inference mismatch (the model was trained *with* this mask).
    mask = torch.tril(torch.ones(L, L, dtype=torch.int, device=input.device))

    logits = model(input, mask)[..., -1, :]    # (1, vocab_size)
    probs = torch.softmax(logits / T, dim=-1)  # (1, vocab_size)

    # top-p (nucleus): keep the smallest set of tokens whose cumulative
    # probability reaches top_p, drop the rest, renormalise.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # remove tokens whose *preceding* cumulative mass already passed top_p
        # (this always keeps at least the most likely token)
        remove = (cumsum - sorted_probs) > top_p
        sorted_probs[remove] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_tokens = torch.multinomial(probs, num_samples=1)  # (1, 1)

    return next_tokens


@torch.no_grad()
def generate(
    model: Kath20LA,
    prompt: str,
    max_token: int,
    T: float = 0.6,
    top_p: float = 0.9,
    device: str = 'cpu',
):
    model.eval()

    ids = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).copy()
    ids_pt = torch.from_numpy(ids).long().unsqueeze(0).to(device)

    context_len = model.args.context_len
    prompt_len = ids_pt.shape[-1]

    print(prompt, end='', flush=True)

    # generate `max_token` new tokens beyond the prompt
    while ids_pt.shape[-1] - prompt_len < max_token:
        # never feed more than the trained context window (sliding window)
        window = ids_pt[:, -context_len:]
        next_tokens = sample_next_token(model, window, T, top_p)

        ids_pt = torch.concat((ids_pt, next_tokens), dim=-1)

        new_byte = int(next_tokens.item())
        # byte-level output may be an invalid standalone UTF-8 byte -> tolerate it
        char = bytes([new_byte]).decode('utf-8', errors='replace')
        print(char, end='', flush=True)

    print()


# the linear generation

def sample_next_token_recurrent(
    model: Kath20LA,
    new_token: int,
    step_count,
    kv_state,
    phik_state,
    T: float = 0.6,
    top_p: float = 0.9,
    device='cpu',
):
    '''
    only consider batch = 1. input size: (1, 1)
    '''
    input = torch.tensor([[new_token]], dtype=torch.long, device=device) # (1, 1)

    logits, kv_state, phik_state = model.recurrent_forward(input, step_count, kv_state, phik_state)
    probs = torch.softmax(logits[:, -1, :] / T, dim=-1)  # (1, vocab_size)

    # top-p (nucleus): keep the smallest set of tokens whose cumulative
    # probability reaches top_p, drop the rest, renormalise.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # remove tokens whose *preceding* cumulative mass already passed top_p
        # (this always keeps at least the most likely token)
        remove = (cumsum - sorted_probs) > top_p
        sorted_probs[remove] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = int(torch.multinomial(probs, num_samples=1).item())

    return next_token, kv_state, phik_state


@torch.no_grad()
def generate_recurrent(
    model: Kath20LA,
    prompt: str,
    max_token: int,
    T: float = 0.6,
    top_p: float = 0.9,
    device: str = 'cpu',
):
    model.eval()

    ids = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).copy()

    kv_state = None
    phik_state = None

    token_count = 0

    for i in range(len(ids)):
        new_token = int(ids[i])

        new_byte = int(new_token)
        char = bytes([new_byte]).decode('utf-8', errors='replace')
        print(char, end='', flush=True)

        next_token, kv_state, phik_state = sample_next_token_recurrent(model, new_token, token_count, kv_state, phik_state, T, top_p, device)

        token_count += 1

    # generate `max_token` new tokens beyond the prompt
    while token_count < max_token:

        new_byte = int(next_token)
        char = bytes([new_byte]).decode('utf-8', errors='replace')
        print(char, end='', flush=True)

        next_token, kv_state, phik_state = sample_next_token_recurrent(model, next_token, token_count, kv_state, phik_state, T, top_p, device)
        token_count += 1

    print()


def load_and_generate(
    folder_path: str,
    ckpt_name: str,
    prompt: str,
    max_token: int,
    T: float = 0.6,
    top_p: float = 0.9,
    device='cpu',
    recurrent_gen=False,
):
    path = Path(folder_path)
    config = TrainConfig.from_yaml(path / "config.yaml")
    model = Kath20LA(config.model_args)
    ckpt_data = torch.load(path / ckpt_name, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data['model'])
    model.to(device)

    if recurrent_gen:
        generate_recurrent(
            model,
            prompt,
            max_token,
            T,
            top_p,
            device,
        )
    else:
        generate(
            model,
            prompt,
            max_token,
            T,
            top_p,
            device,
        )

