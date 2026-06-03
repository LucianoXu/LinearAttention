from la.ds import get_tiny_shakespeare, txt_to_np, circle_slice
from la.model import Transformer, TransformerArgs

import torch
import tqdm

import numpy as np

from torch.utils.data import DataLoader, IterableDataset

import random
import datetime

from torch.utils import tensorboard

class SingleTextDataset(IterableDataset):
    def __init__(self, txt_path: str, slice_len: int):
        self.txt_path = txt_path
        self.slice_len = slice_len

        self.arr = txt_to_np(txt_path)
        self.arr_len = len(self.arr)
    
    def __iter__(self):
        while True:
            idx = random.randint(0, self.arr_len-1)
            yield circle_slice(self.arr, idx, self.slice_len)

def get_SingleTextDataLoader(
        txt_path: str,
        args: TransformerArgs, 
        batch_size: int):

    def collate_fn(batch: list[np.ndarray]):

        batch_np = np.array(batch)
        batch_ts = torch.tensor(batch_np, dtype=torch.long)

        input = batch_ts[:, :-1]
        label = batch_ts[:, 1:]
        return input, label


    ds = SingleTextDataset(txt_path, args.context_len + 1)

    return DataLoader(dataset=ds, batch_size=batch_size, collate_fn=collate_fn)

def single_batch_test_train(
            args: TransformerArgs,
    lr = 1e-4,
    betas = (0.9, 0.999),
    step_limit = 200,
    batch_size = 4,
    seed=42,
    device='cpu'
    ):

    random.seed(seed)

    model = Transformer(args)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        betas = betas
    )
    loss = torch.nn.CrossEntropyLoss()

    dataloader = get_SingleTextDataLoader(
        'ds/tiny_shakespeare.txt', 
        args, 
        batch_size=batch_size
    )

    step = 0

    # one subdirectory per run so TensorBoard shows them as separate runs
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tensorboard.SummaryWriter(f'log/{run_name}')

    # build the causal mask

    causal_mask = torch.tril(torch.ones(args.context_len, args.context_len, dtype=torch.int))

    progress_bar = tqdm.tqdm(range(step_limit), desc="Training Progress")

    single_batch = next(iter(dataloader))

    try:
        while True:
            optimizer.zero_grad()
            step += 1
            input, label = single_batch

            input = input.to(device=device)
            label = label.to(device=device)

            logits = model.forward(input, causal_mask)

            # logits: (B, T, V), label: (B, T)
            # CrossEntropyLoss needs (N, C) + (N,), so flatten batch and time dims
            B, T, V = logits.shape
            l : torch.Tensor = loss(logits.reshape(B * T, V), label.reshape(B * T))

            l.backward()
            optimizer.step()


            loss_value = l.item()
            writer.add_scalar("loss", loss_value, step)

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss_value)

            if step >= step_limit:
                break
    finally:
        # flush buffered events to disk and close the file, even on error/interrupt
        writer.close()

def train(
    args: TransformerArgs,
    lr = 1e-4,
    betas = (0.9, 0.999),
    step_limit = 200,
    batch_size = 4,
    seed=42,
    device='cpu'
    ):

    random.seed(seed)

    model = Transformer(args)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        betas = betas
    )
    loss = torch.nn.CrossEntropyLoss()

    dataloader = get_SingleTextDataLoader(
        'ds/tiny_shakespeare.txt', 
        args, 
        batch_size=batch_size
    )

    batches = iter(dataloader)

    step = 0

    # one subdirectory per run so TensorBoard shows them as separate runs
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tensorboard.SummaryWriter(f'log/{run_name}')

    # build the causal mask

    causal_mask = torch.tril(torch.ones(args.context_len, args.context_len, dtype=torch.int))

    progress_bar = tqdm.tqdm(range(step_limit), desc="Training Progress")

    try:
        while True:
            optimizer.zero_grad()
            step += 1
            input, label = next(batches)

            input = input.to(device=device)
            label = label.to(device=device)

            logits = model.forward(input, causal_mask)

            # logits: (B, T, V), label: (B, T)
            # CrossEntropyLoss needs (N, C) + (N,), so flatten batch and time dims
            B, T, V = logits.shape
            l : torch.Tensor = loss(logits.reshape(B * T, V), label.reshape(B * T))

            l.backward()
            optimizer.step()


            loss_value = l.item()
            writer.add_scalar("loss", loss_value, step)

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss_value)

            if step >= step_limit:
                break
    finally:
        # flush buffered events to disk and close the file, even on error/interrupt
        writer.close()








