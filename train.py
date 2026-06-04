from la.ds import get_tiny_shakespeare, txt_to_np, circle_slice
from la.transformer import Transformer, TransformerArgs
from pathlib import Path

import torch
import tqdm

import numpy as np

from torch.utils.data import DataLoader, Dataset, IterableDataset

import random
import math

from torch.utils import tensorboard

class SingleNumpyDataset(IterableDataset):
    def __init__(self, arr: np.ndarray, slice_len: int):
        self.slice_len = slice_len
        self.arr = arr
        self.arr_len = len(self.arr)
    
    def __iter__(self):
        while True:
            idx = random.randint(0, self.arr_len-1)
            yield circle_slice(self.arr, idx, self.slice_len)

class NumpyValidDataset(Dataset):
    def __init__(self, arr: np.ndarray, slice_len: int):
        self.slice_len = slice_len
        self.arr = arr
        self.arr_len = len(self.arr)

    def __len__(self):
        return self.arr_len // self.slice_len
    
    def __getitem__(self, i: int):
        return circle_slice(self.arr, i * self.slice_len, self.slice_len)

def get_SingleNumpy_train_valid_DataLoader(
        txt_path: str,
        train_ratio: float,
        args: TransformerArgs, 
        batch_size: int):
    
    arr = txt_to_np(txt_path)

    slice_idx = round(len(arr) * train_ratio)
    
    arr_train = arr[:slice_idx]
    arr_valid = arr[slice_idx:]

    def collate_fn(batch: list[np.ndarray]):

        batch_np = np.array(batch)
        batch_ts = torch.tensor(batch_np, dtype=torch.long)

        input = batch_ts[:, :-1]
        label = batch_ts[:, 1:]
        return input, label


    train_ds = SingleNumpyDataset(arr_train, args.context_len + 1)
    valid_ds = NumpyValidDataset(arr_valid, args.context_len + 1)

    train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=valid_ds, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
        args: TransformerArgs,
        lr = 1e-4,
        betas = (0.9, 0.999),
        grad_norm_clip = 1.0,
        warm_up_steps = 300,
        step_limit = 2000,
        batch_size = 4,
        seed=42,
        train_ratio = 0.9,
        device='cpu',
        output_path='ckpt/',
        experiment_name='example',
        save_interval = None,
        valid_interval = 50,
        single_batch_test=False,
    ):

    # initialization

    set_seeds(seed)

    model = Transformer(args)
    model.to(device=device)

    print("TransformerArgs: ")
    print(args)

    print()
    print("Model Parameters: ", sum([param.numel() for param in model.parameters()]))
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        betas = betas
    )

    # cosine lr scheduler with warm-up
    def lr_lambda(current_step):
        if current_step < warm_up_steps:
            return current_step / max(1, warm_up_steps)
        
        progress = (current_step - warm_up_steps) / max(1, step_limit - warm_up_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0

    output_path = Path(output_path)

    def save_model_optimizer(step: int):
        folder = (output_path / experiment_name)
        folder.mkdir(parents=True, exist_ok=True)
        state = {
            'step': step,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        save_name = "ckpt-" + str(step) + ".pth"
        torch.save(state, folder / save_name)

    # test the saving
    save_model_optimizer(0)

    loss = torch.nn.CrossEntropyLoss()

    train_dataloader, valid_dataloader = get_SingleNumpy_train_valid_DataLoader(
        'ds/tiny_shakespeare.txt', 
        train_ratio,
        args, 
        batch_size=batch_size
    )

    batches = iter(train_dataloader)

    if single_batch_test:
        single_batch = next(batches)

    # train and valid write the SAME tag ("loss") into separate subdirs, so
    # TensorBoard overlays them as two lines in one chart (legend: train/valid)
    log_dir = output_path / experiment_name
    train_writer = tensorboard.SummaryWriter(log_dir / "train")
    valid_writer = tensorboard.SummaryWriter(log_dir / "valid")

    # build the causal mask

    causal_mask = torch.tril(torch.ones(args.context_len, args.context_len, dtype=torch.int, device=device))

    progress_bar = tqdm.tqdm(range(step_limit), desc="Training Progress")

    try:
        while True:
            optimizer.zero_grad()
            step += 1

            if single_batch_test:
                input, label = single_batch
            else:
                input, label = next(batches)

            input = input.to(device=device)
            label = label.to(device=device)

            logits = model.forward(input, causal_mask)

            # logits: (B, T, V), label: (B, T)
            # CrossEntropyLoss needs (N, C) + (N,), so flatten batch and time dims
            B, T, V = logits.shape
            l : torch.Tensor = loss(logits.reshape(B * T, V), label.reshape(B * T))

            l.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_norm_clip)

            optimizer.step()
            scheduler.step()

            loss_value = l.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss_value)

            # record in tensorboard
            train_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
            train_writer.add_scalar("loss", loss_value, step)
            train_writer.add_scalar("grad_nom", grad_norm.item(), step)


            if isinstance(save_interval, int) and step % save_interval == 0:
                save_model_optimizer(step)

            if step % valid_interval == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    for batch in valid_dataloader:
                        input, label = batch

                        input = input.to(device=device)
                        label = label.to(device=device)

                        logits = model(input, causal_mask)

                        B, T, V = logits.shape
                        l = loss(logits.reshape(B * T, V), label.reshape(B * T))

                        valid_loss += l.item()

                    valid_loss /= len(valid_dataloader)

                    valid_writer.add_scalar("loss", valid_loss, step)

                model.train()


            if step >= step_limit:
                break
    finally:
        # flush buffered events to disk and close the file, even on error/interrupt
        save_model_optimizer(step)
        train_writer.close()
        valid_writer.close()








