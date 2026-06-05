from .ds import get_tiny_shakespeare, txt_to_np, circle_slice
from .model import RetNet
from .config import TrainConfig

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
        args,
        batch_size: int,
        eval_batch_size: int):

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
    valid_ds = SingleNumpyDataset(arr_valid, args.context_len + 1)   # random sampling, like train

    train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=valid_ds, batch_size=eval_batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(config_path: str):

    config = TrainConfig.from_yaml(config_path)

    # persist the resolved config (defaults merged in) for reproducibility
    config.save()

    # initialization

    set_seeds(config.seed)

    model_args = config.model_args
    device = config.device

    model = RetNet(model_args)
    model.to(device=device)

    print("ModelArgs: ")
    print(model_args)

    print()
    print("Model Parameters: ", sum(param.numel() for param in model.parameters()))
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.wd
    )

    # cosine lr scheduler with warm-up
    def lr_lambda(current_step):
        if current_step < config.warm_up_steps:
            return current_step / max(1, config.warm_up_steps)

        progress = (current_step - config.warm_up_steps) / max(1, config.step_limit - config.warm_up_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0

    def save_model_optimizer(step: int):
        folder = config.exp_dir
        folder.mkdir(parents=True, exist_ok=True)
        state = {
            'step': step,
            'args': model_args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, folder / f"ckpt-{step}.pth")

    # test the saving
    save_model_optimizer(0)

    loss = torch.nn.CrossEntropyLoss()

    train_dataloader, valid_dataloader = get_SingleNumpy_train_valid_DataLoader(
        config.dataset_path,
        config.train_ratio,
        model_args,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
    )

    batches = iter(train_dataloader)
    valid_iter = iter(valid_dataloader)

    if config.single_batch_test:
        single_batch = next(batches)

    # train and valid write the SAME tag ("loss") into separate subdirs, so
    # TensorBoard overlays them as two lines in one chart (legend: train/valid)
    log_dir = config.exp_dir
    train_writer = tensorboard.SummaryWriter(log_dir / "train")
    valid_writer = tensorboard.SummaryWriter(log_dir / "valid")

    # build the causal mask

    causal_mask = torch.tril(torch.ones(model_args.context_len, model_args.context_len, dtype=torch.int, device=device))

    progress_bar = tqdm.tqdm(range(config.step_limit), desc="Training Progress")

    try:
        while True:
            optimizer.zero_grad()
            step += 1

            if config.single_batch_test:
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

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)

            optimizer.step()
            scheduler.step()

            loss_value = l.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss_value)

            # record in tensorboard
            train_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
            train_writer.add_scalar("loss", loss_value, step)
            train_writer.add_scalar("grad_norm", grad_norm.item(), step)


            if isinstance(config.save_interval, int) and step % config.save_interval == 0:
                save_model_optimizer(step)

            if step % config.valid_interval == 0:
                model.eval()
                with torch.inference_mode():
                    # sampling-based eval: average loss over a fixed number of
                    # randomly-sampled batches instead of the whole valid set
                    valid_loss = 0
                    for _ in range(config.valid_batches):
                        input, label = next(valid_iter)

                        input = input.to(device=device)
                        label = label.to(device=device)

                        logits = model(input, causal_mask)

                        B, T, V = logits.shape
                        l = loss(logits.reshape(B * T, V), label.reshape(B * T))

                        valid_loss += l.item()

                    valid_loss /= config.valid_batches

                    valid_writer.add_scalar("loss", valid_loss, step)

                model.train()


            if step >= config.step_limit:
                break
    finally:
        # flush buffered events to disk and close the file, even on error/interrupt
        save_model_optimizer(step)
        train_writer.close()
        valid_writer.close()
