from .ds import txt_to_np, circle_slice
from .model import Transformer
from .config import TrainConfig

import os
import math
import random
from contextlib import nullcontext

import numpy as np

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

import tqdm


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed(config: TrainConfig):
    """Initialise torch.distributed from the env vars that ``torchrun`` sets.

    Returns ``(rank, world_size, local_rank, device, is_distributed)``. When the
    process was *not* launched by torchrun (no RANK/WORLD_SIZE in the env) we
    fall back to a single-process run on ``config.device`` (cpu / mps / cuda),
    so the same script still works for local debugging on the laptop.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device, True

    # single-process fallback
    device = torch.device(config.device)
    return 0, 1, 0, device, False


def cleanup_distributed(is_distributed: bool):
    # No barrier() here on purpose: this runs from the training `finally`, so if
    # ONE rank crashed, a barrier would block the surviving ranks until the
    # 10-min NCCL watchdog fires (exactly what happened in job 27725880). Going
    # straight to destroy lets the dead rank exit immediately; torchrun then
    # tears the rest down within seconds.
    if is_distributed:
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip torch.compile (``_orig_mod``) and DDP (``module``) wrappers so we
    can read the underlying ``Transformer`` (for state_dict / .args)."""
    model = getattr(model, "_orig_mod", model)   # torch.compile wrapper
    model = getattr(model, "module", model)       # DDP wrapper
    return model


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SingleNumpyDataset(IterableDataset):
    """Infinite stream of random fixed-length slices.

    Each (rank, dataloader-worker) pair gets a *distinct* RNG seed so that the
    GPUs never train on identical batches — otherwise DDP would just average
    identical gradients and waste 3 of the 4 devices.
    """

    def __init__(self, arr: np.ndarray, slice_len: int, seed: int = 0, rank: int = 0):
        self.slice_len = slice_len
        self.arr = arr
        self.arr_len = len(self.arr)
        self.seed = seed
        self.rank = rank

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + self.rank * 1_000_003 + worker_id)
        while True:
            idx = rng.randint(0, self.arr_len - 1)
            yield circle_slice(self.arr, idx, self.slice_len)


def _collate_fn(batch: list[np.ndarray]):
    batch_ts = torch.tensor(np.array(batch), dtype=torch.long)
    return batch_ts[:, :-1], batch_ts[:, 1:]


def get_SingleNumpy_train_valid_DataLoader(
        txt_path: str,
        train_ratio: float,
        args,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        seed: int,
        rank: int,
        world_size: int):

    arr = txt_to_np(txt_path)
    slice_idx = round(len(arr) * train_ratio)
    arr_train = arr[:slice_idx]
    arr_valid = arr[slice_idx:]

    train_ds = SingleNumpyDataset(arr_train, args.context_len + 1, seed=seed, rank=rank)
    # validation also samples random *fixed-size* slices (per-rank decorrelated,
    # offset seed so it isn't the same stream as train). Fixed eval_batch_size
    # means no torch.compile shape recompiles, and we average over a fixed number
    # of batches (config.valid_batches) rather than the whole valid set.
    valid_ds = SingleNumpyDataset(arr_valid, args.context_len + 1, seed=seed + 7, rank=rank)

    train_dataloader = DataLoader(
        dataset=train_ds, batch_size=batch_size, collate_fn=_collate_fn,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    valid_dataloader = DataLoader(
        dataset=valid_ds, batch_size=eval_batch_size, collate_fn=_collate_fn,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_dataloader, valid_dataloader


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def train(config_path: str):

    config = TrainConfig.from_yaml(config_path)

    rank, world_size, local_rank, device, is_distributed = setup_distributed(config)
    is_main = (rank == 0)

    def log(*a, **kw):
        if is_main:
            print(*a, **kw)

    # identical seed on every rank -> identical model init (DDP also broadcasts,
    # so this is belt-and-suspenders). Data streams are decorrelated separately.
    set_seeds(config.seed)

    # Ampere TF32: big matmul/conv speedup at negligible accuracy cost.
    if config.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if is_main:
        config.save()   # persist resolved config (rank 0 only)

    model_args = config.model_args

    model = Transformer(model_args).to(device=device)

    log("ModelArgs:", model_args)
    log("Model Parameters:", sum(p.numel() for p in model.parameters()))
    log(f"World size: {world_size}  |  per-GPU batch: {config.batch_size}  |  "
        f"grad_accum: {config.grad_accum_steps}  |  "
        f"effective tokens/step: {world_size * config.batch_size * config.grad_accum_steps * model_args.context_len}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if config.compile:
        log("Compiling model with torch.compile ...")
        model = torch.compile(model)

    # autocast / mixed precision
    amp_dtype = _DTYPES[config.dtype]
    use_amp = amp_dtype is not torch.float32 and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    )
    # GradScaler is only needed for fp16; bf16 has enough dynamic range without it.
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.wd,
        fused=(device.type == "cuda"),
    )

    def lr_lambda(current_step):
        if current_step < config.warm_up_steps:
            return current_step / max(1, config.warm_up_steps)
        progress = (current_step - config.warm_up_steps) / max(1, config.step_limit - config.warm_up_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0

    def save_model_optimizer(step: int):
        if not is_main:
            return
        folder = config.exp_dir
        folder.mkdir(parents=True, exist_ok=True)
        state = {
            'step': step,
            'args': model_args,
            'model': unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, folder / f"ckpt-{step}.pth")

    save_model_optimizer(0)   # sanity-check the save path early

    ce = torch.nn.CrossEntropyLoss()

    train_dataloader, valid_dataloader = get_SingleNumpy_train_valid_DataLoader(
        config.dataset_path,
        config.train_ratio,
        model_args,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
    )

    batches = iter(train_dataloader)
    valid_iter = iter(valid_dataloader)
    single_batch = next(batches) if config.single_batch_test else None

    # tensorboard is optional (not in the base module); degrade gracefully.
    train_writer = valid_writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            train_writer = SummaryWriter(config.exp_dir / "train")
            valid_writer = SummaryWriter(config.exp_dir / "valid")
        except Exception as e:
            log(f"[warn] tensorboard unavailable ({e}); skipping scalar logging.")

    causal_mask = torch.tril(torch.ones(
        model_args.context_len, model_args.context_len, dtype=torch.int, device=device))

    progress_bar = tqdm.tqdm(range(config.step_limit), desc="Training", disable=not is_main)

    model.train()
    try:
        while True:
            step += 1
            optimizer.zero_grad(set_to_none=True)

            loss_accum = 0.0
            for micro in range(config.grad_accum_steps):
                if config.single_batch_test:
                    input, label = single_batch
                else:
                    input, label = next(batches)

                input = input.to(device=device, non_blocking=True)
                label = label.to(device=device, non_blocking=True)

                # In DDP, only synchronise gradients on the *last* micro-step;
                # no_sync() on the others avoids redundant all-reduces.
                sync_ctx = (
                    model.no_sync()
                    if is_distributed and micro < config.grad_accum_steps - 1
                    else nullcontext()
                )

                with sync_ctx, autocast_ctx:
                    logits = model(input, causal_mask)
                    B, T, V = logits.shape
                    l = ce(logits.reshape(B * T, V), label.reshape(B * T))
                    l = l / config.grad_accum_steps

                scaler.scale(l).backward()
                loss_accum += l.item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            progress_bar.update(1)
            if is_main and step % config.log_interval == 0:
                progress_bar.set_postfix(loss=loss_accum)
                if train_writer is not None:
                    train_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
                    train_writer.add_scalar("loss", loss_accum, step)
                    train_writer.add_scalar("grad_norm", grad_norm.item(), step)

            if isinstance(config.save_interval, int) and step % config.save_interval == 0:
                save_model_optimizer(step)

            if step % config.valid_interval == 0:
                valid_loss = evaluate(model, valid_iter, config.valid_batches, ce,
                                      causal_mask, device, autocast_ctx,
                                      is_distributed, world_size)
                if is_main and valid_writer is not None:
                    valid_writer.add_scalar("loss", valid_loss, step)
                model.train()

            if step >= config.step_limit:
                break
    finally:
        save_model_optimizer(step)
        if train_writer is not None:
            train_writer.close()
        if valid_writer is not None:
            valid_writer.close()
        cleanup_distributed(is_distributed)


@torch.no_grad()
def evaluate(model, valid_iter, valid_batches, ce, causal_mask, device, autocast_ctx,
             is_distributed, world_size):
    """Average loss over a fixed number of randomly-sampled validation batches
    (each rank draws its own decorrelated batches), all-reduced so every rank
    agrees on the global figure."""
    model.eval()
    # Validate with the EAGER module: even though sampled batches are fixed-size,
    # using the uncompiled module keeps validation independent of torch.compile.
    # The eager module shares weights with the compiled/DDP wrapper, so it's exact.
    fwd_model = unwrap_model(model)
    loss_sum = torch.zeros((), device=device)

    for _ in range(valid_batches):
        input, label = next(valid_iter)
        input = input.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)
        with autocast_ctx:
            logits = fwd_model(input, causal_mask)
            B, T, V = logits.shape
            l = ce(logits.reshape(B * T, V), label.reshape(B * T))
        loss_sum += l.detach()

    if is_distributed:
        # each rank ran `valid_batches` batches -> divide by the global total
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        return (loss_sum / (valid_batches * world_size)).item()
    return (loss_sum / valid_batches).item()
