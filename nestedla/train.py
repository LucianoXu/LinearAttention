"""DDP training for the nested speech-text LM, with per-modality loss logging.

The headline result is the *per-modality* next-token loss (text vs speech vs
cross-modal special tags): the nested slow memory should help speech tokens more
than text tokens. We therefore log text/speech/special CE separately, both for
train and validation, on top of the aggregate loss.

Structurally this mirrors transformer/train.py (torchrun DDP, bf16 autocast,
TF32, fused AdamW, rank-0 logging/ckpt, scratch output) -- see that file for the
rationale behind each distributed choice.
"""

from .config import TrainConfig
from .model import NestedLA

import os
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm

from speechtext.ds import MOD_TEXT, MOD_SPEECH, MOD_SPECIAL
from .ds import build_dataloaders, cycle

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_MODS = {"text": MOD_TEXT, "speech": MOD_SPEECH, "special": MOD_SPECIAL}


def setup_distributed(config: TrainConfig):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}"), True
    return 0, 1, 0, torch.device(config.device), False


def unwrap_model(model: nn.Module) -> nn.Module:
    model = getattr(model, "_orig_mod", model)
    return getattr(model, "module", model)


def set_seeds(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def modality_losses(logits, target_ids, target_mod, pad_id):
    """Return (total_loss, {mod: (sum_loss, count)}).

    total_loss is the mean CE over all non-pad targets (the training objective);
    the per-modality dict carries sums+counts so they can be all-reduced across
    ranks before averaging.
    """
    B, T, V = logits.shape
    ce = nn.functional.cross_entropy(
        logits.reshape(B * T, V), target_ids.reshape(B * T),
        ignore_index=pad_id, reduction="none").reshape(B, T)
    valid = (target_mod != -1)
    total = ce[valid].mean()
    per = {}
    flat_mod = target_mod.reshape(-1)
    flat_ce = ce.reshape(-1)
    for name, code in _MODS.items():
        m = (flat_mod == code)
        per[name] = (flat_ce[m].sum().detach(), m.sum().detach())
    return total, per


def train(config_path: str):
    config = TrainConfig.from_yaml(config_path)
    rank, world_size, local_rank, device, is_distributed = setup_distributed(config)
    is_main = (rank == 0)

    def log(*a, **kw):
        if is_main:
            print(*a, **kw)

    set_seeds(config.seed)
    if config.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if is_main:
        config.save()

    model_args = config.model_args
    model = NestedLA(model_args).to(device=device)
    log("ModelArgs:", model_args)
    log("Model Parameters:", sum(p.numel() for p in model.parameters()))
    log(f"slow_memory={model_args.slow_memory} | world={world_size} | "
        f"per-GPU batch={config.batch_size} | "
        f"eff tokens/step={world_size*config.batch_size*config.grad_accum_steps*model_args.context_len}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if config.compile:
        log("Compiling model with torch.compile ...")
        model = torch.compile(model)

    amp_dtype = _DTYPES[config.dtype]
    use_amp = amp_dtype is not torch.float32 and device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas,
                                  weight_decay=config.wd, fused=(device.type == "cuda"))

    def lr_lambda(step):
        if step < config.warm_up_steps:
            return step / max(1, config.warm_up_steps)
        progress = (step - config.warm_up_steps) / max(1, config.step_limit - config.warm_up_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader, valid_loader, train_sampler, vocab = build_dataloaders(config, rank, world_size)
    pad_id = vocab.pad
    batches = cycle(train_loader, train_sampler)
    valid_batches_iter = cycle(valid_loader, None)
    single_batch = next(batches) if config.single_batch_test else None

    def save_model_optimizer(step):
        if not is_main:
            return
        config.exp_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "args": model_args,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   config.exp_dir / f"ckpt-{step}.pth")
    save_model_optimizer(0)

    train_writer = valid_writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            train_writer = SummaryWriter(config.exp_dir / "train")
            valid_writer = SummaryWriter(config.exp_dir / "valid")
        except Exception as e:
            log(f"[warn] tensorboard unavailable ({e})")

    progress_bar = tqdm.tqdm(range(config.step_limit), desc="Training", disable=not is_main)
    step = 0
    model.train()
    try:
        while True:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for micro in range(config.grad_accum_steps):
                batch = single_batch if config.single_batch_test else next(batches)
                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["target_ids"].to(device, non_blocking=True)
                ymod = batch["target_mod"].to(device, non_blocking=True)
                sync_ctx = (model.no_sync() if is_distributed and micro < config.grad_accum_steps - 1
                            else nullcontext())
                with sync_ctx, autocast_ctx:
                    logits = model(x)
                    l, _ = modality_losses(logits, y, ymod, pad_id)
                    l = l / config.grad_accum_steps
                scaler.scale(l).backward()
                loss_accum += l.item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)
            scaler.step(optimizer); scaler.update(); scheduler.step()

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
                vl = evaluate(model, valid_batches_iter, config.valid_batches, pad_id,
                              device, autocast_ctx, is_distributed)
                if is_main and valid_writer is not None:
                    for k, v in vl.items():
                        valid_writer.add_scalar(k, v, step)
                log(f"[step {step}] valid: " + " ".join(f"{k}={v:.4f}" for k, v in vl.items()))
                model.train()

            if step >= config.step_limit:
                break
    finally:
        save_model_optimizer(step)
        if train_writer is not None:
            train_writer.close()
        if valid_writer is not None:
            valid_writer.close()
        if is_distributed:
            dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, valid_iter, valid_batches, pad_id, device, autocast_ctx, is_distributed):
    """Per-modality validation loss, all-reduced across ranks.

    Returns {"loss", "text", "speech", "special"} as mean CE per token.
    """
    model.eval()
    fwd = unwrap_model(model)
    sums = {k: torch.zeros((), device=device) for k in ("loss", *_MODS)}
    counts = {k: torch.zeros((), device=device) for k in ("loss", *_MODS)}
    for _ in range(valid_batches):
        batch = next(valid_iter)
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["target_ids"].to(device, non_blocking=True)
        ymod = batch["target_mod"].to(device, non_blocking=True)
        with autocast_ctx:
            logits = fwd(x)
            B, T, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.reshape(B * T, V), y.reshape(B * T),
                ignore_index=pad_id, reduction="none").reshape(-1)
        fm = ymod.reshape(-1)
        valid = (fm != -1)
        sums["loss"] += ce[valid].sum(); counts["loss"] += valid.sum()
        for name, code in _MODS.items():
            m = (fm == code)
            sums[name] += ce[m].sum(); counts[name] += m.sum()

    if is_distributed:
        for k in sums:
            dist.all_reduce(sums[k], op=dist.ReduceOp.SUM)
            dist.all_reduce(counts[k], op=dist.ReduceOp.SUM)
    return {k: (sums[k] / counts[k].clamp(min=1)).item() for k in sums}
