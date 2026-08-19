"""DataLoader construction for the nested speech-text LM.

Thin glue over `speechtext` (the shared data layer). Both the flat baseline and
the nested model train on the same `SpokenLMDataset`; here we wrap it with a
`DistributedSampler` so each DDP rank sees a disjoint shard.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from speechtext.ds import SpokenLMDataset, make_collate
from speechtext.vocab import Vocab


def build_dataloaders(config, rank: int, world_size: int):
    args = config.model_args
    vocab = Vocab(n_units=args.vocab_size - 256 - 4)   # bytes + specials accounted for
    collate = make_collate(vocab, args.context_len)

    train_ds = SpokenLMDataset(config.units_dir, config.train_split, vocab,
                               context_len=args.context_len, seed=config.seed)
    valid_ds = SpokenLMDataset(config.units_dir, config.valid_split, vocab,
                               context_len=args.context_len, seed=config.seed + 7)

    def _sampler(ds, shuffle):
        if world_size > 1:
            return DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                      shuffle=shuffle, drop_last=True)
        return None

    train_sampler = _sampler(train_ds, shuffle=True)
    valid_sampler = _sampler(valid_ds, shuffle=False)

    common = dict(num_workers=config.num_workers, pin_memory=True,
                  collate_fn=collate, drop_last=True,
                  persistent_workers=(config.num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None), **common)
    valid_loader = DataLoader(valid_ds, batch_size=config.eval_batch_size,
                              sampler=valid_sampler, shuffle=False, **common)
    return train_loader, valid_loader, train_sampler, vocab


def cycle(loader, sampler):
    """Infinite iterator over a finite loader; advances the sampler epoch so DDP
    shuffling differs each pass."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
