from la.transformer import TransformerArgs
from train import train

if __name__ == '__main__':
    args = TransformerArgs(
        dim = 512,
        n_layers = 8,
        context_len = 1024,
    )
    train(
        args,
        step_limit=3000,
        batch_size = 8,
        device='mps',
        output_path='ckpt/',
        save_interval=100,
        experiment_name='T1',
    )