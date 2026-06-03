from la.model import TransformerArgs
from train import train

if __name__ == '__main__':
    args = TransformerArgs()
    train(
        args,
    )