from la.config import TrainConfig
from train import train
import sys

if __name__ == '__main__':

    if len(sys.argv) == 1:
        config_path = "config.yaml"
    elif len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        raise ValueError("Wrong arguments. Usage: python main.py <config.yaml>")

    config = TrainConfig.from_yaml(config_path)
    train(config)
