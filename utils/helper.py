import os
import yaml
import torch
from torch.optim.lr_scheduler import LambdaLR

from utils.cbow import CBOW_Model


def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")


def get_lr_scheduler(optimizer, total_epochs: int):
    """Schedules learning rate with optimized decay"""
    def lr_lambda(epoch): return (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return lr_scheduler


def save_config(config: dict, model_dir: str):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)


def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
