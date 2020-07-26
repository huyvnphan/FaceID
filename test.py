import os
import shutil
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything

from module import FaceIDModule


def main(hparams):
    # Reproducibility
    seed_everything(1)

    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2:  # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))

    model = FaceIDModule(hparams)

    assert hparams.train_size % hparams.batch_size == 0
    assert hparams.val_size % hparams.batch_size == 0
    assert hparams.test_size % hparams.batch_size == 0

    trainer = Trainer(
        gpus=hparams.gpus, deterministic=True, default_root_dir="test_temp"
    )
    trainer.test(model)
    shutil.rmtree(os.path.join(os.getcwd(), "test_temp"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="Test")
    parser.add_argument("--data_dir", type=str, default="/data/huy/faceid/")
    parser.add_argument("--gpus", type=str, default="0,")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=128 * 100)
    parser.add_argument("--val_size", type=int, default=128 * 50)
    parser.add_argument("--test_size", type=int, default=128 * 100)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--cnn_arch",
        type=str,
        default="squeeze_net",
        choices=["squeeze_net", "shuffle_net", "res_net", "mobile_net"],
    )
    parser.add_argument("--pretrained", type=bool, default=True)
    args = parser.parse_args()
    main(args)
