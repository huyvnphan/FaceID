import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from module import FaceIDModule


def main(hparams):
    # Reproducibility
    seed_everything(1)

    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2:  # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))

    model = FaceIDModule(hparams)

    wandb_logger = WandbLogger(
        name=hparams.description,
        project="faceid",
        save_dir=os.path.join(os.getcwd(), "logs/"),
    )

    trainer = Trainer(
        logger=wandb_logger,
        gpus=hparams.gpus,
        max_epochs=hparams.max_epochs,
        early_stop_callback=False,
        check_val_every_n_epoch=5,
        fast_dev_run=False,
        deterministic=True,
        weights_summary=None,
        weights_save_path="weights/" + hparams.cnn_arch,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="Test")
    parser.add_argument("--data_dir", type=str, default="/data/huy/faceid/")
    parser.add_argument("--gpus", type=str, default="0,")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD"]
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--cnn_arch",
        type=str,
        default="squeeze_net",
        choices=["squeeze_net", "shuffle_net", "res_net", "mobile_net"],
    )
    args = parser.parse_args()
    main(args)
