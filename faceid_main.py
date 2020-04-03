from argparse import ArgumentParser
from pytorch_lightning import Trainer
from faceid_system import FaceIDSystem

def main(hparams):
    model = FaceIDSystem(hparams)
    trainer = Trainer(gpus=[hparams.gpus], min_epoch=hparams.min_epoch)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='/raid/data/pytorch_dataset/faceid/')
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--min_epoch', default=150)
    args = parser.parse_args()

    main(args)