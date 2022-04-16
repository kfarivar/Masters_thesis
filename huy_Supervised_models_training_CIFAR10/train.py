import os
from argparse import ArgumentParser
from urllib import parse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from .data import CIFAR10Data
from .module import CIFAR10Module, Causal3DidentModel

from lib.Get_dataset import Causal_3Dident

from torchinfo import summary


def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(f"huy_Supervised_models_training_CIFAR10/{args.dataset}_logs", name=args.classifier)

        best_checkpoint = ModelCheckpoint(monitor="acc_val", mode="max", save_last=False, filename='best_val_acc_{acc_val:.2f}')
        interval_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=20, filename="{epoch}-{acc_val:.2f}")

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            gpus=[args.gpu_id],
            log_every_n_steps=50,
            max_epochs=args.max_epochs,
            callbacks = [best_checkpoint, interval_checkpoint],
            precision=args.precision,
        )

        if args.dataset == 'cifar10':
            data = CIFAR10Data(args)
            model = CIFAR10Module(dataset_size=len(data.train_dataloader()), **args.__dict__)
        elif args.dataset == '3dident':
            data = Causal_3Dident(data_dir='/home/kiarash_temp/adversarial-components/3dident_causal', 
                                augment_train=True, batch_size=args.batch_size, num_workers=args.num_workers,
                                train_subset=args.train_ratio, val_subset=args.val_ratio)
            data.setup()
            model = Causal3DidentModel(dataset_size = data.num_samples, **args.__dict__)
        else:
            raise NotImplementedError('dataset not supported')


        #summary(model, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, data.test_dataloader())
        else:
            print('Training: ', args.classifier)
            trainer.fit(model, data.train_dataloader(), data.val_dataloader())







if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--dataset", type=str, default="cifar10")

    parser.add_argument("--train_ratio", type=float, default=1)
    parser.add_argument("--val_ratio", type=float, default=1)

    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default='sgd')

    parser.add_argument("--no_augmentation", action="store_true", help="whether to augment the data")

    # command (run from adversarial-components): python -m huy_Supervised_models_training_CIFAR10.train --gpu_id 3 --max_epochs 200

    # (3dident) python -m huy_Supervised_models_training_CIFAR10.train --gpu_id 4 --dataset 3dident --train_ratio 0.04  --val_ratio 0.05 --max_epochs 200 --batch_size 256

    args = parser.parse_args()
    main(args)
