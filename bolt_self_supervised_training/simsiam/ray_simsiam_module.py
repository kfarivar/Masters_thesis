from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn import functional as F

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from .models import SiameseArm
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)



class SimSiam(LightningModule):
    """PyTorch Lightning implementation of Exploring Simple Siamese Representation Learning (SimSiam_)

    Model refactored by Kiya and works correctly on CIFAR10

    .. _SimSiam: https://arxiv.org/pdf/2011.10566v1.pdf
    """

    def __init__(
        self,
        gpu: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpu = gpu
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.init_model()

        # compute iters per epoch 
        self.train_iters_per_epoch = self.num_samples // self.batch_size

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
            input_dim = 512
        elif self.arch == "resnet50":
            backbone = resnet50
            input_dim = 2048
            
        encoder = backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

        # this uses my version of SiameseArm in this same folder
        # this adds the projector and predictor to the encoder given to it.
        # they all have 1 hidden layer. 
        # it returns the result of all 3 networks (encoder, projector, predictor).
        # In the paper they mention that the predictor needs to be a bottleneck and have 1/4 of units in projector output as hidden units.
        self.online_network = SiameseArm(
            encoder, input_dim= input_dim, projector_hidden_size=self.hidden_mlp, predictor_hidden_size= int(self.feat_dim/4), output_dim=self.feat_dim
        )

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    def shared_step(self, batch):
        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        # log results
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        # log results
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optim == 'sgd':
            # kiya: the paper suggested optimizer. 
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        if self.optim=='sgd' and self.batch_size<1024:
            # the paper doesn't use a warmup for small batches
            warmup_steps = 0

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model params
        parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=2048, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpu", default=1, type=int, help="index of gpu to train on")
        parser.add_argument("--optimizer", default="sgd", type=str, help="choose between sgd/adam/lars")
        parser.add_argument("--max_epochs", default=800, type=int, help="number of total epochs to run")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=3e-2, type=float, help="base learning rate")

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    #seed_everything(1234)

    parser = ArgumentParser()

    # model args
    parser = SimSiam.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    # init datamodule
    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 5000
        if args.num_nodes * args.gpu * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpu * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5

    elif args.dataset == "imagenet":
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.batch_size = 64
        args.num_nodes = 8
        args.gpu = 8  # per-node
        args.max_epochs = 800

        args.optimizer = "lars"
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimSiam(**args.__dict__)

    # finetune in real-time
    online_evaluator = None
    if args.online_ft:
        # online eval
        # (kiya)seperate the resnet18 case z_dim=512 (representation dim)
        if args.arch == 'resnet18':
            online_evaluator = SSLOnlineEvaluator(
                drop_p=0.0,
                hidden_dim=None,
                z_dim=512,
                num_classes=dm.num_classes,
                dataset=args.dataset,
            )
        elif args.arch == 'resnet50':
            online_evaluator = SSLOnlineEvaluator(
                drop_p=0.0,
                hidden_dim=None,
                z_dim=args.hidden_mlp,
                num_classes=dm.num_classes,
                dataset=args.dataset,
            )


    lr_monitor = LearningRateMonitor(logging_interval="step")
    save_path = f'./simsiam/simsiam_{args.arch}_logs_and_chekpoints'
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", filename='{epoch}_best_{val_loss}_{online_val_acc:.2f}')
    interval_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=20, filename="{epoch}-{val_loss:.2f}-{online_val_acc:.2f}")
    callbacks = [model_checkpoint, online_evaluator, interval_checkpoint] if args.online_ft else [model_checkpoint, interval_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus= [args.gpu], # kiya: so the cli input is the gpu index not number of gpus 
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        default_root_dir = save_path,
        profiler="simple"
        #fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()

    # (run from parent folder) command: python simsiam/simsiam_module.py --gpu 1 --num_workers 16 --online_ft
