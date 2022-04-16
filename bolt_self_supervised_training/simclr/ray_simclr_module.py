'''Tune the hparams with ray'''
import math
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Tensor, nn
from torch._C import device
from torch.nn import functional as F

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
# from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback



class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(LightningModule):
    def __init__(
        self,
        gpus: int=1,
        num_samples: int=1,
        batch_size: int=1,
        dataset: str='cifar10',
        num_nodes: int = 1,
        arch: str = "resnet18",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = False,
        maxpool1: bool = False,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        augmentation_type = 'original',
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
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
        self.temperature = temperature
        self.augmentaion_type = augmentation_type

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.encoder = self.init_model()

        # kiya: if it is resnet18 the input_dim should be 512 !
        if self.arch == "resnet18":
            self.projection = Projection(input_dim=512, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)
        elif self.arch == "resnet50":
            self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)
        

        # compute iters per epoch
        self.train_iters_per_epoch = self.num_samples // self.batch_size

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)[-1]

    def shared_step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch
        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)
        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        

        # which type of pairing of images do we use 
        # original: 1 image randomly augmented twice, all other images negative
        # unique_images: 2 unique images from same class(randomly chosen), negative other images not in the class.
        if self.augmentaion_type == 'original':
            loss = self.nt_xent_loss(z1, z2, self.temperature)

        elif self.augmentaion_type == 'unique_images':
            # this is not the most efficient implementation since like the original simCLR I recalculate the z2 while it is already 
            # in z1. But since there are indexes being droped in my data module keeping track is hard !
            loss = self.label_based_xent_loss(z1, z2, self.temperature, y)

        elif self.augmentaion_type == 'random_images':
            # I shuffle the images in z2 as a sanity check/ baseline to see if the network still learns
            random_index = torch.randperm(z2.size(0), device=z2.device)
            z2 = z2[random_index] #.contiguous()
            loss = self.nt_xent_loss(z1, z2, self.temperature)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        #avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ray/val_loss", avg_loss)
        #self.log("ptl/val_accuracy", avg_acc)

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
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
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

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self.linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    # warmup + decay as a function I wanted to change this !
    def linear_warmup_decay(self, warmup_steps, total_steps, cosine=True, linear=False):
        """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
        assert not (linear and cosine)

        def fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            if not (cosine or linear):
                # no decay
                return 1.0

            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            if cosine:
                # cosine decay
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            # linear decay
            return 1.0 - progress

        return fn

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.full(neg.size(), math.e ** (1 / temperature), device=neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
    
    def label_based_xent_loss(self, out_1, out_2, temperature, labels, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        This is the loss version where for each image I use the labels to select a positive sample randomly from the same class (already done in my data module)
        Also the nagative samples are only from the other classes. 
        """

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)

        # we don't need to concatenate the two outputs since for denum calculation we just need representation of samples in the out_1.
        # doing a concat will just produce repetative results. 
        # Image B(out_2) is also among the set A images(out_1).
        # And we don't want to use out_2 since it is randomly samples and some images might not be there !
        cov = torch.mm(out_1, out_1.t().contiguous())
        sims = torch.exp(cov / temperature)

        # go through all classes and calculate the sum of similarities betweem 
        # this image and other images not from the same class
        
        # speed things up by not checking the classes !
        if self.dataset != 'cifar10':
            raise NotImplemented('This loss currently just supports cifar10')
        #classes = torch.unique(labels)
        
        neg = torch.zeros(sims.size(0), device=sims.device)
        for c in range(10):
            # create a mask to select columns and rows
            mask = torch.tensor(labels==c, device=sims.device)
            neg[mask] = sims[mask, :][:,~mask].sum(dim=1) 

        # we still need to add the pos to negative to get the denum.
        neg = neg + pos
        # clamp for numerical stability
        neg = torch.clamp(neg, min=eps)  

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


    # this is supposed to make the model run faster !
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")
        parser.add_argument("--augmentation_type", default='original', choices=['original', 'unique_images', 'random_images'], type=str, help="wether the two images are augmented versions of the same image or are two unique images from the same class. In the latter the negative images will only be from other classes.")
        parser.add_argument("--random_train", default=False, action="store_true", help="Use a random 45k subset of train for training.")

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="index of gpu to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=500, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.5, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


def train_model(conf_dict, const_dict):
    '''This should take dict as input, I will convert it back to args.
    conf_dict: dictionary corresponding to hparams we want to tune
    const_dict: hparams we want to keep constant.
    '''

    # Union the two dict inputs
    args = ArgumentParser()
    args.__dict__ = conf_dict | const_dict

    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
    from kiya_data_manipulations import single_images_train_transform, single_images_val_transform, CIFAR10DataModule_class_pairs, CIFAR10_use_all_train
    

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
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        if args.augmentation_type == 'unique_images':
            # set the on_after_batch_transfer by using my module !!
            dm = CIFAR10DataModule_class_pairs(
                data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
            )
        elif args.augmentation_type in ['original', 'random_images']:
            if args.random_train:
                dm = CIFAR10DataModule(
                    data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
                )
            else:
                dm = CIFAR10_use_all_train(
                    data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
                )


        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

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
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = "lars"
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")


    # avoid unnecessary calculations in the uniqe case by not calculating the second transform
    if args.augmentation_type == 'unique_images': 
        dm.train_transforms = single_images_train_transform(
            input_height=args.input_height,
            gaussian_blur=args.gaussian_blur,
            jitter_strength=args.jitter_strength,
            normalize=normalization,
        )
        dm.val_transforms = single_images_val_transform(
            input_height=args.input_height,
            gaussian_blur=args.gaussian_blur,
            jitter_strength=args.jitter_strength,
            normalize=normalization,
        ) 
        

    elif args.augmentation_type in ['original', 'random_images']:
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



    model = SimCLR(**args.__dict__)

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
    save_path = f'./simclr/ray_simCLR_{args.arch}_logs_and_chekpoints'
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", filename='{epoch}_best_{val_loss}_{online_val_acc:.2f}')
    #interval_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=20, filename="{epoch}-{val_loss:.2f}-{online_val_acc:.2f}")
    # Ray callback
    ray_callback = TuneReportCallback(
        {
            "loss": "ray/val_loss",
            #"mean_accuracy": "ptl/val_accuracy"
        },
        on="validation_end")
    callbacks = [model_checkpoint, ray_callback, lr_monitor] 
    if args.online_ft:
        callbacks.append(online_evaluator)



    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus= args.gpus, 
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        default_root_dir = save_path,
        progress_bar_refresh_rate=0,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        #profiler= 'simple' #profiler
        #fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    import os
    os.environ["SLURM_JOB_NAME"] = "bash"

    parser = ArgumentParser()
    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()
    args_dict = args.__dict__


    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=2)

    config = {
        "learning_rate": tune.loguniform(1e-1, 2)
    }
    # remove the tuning params
    for key in config.keys():
        args_dict.pop(key) 

    # remove the config params from the inputs the rest are kept constant
    train_fn_with_parameters = tune.with_parameters(train_model, const_dict=args_dict)


    # Tuning resources
    resources_per_trial = {"cpu": 32, "gpu": 1}

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples= 2,
        scheduler=scheduler,
        #progress_reporter=reporter,
        name="tune_simclr_asha",
        local_dir="./ray_tuning",
        log_to_file = "./ray_logs.logs")

    print("Best hyperparameters found were: ", analysis.best_config)


    train_model(args_dict)

    # CUDA_VISIBLE_DEVICES=5,6 python simclr/ray_simclr_module.py  --batch_size 1024  --num_workers 32 --optimizer lars --exclude_bn_bias --max_epochs 400 --online_ft --augmentation_type original

