import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader



from .model import Model
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lars import LARS


class BarlowTwins(LightningModule):
    '''This is my refactoring/organizing of yaos code in lightining similar to the bolt SSL models'''

    def __init__(
        self,
        gpu: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        lmbda:float,
        arch: str = "resnet18",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
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
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpu = gpu
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim

        self.optim = optimizer
        self.lmbda = lmbda
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
            # get both backbone and projector
            input_dim = 512
            model = Model(input_dim=input_dim, hidden_dim=self.hidden_mlp, feature_dim=self.feat_dim)
        else:
            raise NotImplemented('Other model archs not supported.')
            
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        return y

    
    

    def calculate_loss(self, batch):

        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1 = self.model(img_1)
        _, z2 = self.model(img_2)
        
        # normalize the representations along the batch dimension
        out_1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        out_2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
        
        # cross-correlation matrix
        batch_size = out_1_norm.size(0)
        c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # the loss described in the original Barlow Twin's paper
        # encouraging off_diag to be zero
        off_diag = off_diagonal(c).pow_(2).sum()
        
        loss = on_diag + self.lmbda * off_diag

        return loss



    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        # log results
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
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
            {"params": excluded_params, "weight_decay": 0.0}, # , "lr":0.0048
        ]

    def configure_optimizers(self):
        # exclude BN from weight decay
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        # according to paper
        lr = self.learning_rate * self.batch_size / 256

        # optimizers
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)

        

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs


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
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # model params
        parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=2048, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default="..", help="path to download data")

        # training params
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        #parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
        parser.add_argument('--lmbda', default=0.0078125, type=float, help='Lambda that controls the on- and off-diagonal terms')
        parser.add_argument("--num_workers", default=32, type=int, help="num of workers per GPU")
        parser.add_argument("--gpu", default=1, type=int, help="index of gpu to train on")
        parser.add_argument("--max_epochs", default=800, type=int, help="number of total epochs to run")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")

        return parser





# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
# might be useful for later 
""" def test(net, memory_data_loader, test_data_loader):
    c = len(memory_data.classes)
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
            (data, _), target = data_tuple
            target_bank.append(target)
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100 """
 

if __name__ == '__main__':
    from pl_bolts.datamodules import CIFAR10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

    # previous run params:
    # no lr scheduker

    # The paper didn't have CIFAR experiments an their Imagenet hyperparameters didn't work well for CIFAR
    # So I just use the same Adam that I was using before. with the addition of excluding BN params and bias from weight decay and learning rate cosine decay.
    # command:
    # python barlowtwins_module.py --online_ft --exclude_bn_bias --max_epochs 800  --gpu 1


    parser = argparse.ArgumentParser()
    # model args
    parser = BarlowTwins.add_model_specific_args(parser)
    args = parser.parse_args()
    
    

    # data prepare
    if args.dataset == 'cifar10':
        # use the same data module as bolt !
        val_split = 5000

        dm = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
        normalization = cifar10_normalization()
        args.gaussian_blur = False
        args.jitter_strength = 0.5

    else:
        raise NotImplemented('Other datasets not supported.')

    """ elif dataset == 'stl10':
        train_data = torchvision.datasets.STL10(root='data', split="train+unlabeled", \
                                                  transform=utils.StlPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.STL10(root='data', split="train", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.STL10(root='data', split="test", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
    elif dataset == 'tiny_imagenet':
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = True))
        memory_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', \
                                                      utils.TinyImageNetPairTransform(train_transform = False)) """


    
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

    model = BarlowTwins(**args.__dict__)

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
    save_path = f'./barlow_twins_{args.arch}_logs_and_chekpoints'
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", filename='{epoch}_best_{val_loss}_{online_val_acc:.2f}')
    interval_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=20, filename="{epoch}-{val_loss:.2f}-{online_val_acc:.2f}") 
    callbacks = [model_checkpoint, interval_checkpoint, lr_monitor]
    if args.online_ft:
        callbacks.append(online_evaluator)

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


    

    
    

