import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from pytorch_lightning.metrics import Accuracy

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
#from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

# use the model from lightning bolt
from pl_bolts.models.self_supervised.resnets import resnet18 as lit_ssl_resnet18
from pl_bolts.models.self_supervised.resnets import resnet34 as lit_ssl_resnet34
from pl_bolts.models.self_supervised.resnets import resnet50 as lit_ssl_resnet50

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": lit_ssl_resnet18(),
    "resnet34": lit_ssl_resnet34(),
    "resnet50": lit_ssl_resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.save_hyperparameters()
        self.my_hparams = hyperparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.my_hparams.classifier]
        
        if 'resnet' in self.my_hparams.classifier:
            # add the final linear layer for bolt models
            if self.my_hparams.classifier == 'resnet18':
                self.final_linear = torch.nn.Linear(in_features=512, out_features=10, bias=True)
            else:
                # How do I get the output featur size so I can feed it to in_features ?
                raise NotImplemented('higher resnets not implemened yet.')
            

    # This function was modified by kiya, to correct the abscence of a softmax function.  [-1]
    def forward(self, batch):
        images, labels = batch
        if 'resnet' in self.my_hparams.classifier: # bolt models return a list
            features = self.model(images)[-1]
            outputs = self.final_linear(features)
        else:
            outputs = self.model(images)

        _, predictions = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss_train", loss)
        self.log("acc_train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss_val", loss)
        self.log("acc_val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc_test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.my_hparams.learning_rate,
            weight_decay=self.my_hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.my_hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]



if __name__=='__main__':
    from torchinfo import summary

    # the output is werird !!!!!!!!!! be careful
    lit_model = lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False)
    print('lightning model')
    #summary(lit_model, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)

    """ print('Resnet 18 huy')
    summary(resnet18(), input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10) """

    print('out size:')
    print(lit_model.modules())#[-1].output_size)
    
    