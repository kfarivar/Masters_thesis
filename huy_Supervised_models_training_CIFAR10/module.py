import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy

from .cifar10_models.densenet import densenet121, densenet161, densenet169
from .cifar10_models.googlenet import googlenet
from .cifar10_models.inception import inception_v3
from .cifar10_models.mobilenetv2 import mobilenet_v2
#from cifar10_models.resnet import resnet18, resnet34, resnet50
from .cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .schduler import WarmupCosineLR

# use the model from lightning bolt
from pl_bolts.models.self_supervised.resnets import resnet18 as lit_ssl_resnet18
from pl_bolts.models.self_supervised.resnets import resnet34 as lit_ssl_resnet34
from pl_bolts.models.self_supervised.resnets import resnet50 as lit_ssl_resnet50

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False),
    "resnet34": lit_ssl_resnet34(first_conv=False, maxpool1=False, return_all_feature_maps=False),
    "resnet50": lit_ssl_resnet50(first_conv=False, maxpool1=False, return_all_feature_maps=False),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, classifier='resnet18', optimizer='sgd', learning_rate=1e-2, weight_decay=1e-2, max_epochs=100, dataset_size=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = classifier
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.dataset_size = dataset_size

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.classifier]
        
        if 'resnet' in self.classifier:
            # add the final linear layer for bolt models
            if self.classifier == 'resnet18':
                self.final_linear = torch.nn.Linear(in_features=512, out_features=10, bias=True)
            else:
                # How do I get the output featur size so I can feed it to in_features ?
                raise NotImplemented('higher resnets not implemened yet.')
            

    # modified by me

    def forward(self, images):
        if 'resnet' in self.classifier: # bolt models return a list
            features = self.model(images)[-1]
            outputs = self.final_linear(features)
        else:
            outputs = self.model(images)
        return outputs

    def shared_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        _, predictions = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        accuracy = self.accuracy(predictions, labels) *100
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.shared_step(batch)
        self.log("loss_train", loss)
        self.log("acc_train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.shared_step(batch)
        self.log("loss_val", loss)
        self.log("acc_val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.shared_step(batch)
        self.log("acc_test", accuracy)

    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        total_steps = self.max_epochs * self.dataset_size
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.1, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


class Causal3DidentModel(CIFAR10Module):
    '''
    Used for clasifying spotlight. 
    '''
    def __init__(self, classifier='resnet18', optimizer='sgd', learning_rate=0.01, weight_decay=0.01, max_epochs=100, dataset_size=None, **kwargs):
        super().__init__(classifier, optimizer, learning_rate, weight_decay, max_epochs, dataset_size, **kwargs)

        # 3dident has much bigger images so we reduce the resolution of images in the first filter 
        # default values for first_conv and maxpool1 is True which reduces resolution.
        self.model = lit_ssl_resnet18(first_conv=True, maxpool1=True)

        # I divide spotlight rotaion in 3 classes
        self.final_linear = torch.nn.Linear(in_features=512, out_features=3, bias=True)

    def shared_step(self, batch):
        images, object_class, latents = batch

        # get classes
        labels = Causal3DidentModel.spotlight_label_from_latent(latents)
        labels = labels.to(images.device)
        # filter outputs 
        images = images[labels != -1]
        # filter labels
        labels = labels[labels != -1]

        outputs = self.forward(images)

        _, predictions = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        accuracy = self.accuracy(predictions, labels) *100
        return loss, accuracy

    def spotlight_label_from_latent(latents):
        '''
        returns the labels based on a discritization of latents.

        The latents is assumed to be raw latents. in this format the independent variabels are U[-1,1].
        For angles and hues this corrresponds/maps to [-pi/2, pi/2 ] multiply by pi/2.

        discretizes the spotlight angle into 3 classes: 
                (raw_latents)   (latents)
        class 0: [1/2, 1]     [pi/4, pi/2]
        class 1: [-1/4, 1/4]    [-1/8 pi, 1/8 pi ]
        class 2: [-1, -1/2]    [-pi/2, -pi/4]
        the lengths are chosen to be equal so classes will be balanced.
        samples out of class definition get label -1.

        render internally assumes the variables form these value ranges:
        per object:
            0. x position in [-3, -3]
            1. y position in [-3, -3]
            2. z position in [-3, -3]
            3. alpha rotation in [0, 2pi]
            4. beta rotation in [0, 2pi]
            5. gamma rotation in [0, 2pi]
            6. theta spot light in [0, 2pi] <*********
            7. hue object in [0, 2pi]
            8. hue spot light in [0, 2pi]
        
        per scene:
            9. hue background in [0, 2pi]
        '''

        '''
        The code for spotlight position is here: https://github.com/brendel-group/cl-ica/blob/69568d9448de1d9acd58df98a4014a0c524d5978/tools/3dident/generate_clevr_dataset_images.py#L278
        bpy.data.objects[f"Spotlight_Object_{i}"].location = (
                    4 * np.sin(object_latents[6]),
                    4 * np.cos(object_latents[6]),
                    6 + max_object_size, (I think max_object_size is a const)
                )
        axis (from Figure 3. 3DIdent from "Contrastive Learning Inverts the Data Generating Process")
        x: back (min) to front(max)
        y: left(min) to right (max)
        z: bottom to top 
        It moves in a semi circle shining down on an angle into the object.
        -1(-pi/2) corresponds to the light being in the back of the object and 1 (pi/2) front of the object
        location of shadow relative to object and the parts of the object that are under shadow can help determin the class.
        I might want to make the classes smaller though ! I did try smaller classes wasn't necessarily easier !

    '''
        spotlight = latents[:, 6]
        # samples out of class definition get -1
        labels = torch.full(spotlight.size(), -1)

        # gaps are pi/8
        labels[spotlight> 1/2] = 0
        labels[(spotlight > -1/4) & (spotlight <1/4)] = 1
        labels[spotlight < -1/2] = 2

        """ conteracted verion gaps are pi/4
        labels[spotlight> 2/3] = 0
        labels[(spotlight > -1/6) & (spotlight <1/6)] = 1
        labels[spotlight < -2/3] = 2 """

        return labels





if __name__=='__main__':
    #Debug 

    from torchinfo import summary

    # the output is werird !!!!!!!!!! be careful
    lit_model = lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False)
    print('lightning model')
    #summary(lit_model, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)

    """ print('Resnet 18 huy')
    summary(resnet18(), input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10) """

    print('out size:')
    print(lit_model.modules())#[-1].output_size)
    
    
