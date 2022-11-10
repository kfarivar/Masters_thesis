# Experiments on Adversarial robustness and Self-supervised learning

* The loss function is supposed to be a proxy for the accuracy measure used to analyse a model. But this proxy is not always perfect.

* I explore how the definition of the loss function and the specific method of learning used (self-supervised or supervised) affects the robustness of a model to adversarial examples.

* A synthetic dataset such as 3dident could help us isolate the factors that contribute to adversarial susceptibility. 


* refer to environment_setup_newpytorch.yml for required libraries.

# Traning models

The code to train each SSL model is in model_name_module.py

The supervised model: ./huy_Supervised_models_training_CIFAR10/train.py

simclr and simsiam: ./bolt_self_supervised_training/[simclr or simsiam]

barlow twins: ./home/kfarivar/adversarial-components/barlow_twins_yao_training

* I also create simple toy datasets and showed that it is possible to train a 100\% robust model using the standard supervised training pipeline.

* for more information see [report](https://github.com/kfarivar/Masters_thesis/blob/main/thesis_report/thesis.pdf).


# Ideas to explore furthur 

* In connection to footnote on page 40. Can we attack a human brain using a back box attack. Will we beable to full a human in l_infinity norm and with a small epsilon ?? 
