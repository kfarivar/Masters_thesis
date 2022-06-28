# adversarial-components

A benchmarking suite aimed at measuring disentangled components of adversarial susceptibility.

See also https://robustbench.github.io/ for a similar work which focuses only on deep image classifier robustness.

# Libraries used

refer to environment_setup_newpytorch.yml

# Traning models

The code to train each SSL model is in model_name_module.py

The supervised model: ./huy_Supervised_models_training_CIFAR10/train.py

simclr and simsiam: ./bolt_self_supervised_training/[simclr or simsiam]

barlow twins: ./home/kfarivar/adversarial-components/barlow_twins_yao_training

