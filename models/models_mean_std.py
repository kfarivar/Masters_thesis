# each implementaion's (mean , std) used during training to normalize the dataset

supervised_huy = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))

barlow_twins_yao = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

simCLR_bolts = ([x / 255.0 for x in [125.3, 123.0, 113.9]], [x / 255.0 for x in [63.0, 62.1, 66.7]] )

