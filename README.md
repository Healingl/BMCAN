# BMCAN
The Pytorch implementation of *"A bidirectional multilayer contrastive adaptation network with anatomical structure preservation for unpaired cross-modality medical image segmentation"*.




## Requirements

Experiments were performed on an Ubuntu 18.04 workstation with one 24G NVIDIA GeForce RTX 3090 GPUs , CUDA 11.1, and install the virtual environment (python3.8) by:

```
pip install -r requirements.txt
```



## Implementation

Step1: Get the official data from the SIFA project: https://github.com/cchen-cc/SIFA, and convert them to the numpy files:

```
python step1_convert_tfrecords_to_numpy.py
```

Step2: Get the csv files from the numpy files (need to modify codes by your needs):

```
python step1_convert_tfrecords_to_numpy.py
```

Step3: train and evaluate models:

```
python step3_train_ct2mr_BMCAN_Res2Next.py
python step3_train_mr2ct_BMCAN_Res2Next.py
```



## Citation

If our projects are beneficial for your works, please cite:

```
@article{LIU2022105964,
title = {A bidirectional multilayer contrastive adaptation network with anatomical structure preservation for unpaired cross-modality medical image segmentation},
journal = {Computers in Biology and Medicine},
volume = {149},
pages = {105964},
year = {2022},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2022.105964},
url = {https://www.sciencedirect.com/science/article/pii/S0010482522006977},
author = {Hong Liu and Yuzhou Zhuang and Enmin Song and Xiangyang Xu and Chih-Cheng Hung},
}
```

