This is code page for paper "Adaptive Teaching with Shared Classifier for Knowledge Distillation" submitted to NeurIPS 2024.

This repository provides several representative knowledge distillation approaches on standard image classification tasks (e.g., CIFAR100, ImageNet).

- The following approaches are currently supported by this repository:
  - [x] [Vanilla KD](https://arxiv.org/abs/1503.02531)
  - [x] [FitNet](https://arxiv.org/abs/1412.6550) [ICLR-2015]
  - [x] [AT](https://arxiv.org/abs/1612.03928) [ICLR-2017]
  - [x] [SP](https://arxiv.org/abs/1612.03928) [CVPR-2019]
  - [x] [VID](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) [CVPR-2019]
  - [x] [CRD](https://arxiv.org/abs/1910.10699) [ICLR-2020]
  - [x] [SRRL](https://openreview.net/forum?id=ZzwDy_wiWv) [ICLR-2021]
  - [x] [SemCKD](https://arxiv.org/abs/2012.03236) [AAAI-2021]
  - [x] [SimKD](https://arxiv.org/abs/2203.14001) [CVPR-2022] 
  - [x] ATSC

This repository is built on a open-source benchmark and previous repositories (SemCKD [AAAI-2021] and SimKD [CVPR-2022]).

Pretrain teacher model
# CIFAR-100
python train_teacher.py --batch_size 64 --epochs 240 --dataset cifar100 --model resnet32x4 --learning_rate 0.05 --lr_decay_epochs 150,180,210 --weight_decay 5e-4 --trial 0 --gpu_id 0

# ImageNet
python train_teacher.py --batch_size 256 --epochs 120 --dataset imagenet --model ResNet18 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23333 --multiprocessing-distributed --dali gpu --trial 0

