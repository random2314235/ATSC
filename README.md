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

This repository is built on a open-source benchmark and previous repositories of the listed approaches (particularly SimKD).

## Pretrain teacher model
```bash
# CIFAR-100
python train_teacher.py --batch_size 64 --epochs 240 --dataset cifar100 --model resnet32x4 --learning_rate 0.05 --lr_decay_epochs 150,180,210 --weight_decay 5e-4 --trial 0 --gpu_id 0

# ImageNet
python train_teacher.py --batch_size 256 --epochs 120 --dataset imagenet --model ResNet18 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23333 --multiprocessing-distributed --dali gpu --trial 0
```

## Train student model with ATSC
```bash
# CIFAR-100
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill atsc --model_s resnet8x4 -c 0 -d 0 -b 1 -w 1 -f 2 --trial 0

# ImageNet
python train_student.py --path-t './save/teachers/models/ResNet50_vanilla/ResNet50_best.pth' --batch_size 64 --epochs 120 --dataset imagenet --model_s ResNet18 --distill atsc -c 0 -d 0 -b 1 -w 10 -f 2 --learning_rate 0.01 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23444 --multiprocessing-distributed --dali gpu --trial 0 
```

Additional scripts for various knowledge distillation (KD) approaches are available in the ./scripts directory within the [SimKD]https://github.com/DefangChen/SimKD).
