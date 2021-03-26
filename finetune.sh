#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 finetune.py --supervise_types supervised  \
#                                                                                                        --save_path ./result/finetune/ablation/supervised/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 finetune.py --supervise_types semi_supervised \
                                                                                                        --save_path ./result/finetune/ablation/semi/0_1_denormalization/ \
                                                                                                        --loss_weight 0.1 \
                                                                                                        --denormalization True



#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 finetune.py --supervise_types semi_supervised \
#                                                                                                        --save_path ./result/finetune/ablation/mask_semi/0_2/ \
#                                                                                                        --loss_weight 0.2