#!/bin/bash
ARCH="resnet18" # vgg16_bn, wrn_28_4, resnet18
P_Rate=0.1
P_Reg='weight'
STG_ID='010_0'
GPU=0

echo "Robust-ADMM pruning with non-uniform strategy"
python adv_main.py --config_file config.yaml --stage admm --arch $ARCH --gpu $GPU --sparsity_type $P_Reg --pruning_rate $P_Rate --run_id $STG_ID
python adv_main.py --config_file config.yaml --stage retrain --arch $ARCH --gpu $GPU --sparsity_type $P_Reg --pruning_rate $P_Rate --run_id $STG_ID

echo "Evaluate robustness against FGSM, PGD-10, PGD-20 and CW attacks"
python eval.py --config_file config.yaml --stage retrain --gpu $GPU --arch $ARCH --sparsity_type $P_Reg --pruning_rate $P_Rate --run_id $STG_ID
