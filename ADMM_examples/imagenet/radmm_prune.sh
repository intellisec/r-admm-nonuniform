#!/bin/bash
P_Rate=0.1
P_Reg='weight'
STG_ID='010_0'

echo "Robust-ADMM pruning with non-uniform strategy"
python adv_main.py --config_file config.yaml --stage admm --sparsity_type $P_Reg --pruning_rate $P_Rate --run_id $STG_ID
python adv_main.py --config_file config.yaml --stage retrain --sparsity_type $P_Reg --pruning_rate $P_Rate --run_id $STG_ID

echo "Evaluate robustness against FGSM, PGD20, CW attacks"
python eval_adv.py --config_file config.yaml --stage retrain --sparsity_type $P_Reg --pruning_rate $P_Rate --run_id $STG_ID
