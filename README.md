### Extended Robust-ADMM([Shaokai Ye et al.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf)) implementation with additional supports to focus on the non-unform adversarial robust pruning: 

We fork the code from [Robustness-Aware-Pruning-ADMM](https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM) and extend with following additional supports to focus on the non-unform adversarially robust pruning:

1. Extend pruning on channel granularity in `./admm/admm.py`
2. Add reading function into every `configs.py` in `./ADMM_examples` to fetch a strategy from `strategies/$dataset$.json`
3. Apply non-uniform compression strategy via function `insert_nonuniform_strategy` in every `configs.py` from `./ADMM_examples`.
4. Add Free-Adversarial-Training in `./ADMM_examples/imagenet/adv_main.py` for ImageNet experiments.
