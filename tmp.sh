#CUDA_VISIBLE_DEVICES=0 \
    #python run_generator_vc2.py generate-traversals \
    #--network /mnt/hdd/repo_results/stylegan2/results_vc2_PG_clevrSim_byvae/00009-vc2_gan_byvae-clevr_simple-2gpu-config-c/network-snapshot-001440.pkl \
    #--seeds 1-20 \
    #--result-dir /mnt/hdd/repo_results/stylegan2/results_vc2_PG_clevrSim_byvae/00009-vc2_gan_byvae-clevr_simple-2gpu-config-c/1440_travs \
    #--tpl_metric tpl_small \
    #--n_samples_per 5 \
    #--topk_dims_to_show 8

CUDA_VISIBLE_DEVICES=0 \
    python run_generator_vc2.py generate-traversals \
    --network /mnt/hpc_hdd/stylegan2/results_vc2_byvae_celeba/00025-vc2_gan_byvae-celeba_tfr-2gpu-config-c/network-snapshot-003840.pkl \
    --seeds 1-20 \
    --result-dir /mnt/hpc_hdd/stylegan2/results_vc2_byvae_celeba/00025-vc2_gan_byvae-celeba_tfr-2gpu-config-c/3840_travs \
    --tpl_metric tpl_small \
    --n_samples_per 3 \
    --topk_dims_to_show 8
