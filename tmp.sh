#for i in 0.5 0.8
#do
    #for j in 0.1 0.01
    #do
        #python run_training_vc2.py \
            #--result-dir /project/xqzhu/repo_results/stylegan2/results_vc2_infogan2_celeba \
            #--data-dir /project/xqzhu/Datasets/CelebA_dataset \
            #--dataset celeba_tfr \
            #--config config-c \
            #--metrics fid50k,tpl \
            #--num-gpus 4 \
            #--model_type vc2_info_gan2 \
            #--C_lambda ${j} \
            #--fmap_decay 1 \
            #--random_seed 1000 \
            #--latent_type normal \
            #--norm_ord ${i} \
            #--n_dim_strict 1 \
            #--loose_rate 0.1 \
            #--gamma 10 \
            #--batch_size 128 \
            #--batch_per_gpu 32 \
            #--n_samples_per 7 \
            #--topk_dims_to_show 20 \
            #--I_fmap_base 8 \
            #--G_fmap_base 8 \
            #--G_nf_scale 4 \
            #--D_fmap_base 8 \
            #--fmap_min 32 \
            #--fmap_max 512 \
            #--module_list '[Const-512, ResConv-up-1, C_global-10, ResConv-id-1, Noise-2, ResConv-up-1, C_global-10, ResConv-id-1, Noise-2, ResConv-up-1, C_global-5, ResConv-id-1, Noise-2, ResConv-up-1, C_global-5, ResConv-id-1, Noise-2, ResConv-up-1, ResConv-id-2]'
            ##--module_list '[Const-256, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, ResConv-id-2]'
            ##--resume_pkl /project/xqzhu/repo_results/stylegan2/results_vc2_ffhq512/00014-vc2_gan-ffhq-2gpu-config-c/network-snapshot-000180.pkl \
    #done
#done

#python run_training_vc2.py \
    #--result-dir /project/xqzhu_dis/repo_results/stylegan2/results_vc2_byvae_celeba \
    #--data-dir /project/xqzhu/Datasets/CelebA_dataset \
    #--dataset celeba_tfr \
    #--resume_pkl /project/xqzhu_dis/repo_results/stylegan2/results_vc2_byvae_celeba/00017-vc2_gan_byvae-celeba_tfr-2gpu-config-c/network-snapshot-000120.pkl \
    #--config config-c \
    #--metrics fid50k,tpl \
    #--num-gpus 2 \
    #--model_type vc2_gan_byvae \
    #--C_lambda 0.01 \
    #--fmap_decay 1 \
    #--epsilon_loss 3 \
    #--random_seed 1000 \
    #--random_eps True \
    #--latent_type normal \
    #--delta_type onedim \
    #--gamma 100 \
    #--batch_size 8 \
    #--batch_per_gpu 4 \
    #--n_samples_per 7 \
    #--return_atts True \
    #--I_fmap_base 10 \
    #--G_fmap_base 9 \
    #--G_nf_scale 6 \
    #--D_fmap_base 10 \
    #--fmap_min 64 \
    #--fmap_max 512 \
    #--topk_dims_to_show -1 \
    #--module_list '[Const-512, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-6-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-6-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-8-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-8-5, ResConv-id-1, Noise-2, ResConv-id-3]'

#python run_training_vc2.py \
    #--result-dir /project/xqzhu_dis/repo_results/stylegan2/results_vc2_byvae_celeba \
    #--data-dir /project/xqzhu/Datasets/CelebA_dataset \
    #--dataset celeba_tfr \
    #--config config-c \
    #--arch orig \
    #--metrics fid50k,tpl \
    #--num-gpus 2 \
    #--model_type vc2_gan_byvae \
    #--C_lambda 1 \
    #--fmap_decay 1 \
    #--epsilon_loss 3 \
    #--random_seed 1000 \
    #--random_eps True \
    #--latent_type normal \
    #--delta_type onedim \
    #--gamma 100 \
    #--batch_size 32 \
    #--batch_per_gpu 16 \
    #--n_samples_per 7 \
    #--return_atts True \
    #--I_fmap_base 9 \
    #--G_fmap_base 10 \
    #--G_nf_scale 6 \
    #--D_fmap_base 9 \
    #--fmap_min 64 \
    #--fmap_max 512 \
    #--topk_dims_to_show -1 \
    #--latent_split_ls_for_std_gen [2,2,2,2,2,2,2,2,2,2,2,2] \
    #--module_list '[STD_gen_sp-4-24]'

#CUDA_VISIBLE_DEVICES=0,1 \
    #python run_training_vc2.py \
    #--result-dir /project/xqzhu_dis/repo_results/stylegan2/results_vc2_byvae_celeba \
    #--data-dir /project/xqzhu/Datasets/CelebA_dataset \
    #--dataset celeba_tfr \
    #--config config-c \
    #--metrics fid50k,tpl \
    #--num-gpus 2 \
    #--model_type vc2_gan_byvae \
    #--C_lambda 0.02 \
    #--fmap_decay 1 \
    #--epsilon_loss 3 \
    #--random_seed 1000 \
    #--random_eps True \
    #--latent_type normal \
    #--delta_type onedim \
    #--gamma 100 \
    #--batch_size 32 \
    #--batch_per_gpu 16 \
    #--n_samples_per 7 \
    #--return_atts True \
    #--I_fmap_base 9 \
    #--G_fmap_base 10 \
    #--G_nf_scale 6 \
    #--D_fmap_base 9 \
    #--fmap_min 64 \
    #--fmap_max 512 \
    #--topk_dims_to_show -1 \
    #--latent_split_ls_for_std_gen [2,2,2,2,2,2,2,2,2,2] \
    #--module_list '[PG_gen_sp-4-20]'

# Good one. 00025-vc2_gan_byvae-celeba_tfr-2gpu-config-c
CUDA_VISIBLE_DEVICES=0,1 \
    python run_training_vc2.py \
    --result-dir /project/xqzhu_dis/repo_results/stylegan2/results_vc2_byvae_celeba \
    --data-dir /project/xqzhu/Datasets/CelebA_dataset \
    --dataset celeba_tfr \
    --config config-c \
    --metrics fid50k,tpl \
    --num-gpus 2 \
    --model_type vc2_gan_byvae \
    --C_lambda 0.01 \
    --fmap_decay 1 \
    --epsilon_loss 3 \
    --random_seed 1000 \
    --random_eps True \
    --latent_type normal \
    --delta_type onedim \
    --gamma 100 \
    --batch_size 8 \
    --batch_per_gpu 4 \
    --n_samples_per 7 \
    --return_atts True \
    --I_fmap_base 10 \
    --G_fmap_base 9 \
    --G_nf_scale 6 \
    --D_fmap_base 10 \
    --fmap_min 64 \
    --fmap_max 512 \
    --topk_dims_to_show -1 \
    --module_list '[Const-512, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-6-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-6-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-6-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-id-2]'
