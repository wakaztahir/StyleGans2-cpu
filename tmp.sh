# j=0.1, sp=4, n_code=9, 1code1layer, gamma=1: 4:
#for i in 5 6 7 8 9 10 3 4 
#do
    #for j in 0.1 0.01 0.5
    #do
        #for k in 1 10 100
        #do
            #python run_training_vc2.py \
                #--result-dir /project/xqzhu/repo_results/stylegan2/results_vc2_dsp_all_byvae \
                #--data-dir /project/xqzhu/disentangle_datasets/dsprites \
                #--dataset dsprites_all_noshuffle_nolabel_tfr \
                #--config config-c \
                #--total-kimg 15000 \
                #--metrics tpl,factorvae_dsprites_all_hpc \
                #--num-gpus 4 \
                #--model_type vc2_gan_byvae \
                #--C_lambda ${j} \
                #--fmap_decay 1 \
                #--epsilon_loss 3 \
                #--random_eps True \
                #--latent_type normal \
                #--delta_type onedim \
                #--gamma ${k} \
                #--batch_size 128 \
                #--batch_per_gpu 32 \
                #--return_atts True \
                #--random_seed ${i} \
                #--I_fmap_base 8 \
                #--G_fmap_base 7 \
                #--G_nf_scale 4 \
                #--D_fmap_base 8 \
                #--fmap_min 32 \
                #--fmap_max 512 \
                #--module_list '[Const-512, ResConv-up-1, C_spgroup-4-1, ResConv-id-1, Noise-1, C_spgroup-4-1, ResConv-id-1, Noise-1, C_spgroup-4-1, ResConv-id-1, Noise-1, ResConv-up-1, C_spgroup-4-1, ResConv-id-1, Noise-1, C_spgroup-4-1, ResConv-id-1, Noise-1, C_spgroup-4-1, ResConv-id-1, Noise-1, ResConv-up-1, C_spgroup-4-1, ResConv-id-1, Noise-1, C_spgroup-4-1, ResConv-id-1, Noise-1, C_spgroup-4-1, ResConv-id-1, Noise-1, ResConv-up-1, ResConv-id-2]'
        #done
    #done
#done

# j=0.01, sp=8, n_code=7, 1code1layer, gamma=1: 5:
for i in 5 6 7 8 9 10 3 4 
do
    for j in 0.01 0.1 0.001
    do
        for k in 1
        do
            python run_training_vc2.py \
                --result-dir /project/xqzhu_dis/repo_results/stylegan2/results_vc2_dsp_all_byvae \
                --data-dir /project/xqzhu/disentangle_datasets/dsprites \
                --dataset dsprites_all_noshuffle_nolabel_tfr \
                --config config-c \
                --learning_rate 0.002 \
                --total-kimg 15000 \
                --metrics tpl,factorvae_dsprites_all_hpc \
                --num-gpus 4 \
                --model_type vc2_gan_byvae \
                --C_lambda ${j} \
                --fmap_decay 1 \
                --epsilon_loss 3 \
                --random_eps True \
                --latent_type normal \
                --delta_type onedim \
                --gamma ${k} \
                --batch_size 256 \
                --batch_per_gpu 64 \
                --return_atts False \
                --random_seed ${i} \
                --I_fmap_base 8 \
                --G_fmap_base 7 \
                --G_nf_scale 4 \
                --D_fmap_base 8 \
                --fmap_min 32 \
                --fmap_max 512 \
                --module_list '[Const-512, ResConv-up-1, C_global-2, C_global-2, Noise-1, ResConv-up-1, C_global-2, C_global-2, Noise-1, ResConv-up-1, Noise-1, ResConv-up-1, ResConv-id-1]'
        done
    done
done
