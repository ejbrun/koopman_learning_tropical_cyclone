


# python train_model_koopman_kernel.py --model=RRR --koopman_kernel_num_train_stops=5 --year_range=1990,2000
# python train_model_koopman_kernel.py --model=Randomized_RRR --koopman_kernel_num_train_stops=5 --year_range=1990,2010
# python train_model_koopman_kernel.py --model=Nystroem_RRR --koopman_kernel_num_train_stops=5 --year_range=1990,2010


koopman_kernel_length_scale=(10)
koopman_kernel_rank=(45)
koopman_kernel_num_centers=(350)

for kk_length_scale in ${koopman_kernel_length_scale[@]};
do
    for kk_krank in ${koopman_kernel_rank[@]};
    do
        for kk_ncenters in ${koopman_kernel_num_centers[@]};
        do
            python train_model_koopman_kernel_save_model.py --model=Nystroem_RRR --koopman_kernel_num_train_stops=10 --year_range=1970,2020 --koopman_kernel_length_scale=$kk_length_scale --koopman_kernel_rank=$kk_krank --koopman_kernel_num_centers=$kk_ncenters
        done
    done
done