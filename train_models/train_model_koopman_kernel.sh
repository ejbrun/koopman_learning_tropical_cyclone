


# python train_model_koopman_kernel.py --model=RRR --koopman_kernel_num_train_stops=5 --year_range=1990,2000
# python train_model_koopman_kernel.py --model=Randomized_RRR --koopman_kernel_num_train_stops=5 --year_range=1990,2010
# python train_model_koopman_kernel.py --model=Nystroem_RRR --koopman_kernel_num_train_stops=5 --year_range=1990,2010



# koopman_kernel_length_scale=(0.01 0.1 1 10)
koopman_kernel_length_scale=(0.5 1 5 10 50 100 500)
koopman_kernel_rank=(5 25 45)
koopman_kernel_num_centers=(100 150 200 250 300 350)

kk_num_train_stops=10
tikhonov_reg=1e-8
basins=("EP" "NA" "SI" "SP" "WP")

for kk_length_scale in ${koopman_kernel_length_scale[@]};
do
    for kk_krank in ${koopman_kernel_rank[@]};
    do
        for kk_ncenters in ${koopman_kernel_num_centers[@]};
        do
            for basin in ${basins[@]};
            do
                python train_model_koopman_kernel.py --model=Nystroem_RRR --basin=$basin --year_range=1980,2021 --koopman_kernel_length_scale=$kk_length_scale --koopman_kernel_rank=$kk_krank --koopman_kernel_num_centers=$kk_ncenters --koopman_kernel_num_train_stops=$kk_num_train_stops --tikhonov_reg=$tikhonov_reg
            done
        done
    done
done
