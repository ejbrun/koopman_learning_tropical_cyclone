
python train_model_KNF.py --num_epochs=30 --year_range=1990,2010

python train_model_koopman_kernel.py --model=Nystroem_RRR --koopman_kernel_num_train_stops=10 --year_range=1990,2010
python train_model_koopman_kernel.py --model=Randomized_RRR --koopman_kernel_num_train_stops=10 --year_range=1990,2010
python train_model_koopman_kernel.py --model=RRR --koopman_kernel_num_train_stops=10 --year_range=1990,2010

