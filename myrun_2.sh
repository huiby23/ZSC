#!/bin/bash
for seed in 0 1 2 3 4 5 6 7 8 9
do
python -u selfplay.py --save_dir models/vdn/seed${seed} --num_thread 32 --num_game_per_thread 80 --method vdn --sad 0 --lr 6.25e-05 --eps 1.5e-05 --gamma 0.999 --seed ${seed} --burn_in_frames 10000 --replay_buffer_size 80000 --batchsize 128 --epoch_len 1000 --act_device cuda:1 --train_device cuda:1 --num_epoch 400 --num_player 2 --net lstm --num_lstm_layer 2 --multi_step 3
done
