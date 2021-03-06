#!/usr/bin/env bash

# Сбор результатов эксперимента l5_bn_l2_adam в одну таблицу.
cd ~/nc-ff/results/conv
python3 ../../results_to_data_array.py l5_bn_l2_adam/ \
    "^\./lr(0\.0*[13])/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d learning_rate launch_number metric step


# Сбор результатов эксперимента l5_bn_l2_lr0.01_sgd в одну таблицу.
cd ~/nc-ff/results/conv
python3 ../../results_to_data_array.py l5_bn_l2_lr0.01_sgd/ \
    "^\./l2_([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d lambda launch_number metric step


# Сбор результатов эксперимента l5_l2_sgd в одну таблицу.
cd ~/nc-ff/results/conv
python3 ../../results_to_data_array.py l5_l2_sgd/ \
    "^\./lr(0\.0*[13])/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d learning_rate launch_number metric step


# Сбор результатов эксперимента adam lr 0.001 при варьировании энтропии.
cd ~/nc-ff/results/conv
python3 ../../results_to_data_array.py unbalanced_ds_entropy/adam_lr0.001 \
    "^\./en_([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d entropy launch_number metric step


cd ~/nc-ff/results/conv
python3 ../../results_to_data_array.py unbalanced_ds_entropy/en_1.829_adam/ \
    "^\./lr([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d learning_rate launch_number metric step


# NO BIAS
# Сбор результатов эксперимента adam lr 0.001 при варьировании энтропии.
cd ~/nc-ff/results/conv
python3 ../../results_to_data_array.py \
    unbalanced_ds_entropy/adam_lr0.0001_no_bias \
    "^\./en_([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d entropy launch_number metric step

# fully connected 4 layers
cd ~/nc-ff/results/ff
python3 ../../results_to_data_array.py adam4/ \
    "^\./lr([0-9\.\-e+]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d learning_rate launch_number metric step


# ff unbalanced
# Сбор результатов эксперимента adam lr 0.001 при варьировании энтропии.
cd ~/nc-ff/results/ff
python3 ../../results_to_data_array.py unbalanced_ds_entropy/adam4_lr0.0001 \
    "^\./en_([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d entropy launch_number metric step
python3 ../../results_to_data_array.py unbalanced_ds_entropy/en_1.829_adam4/ \
    "^\./lr([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d learning_rate launch_number metric step
python3 ../../results_to_data_array.py unbalanced_ds_entropy/sgd4_lr0.001 \
    "^\./en_([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d entropy launch_number metric step
python3 ../../results_to_data_array.py unbalanced_ds_entropy/en_1.829_sgd4/ \
    "^\./lr([0-9\.]*)/([0-9])/\w*/(?:(?:valid/(\w*).*)|(?:(\w*).pickle))$" \
    -t float int str -d learning_rate launch_number metric step