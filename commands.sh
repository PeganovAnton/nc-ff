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