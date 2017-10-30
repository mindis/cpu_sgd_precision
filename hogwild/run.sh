#!/bin/bash

echo "Run all the the: Hogwild" >> $1

echo "char avx" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_AVX_CHAR_BIND --beta=0.001 --epochs=40 --stepinitial=10.1785 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "short avx" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_AVX_SHORT_BIND --beta=0.001 --epochs=40 --stepinitial=10.1785 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "FP avx" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_AVX_FP_BIND --beta=0.001 --epochs=40 --stepinitial=10.1785 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1



