#!/bin/bash

echo "Run all the the: modelsync" >> $1

echo "char avx" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_AVX_CHAR --beta=0.001 --epochs=20 --stepinitial=0.00085 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "char scalar" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_SCALAR_CHAR --beta=0.001 --epochs=20 --stepinitial=0.00085 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "short avx" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_AVX_SHORT --beta=0.001 --epochs=20 --stepinitial=0.00085 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "short scalar" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_SCALAR_SHORT --beta=0.001 --epochs=20 --stepinitial=0.00085 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "FP avx" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_AVX_FP --beta=0.001 --epochs=20 --stepinitial=0.00085 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "FP scalar" >> $1
./bin/LINREG_HOGWILD_NON_DENSE_SCALAR_FP --beta=0.001 --epochs=20 --stepinitial=0.00085 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=8  --dimension=47236 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1


