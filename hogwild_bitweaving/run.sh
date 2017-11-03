#!/bin/bash

echo "Run all the the: Hogwild with 27 threads... Step size: 1" >> $1

echo "1-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=1 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "2-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=2 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "3-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=3 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "4-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=4 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "5-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=5 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "6-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=6 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "7-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=7 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "8-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=8 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "9-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=9 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "10-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=10 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "11-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=11 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "12-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=12 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "13-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=13 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "14-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=14 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "15-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=15 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "16-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=27 --pcm_epoch=2  --dimension=47236 --num_bits=16 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1



echo "Run all the the: Hogwild with 14 threads... Step size: 1" >> $1

echo "1-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=1 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "2-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=2 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "3-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=3 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "4-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=4 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "5-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=5 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "6-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=6 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "7-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=7 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "8-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=8 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "9-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=9 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "10-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=10 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "11-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=11 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "12-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=12 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "13-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=13 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "14-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=14 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "15-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=15 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1

echo "16-bit" >> $1
sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=14 --pcm_epoch=2  --dimension=47236 --num_bits=16 ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1



