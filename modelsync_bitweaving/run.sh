#!/bin/bash

for i in 14 27
do
  echo "ModelSync_bitweaving: Step size: 1.41, threads: " $i >> $1
  for j in  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
  do
    sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=1.41 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=$i --pcm_epoch=2  --dimension=47236 --num_bits=$j ../../data/rcv1_train.tsv ../../data/rcv1_train.tsv  >>$1  
  done	
done


