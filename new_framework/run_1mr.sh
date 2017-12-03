#!/bin/bash

for i in 14 27
do
  echo "ModelSync_bitweaving with memory regions: 1. Step size: 0.81, threads: " $i >> $1
  for j in  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
  do
    echo "Bits: " $j >>$1
    echo "Bits: " $j
    sync 
    echo 3 > /proc/sys/vm/drop_caches
    sudo ./bin/LINREG_HOGWILD_NON_DENSE_AVX_INT_BIND --beta=0.001 --epochs=40 --stepinitial=0.81 --step_decay=0.98 --class_model=1 --target_label=1 --batch_size=10 --splits=$i --bits_per_mr=32  --pcm_epoch=2 --model=0 --num_bits=$j ../../data/data_4G_4k_1mr.dat ../../data/data_4G_4k_1mr.dat >> $1
  done	
done





