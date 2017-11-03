#!/bin/bash

echo "Run all the stratedies: hogwill, modelsync hogwill_bitweaving, modelsync_bitweaving"    #output file name.....

cd hogwild; 
echo "Enter hogwill"   
./run.sh hogwild_log_tune_step_size.txt
cd ..
echo "Leave hogwill"    

cd modelsync
echo "Enter modelsync"   
./run.sh modelsync_log_tune_step_size.txt
cd ..
echo "Leave modelsync"  


cd hogwild_bitweaving; 
echo "Enter hogwill_bitweaving"   
./run.sh hogwild_bitweaving_log_tune_step_size.txt
cd ..
echo "Leave hogwill_bitweaving"    

cd modelsync_bitweaving
echo "Enter modelsync_bitweaving"   
./run.sh modelsync_bitweaving_log_tune_step_size.txt
cd ..
echo "Leave modelsync_bitweaving"  


