#!/bin/bash

echo "Run all the stratedies: hogwill, modelsync"    #output file name.....

cd hogwild; 

./run.sh hogwild_log.txt

cd ..

cd modelsync

./run.sh modelsync_log.txt

cd ..
