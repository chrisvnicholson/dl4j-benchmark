#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/CNNMnist/tensorflow_lenet.py ;;
    'GPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/CNNMnist/tensorflow_lenet.py --num_gpus 1 --core_type "GPU";;
    'MULTI' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/CNNMnist/tensorflow_lenet.py --num_gpus 4 --core_type "MULTI";;
    *) echo "Invalid response";;
esac