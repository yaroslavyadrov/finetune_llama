#!/bin/sh

# Use command line arguments if provided; otherwise, use default values
GPU_ID=${1:-1}
FREQ=${2:-10}

#nvidia-smi -i $GPU_ID -l $FREQ --query-gpu=name,index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv
nvidia-smi -i $GPU_ID -l $FREQ --query-gpu=name,index,memory.used,temperature.gpu,power.draw,fan.speed,clocks.gr,clocks.mem,utilization.memory,utilization.gpu --format=csv

