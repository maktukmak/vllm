#!/bin/sh

BATCH_SIZE=16

for BATCH_SIZE in 16 64 256 1024;
do

    INP_LEN=16
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN

    INP_LEN=64
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN

    INP_LEN=256
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN

    INP_LEN=1024
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN

    python generate_figure.py $BATCH_SIZE

done