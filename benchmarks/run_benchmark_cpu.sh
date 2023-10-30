#!/bin/sh

export VLLM_CPU_ONLY=1
BATCH_SIZE=16
NUM_ITERS=1
NUM_TOKENS=128

for BATCH_SIZE in 16 64 256 1024;
do

    INP_LEN=16
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS

    INP_LEN=64
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS

    INP_LEN=256
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS

    INP_LEN=1024
    python benchmark_latency_cpu.py --batch-size 1 --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS
    python benchmark_latency_cpu.py --batch-size $BATCH_SIZE --input-len $INP_LEN --num-iters $NUM_ITERS --output-len $NUM_TOKENS

    python generate_figure.py $BATCH_SIZE

done