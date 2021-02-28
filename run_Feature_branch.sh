#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=4

WIDTH=128
HEIGHT=128
L_DIM=64  # small-model: 64; middle-model: 128; large-model: 384
M_DIM=64  # small-model: 64; middle-model: 128; large-model: 256
H_DIM=64  # small-model: 64; middle-model: 128; large-model: 512

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    Feature_branch.py --launcher pytorch \
    --branch-1-dim ${L_DIM} --branch-m-dim ${M_DIM} --branch-h-dim ${H_DIM} \
    --width ${WIDTH} --height ${HEIGHT}
