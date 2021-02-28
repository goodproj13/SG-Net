#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=4

RESUME=${1-'pretrained_model/model_best.pth'}
ARCH=vgg16

DATASET=${2-pitts}
SCALE=${3-30k}

IMG=${4-'sample/image.jpg'}

L_DIM=64  # small-model: 64; middle-model: 128; large-model: 384
M_DIM=64  # small-model: 64; middle-model: 128; large-model: 256
H_DIM=64  # small-model: 64; middle-model: 128; large-model: 512

# if [ $# -lt 1 ]
#   then
#     echo "Arguments error: <MODEL PATH>"
#     echo "Optional arguments: <DATASET (default:pitts)> <SCALE (default:250k)>"
#     exit 1
# fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
inference.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --branch-1-dim ${L_DIM} --branch-m-dim ${M_DIM} --branch-h-dim ${H_DIM} \
    --test-batch-size 8 -j 16 \
    --vlad --reduction \
    --resume ${RESUME} \
    --img-path ${IMG} \
    --sync-gather
