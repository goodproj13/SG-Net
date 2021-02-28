#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=4

DATASET=pitts
SCALE=30k
ARCH=vgg16
LAYERS=conv5
LOSS=sare_joint
LR=0.001
WIDTH=128  # 640
HEIGHT=128 # 480
L_DIM=64  # small-model: 64; middle-model: 128; large-model: 384
M_DIM=64  # small-model: 64; middle-model: 128; large-model: 256
H_DIM=64  # small-model: 64; middle-model: 128; large-model: 512

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        echo "find PORT"
        break;
    fi
done

# CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
# netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
#   -d ${DATASET} --scale ${SCALE} \
#   -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
#   --width 640 --height 480 --tuple-size 1 -j 2 --neg-num 4 --test-batch-size 4 \
#   --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
#   --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
#   --logs-dir logs/saved_models/${DATASET}${SCALE}-${ARCH}/${LAYERS}-${LOSS}-lr${LR}-tuple${GPUS}

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --branch-1-dim ${L_DIM} --branch-m-dim ${M_DIM} --branch-h-dim ${H_DIM} \
  --width ${WIDTH} --height ${HEIGHT} --tuple-size 1 -j 4 --neg-num 4 --test-batch-size 4 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 10 --step-size 5 --cache-size 1000 \
  --logs-dir logs/saved_models/${DATASET}${SCALE}-${ARCH}/${LAYERS}-${LOSS}-lr${LR}-tuple${GPUS}-${WIDTH}x${HEIGHT}-${L_DIM}-${M_DIM}-${H_DIM}
