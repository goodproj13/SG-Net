#!/bin/sh
ARCH=vgg16 # $1
L_DIM=64  # small-model: 64; middle-model: 128; large-model: 384
M_DIM=64  # small-model: 64; middle-model: 128; large-model: 256
H_DIM=64  # small-model: 64; middle-model: 128; large-model: 512

# if [ $# -ne 1 ]
#   then
#     echo "Arguments error: <ARCH>"
#     exit 1
# fi

python -u cluster.py -d pitts -a ${ARCH} -b 64 --width 640 --height 480 \
    --branch-1-dim ${L_DIM} --branch-m-dim ${M_DIM} --branch-h-dim ${H_DIM} 
