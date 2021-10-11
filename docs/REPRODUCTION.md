
#####  Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
e.g.,
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/sgnet/sgnet_r50_caffe_fpn_gn_1x.py 4
```

