
#####  Test with a single GPU

```shell
python tools/test_video.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] --eval segm
e.g.,
python ./tools/test_video.py ./configs/sgnet/sgnet_r50_caffe_fpn_gn_1x.py ./work_dirs/sgnet_r50_fpn_1x.pth --out results.pkl --eval segm
```
If you want to save the results of video instance segmentation, please use the following command:
```shell
python tools/test_video.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] --eval segm --show --save_path= ${SAVE_PATH}
```
