## Inference

Please download the trained model weights and PCA parameters from [Here](docs/MODEL_ZOO.md) and put them under the `pretrained_model/` directory and run:
```bash
python inference.py
```
It runs inference on `sample/image.jpg` and save predicted features and top-k images under `result/`. **Note:** `sample/image.jpg` should be in the test split of the dataset.

The default scripts adopt `4` Tesla V100 GPUs (require ~32G per GPU) for inference.
+ In case you want to fasten training, enlarge `GPUS` for more GPUs, or enlarge the `--test-batch-size` for larger batch size on one GPU;
+ In case your GPU does not have enough memory (e.g. <32G), reduce `--test-batch-size` for smaller batch size on one GPU.