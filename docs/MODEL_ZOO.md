## Model Zoo

**Note:** the following models are not the original implementation from the paper, in order to accommodate small computation, we use channel dimensions of 64*3 and feature sizes of 128/2 X 128/2. We encourage you to change the channel dimensions and feature sizes to increase the performance.

|   Model   |  Trained on  |   Tested on    |  Recall@1    |  Recall@5    |  Recall@10   | Download Link |
| :--------: | :---------: | :-----------: | :----------: | :----------: | :----------: | :----------: |
| small-model (for one V100 GPU) | Pitts30k-train | Pitts30k-test | 63.7%  |   78.3% |  83.7%  | [Google Drive](https://drive.google.com/drive/folders/1UpU6o3FwZL5sx__VaO79ASOvkWfPTmYD?usp=sharing) |
| middle-model (for both efficiency and performance) | Pitts30k-train | Pitts30k-test |    67.0%   |    81.5%   |    86.3%   |    [Google Drive](https://drive.google.com/drive/folders/1Z4_xPvLH6XG9SAuhlmDpwE-FwhlxpIjO?usp=sharing)   |
| large-model (the proposed one in paper) | Pitts30k-train | Pitts30k-test |       |       |       |       |
