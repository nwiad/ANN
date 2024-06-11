# README

## 操作指南

包含3 层和 12 层的预训练模型的文件夹model_3_layers 和 model_12_layers 应该放置在/codes/pretrained/下。

运行 run.sh 即可执行所有实验。

## main.py 修改说明

引入了 wandb 模块用以绘图。

对微调过程进行了修改，从而可以选取预训练模型的第 1、6、12 层进行训练，并会在训练后马上进行测试。

将 test perplexity 和 BLEU-4 指标也记录在 output 文件中。